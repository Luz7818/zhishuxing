/* ============================================================
   智枢星 Web 控制台交互 v2
   视图路由 / API 客户端 / Canvas 网格与图表 / 全流程反馈
   ============================================================ */

"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- API 客户端 ---------------- */

async function request(url, opts) {
  const response = await fetch(url, opts);
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok || (payload && payload.ok === false)) {
    throw new Error((payload && payload.error) || `请求失败: ${response.status}`);
  }
  return payload ? payload.data : null;
}

const api = {
  get: (url) => request(url),
  post: (url, body) =>
    request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }),
};

/* ---------------- 反馈组件 ---------------- */

function toast(message, type = "ok") {
  const host = $("toastHost");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${type === "err" ? "⚠️" : "✅"}</span><span></span>`;
  el.lastElementChild.textContent = message;
  host.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 320);
  }, 2800);
}

async function withBusy(btn, runningText, fn) {
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>${runningText}`;
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

function showRaw(preId, data) {
  $(preId).textContent = JSON.stringify(data, null, 2);
}

function setImage(imgId, frameId, url) {
  const img = $(imgId);
  img.src = `${url}?t=${Date.now()}`;
  $(frameId).classList.add("show");
}

/* ---------------- 视图路由 ---------------- */

const VIEW_META = {
  overview: ["概览", "系统状态总览与快捷入口"],
  navigation: ["枢纽导航", "网格可视化 · 点击选取起终点 · A* 路径规划"],
  rl: ["RL 智能体", "MADDPG 策略权重 · 推理 · 多智能体引导仿真"],
  flow: ["客流面板", "乘客分组 · 动态客流热力与引导路径"],
  plan: ["路线规划", "自然语言诉求 → 真实换乘方案"],
  llm: ["LLM 引擎", "模型加载 · 微调接口 · 指标报告"],
  reports: ["分析报告", "七类可视化报告一键生成"],
};

function switchView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(`view-${name}`).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === name));
  const meta = VIEW_META[name] || [name, ""];
  $("viewTitle").textContent = meta[0];
  $("viewSub").textContent = meta[1];
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

/* ---------------- Canvas 工具 ---------------- */

const LANDMARK_ALIAS = {
  entry_a: "A口进站",
  entry_b: "B口进站",
  security: "主安检",
  security_backup: "备用安检",
  metro_gate: "地铁闸机",
  rail_gate: "高铁闸机",
  bus_gate: "公交闸机",
};

const lmName = (key) => LANDMARK_ALIAS[key] || key;

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawGrid(canvas, grid, view) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cell = Math.min((W - 30) / grid.width, (H - 30) / grid.height);
  const ox = (W - cell * grid.width) / 2;
  const oy = (H - cell * grid.height) / 2;

  // 底色网格
  ctx.fillStyle = "#0a1224";
  ctx.fillRect(ox, oy, cell * grid.width, cell * grid.height);
  ctx.strokeStyle = "rgba(126,168,220,0.10)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= grid.width; x++) {
    ctx.beginPath();
    ctx.moveTo(ox + x * cell, oy);
    ctx.lineTo(ox + x * cell, oy + cell * grid.height);
    ctx.stroke();
  }
  for (let y = 0; y <= grid.height; y++) {
    ctx.beginPath();
    ctx.moveTo(ox, oy + y * cell);
    ctx.lineTo(ox + cell * grid.width, oy + y * cell);
    ctx.stroke();
  }

  // 阻挡区域
  ctx.fillStyle = "#33477a";
  grid.blocked.forEach(([bx, by]) => {
    roundRect(ctx, ox + bx * cell + 1, oy + by * cell + 1, cell - 2, cell - 2, 3);
    ctx.fill();
  });

  // 地标
  Object.entries(grid.landmarks).forEach(([key, [lx, ly]]) => {
    const cx = ox + lx * cell + cell / 2;
    const cy = oy + ly * cell + cell / 2;
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(4, cell * 0.22), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#dce6f7";
    ctx.font = `${Math.max(10, cell * 0.52)}px "Microsoft YaHei"`;
    ctx.textAlign = "center";
    ctx.fillText(lmName(key), cx, cy - cell * 0.55);
  });

  // 路径（支持动画进度）
  const route = (view && view.route) || [];
  if (route.length > 1) {
    const progress = view && view.progress !== undefined ? view.progress : 1;
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = Math.max(3, cell * 0.28);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.shadowColor = "rgba(56,189,248,0.7)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    const upto = 1 + Math.floor((route.length - 1) * progress);
    for (let i = 0; i < upto; i++) {
      const [px, py] = route[i];
      const x = ox + px * cell + cell / 2;
      const y = oy + py * cell + cell / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // 起点 / 终点
  const marker = (point, color, label) => {
    if (!point) return;
    const cx = ox + point[0] * cell + cell / 2;
    const cy = oy + point[1] * cell + cell / 2;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(6, cell * 0.34), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.max(10, cell * 0.5)}px "Microsoft YaHei"`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, cx, cy + 0.5);
    ctx.textBaseline = "alphabetic";
  };
  marker(view && view.start, "#22d3ee", "起");
  marker(view && view.goal, "#34d399", "终");
}

const CHART_COLORS = ["#38bdf8", "#f59e0b", "#34d399", "#a78bfa", "#f87171"];

function drawLineChart(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const pad = { l: 52, r: 14, t: 14, b: 30 };
  const allY = series.flatMap((s) => s.y);
  if (!allY.length) return;
  let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...allY);
  let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...allY);
  if (yMax - yMin < 1e-9) yMax = yMin + 1;
  const xMax = Math.max(...series.map((s) => (s.x && s.x.length ? s.x[s.x.length - 1] : s.y.length - 1)));
  const xMin = 0;

  const toPx = (xi, yi) => [
    pad.l + ((xi - xMin) / Math.max(xMax - xMin, 1e-9)) * (W - pad.l - pad.r),
    H - pad.b - ((yi - yMin) / (yMax - yMin)) * (H - pad.t - pad.b),
  ];

  ctx.strokeStyle = "rgba(126,168,220,0.12)";
  ctx.fillStyle = "#6e7fa0";
  ctx.font = '11px "Microsoft YaHei"';
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + ((yMax - yMin) * i) / 4;
    const [, py] = toPx(xMin, yv);
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(W - pad.r, py);
    ctx.stroke();
    ctx.fillText(yv.toFixed(opts.yDecimals !== undefined ? opts.yDecimals : 1), pad.l - 8, py + 4);
  }

  series.forEach((s, si) => {
    const color = s.color || CHART_COLORS[si % CHART_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const n = s.y.length;
    for (let i = 0; i < n; i++) {
      const xi = s.x ? s.x[i] : i;
      const [px, py] = toPx(xi, s.y[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });
}

function ctxClear(canvas) {
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

function drawChartEmpty(canvas, text) {
  ctxClear(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#6e7fa0";
  ctx.font = '13px "Microsoft YaHei"';
  ctx.textAlign = "center";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

function drawPolyline(canvas, coords, labels = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!coords || coords.length < 2) return;

  const lons = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  const pad = 36;
  const lonMin = Math.min(...lons);
  const lonMax = Math.max(...lons);
  const latMin = Math.min(...lats);
  const latMax = Math.max(...lats);
  const scale = Math.min(
    (W - pad * 2) / Math.max(lonMax - lonMin, 1e-9),
    (H - pad * 2) / Math.max(latMax - latMin, 1e-9)
  );
  const toXY = ([lon, lat]) => [
    pad + (lon - lonMin) * scale + (W - pad * 2 - (lonMax - lonMin) * scale) / 2,
    H - pad - (lat - latMin) * scale - (H - pad * 2 - (latMax - latMin) * scale) / 2,
  ];

  ctx.fillStyle = "#0a1224";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(56,189,248,0.6)";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  coords.forEach(([lon, lat], i) => {
    const [x, y] = toXY([lon, lat]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  const pin = (point, color, label) => {
    const [x, y] = toXY(point);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (label) {
      ctx.fillStyle = "#dce6f7";
      ctx.font = '12px "Microsoft YaHei"';
      ctx.textAlign = "center";
      ctx.fillText(label, x, y - 13);
    }
  };
  pin(coords[0], "#22d3ee", labels.origin || "起点");
  pin(coords[coords.length - 1], "#34d399", labels.destination || "终点");
}

/* ---------------- 全局状态 ---------------- */

const state = {
  nav: { grid: null, start: null, goal: null, via: new Set(), route: null, progress: 1 },
  plan: { strategy: "balanced", walkShort: false, crowdAvoid: false },
  llm: { preferReal: false },
};

/* ---------------- 顶栏 ---------------- */

async function refreshTopbar() {
  try {
    await api.get("/health");
    const pill = $("healthPill");
    pill.classList.add("online");
    pill.innerHTML = '<span class="dot"></span>服务在线';
  } catch (_) {
    const pill = $("healthPill");
    pill.classList.add("offline");
    pill.innerHTML = '<span class="dot"></span>服务离线';
  }

  const status = await api.get("/api/rl/status").catch(() => null);
  if (status) {
    updatePolicyUI(status);
  }
}

function updatePolicyUI(status) {
  const badge = $("policyBadge");
  badge.className = `badge ${status.policy_loaded ? "ok" : "warn"}`;
  badge.innerHTML = `<span class="dot"></span>策略：${status.policy_source}`;

  $("kpiPolicy").textContent = status.policy_source;
  $("kpiSteps").textContent = status.latest_step_k ? `${status.latest_step_k}k` : "—";
  $("kpiStepsFoot").textContent = status.latest_step_k ? `检查点 step_${status.latest_step_k}k` : "检查点加载后显示";
  $("kpiCheckpoints").textContent = status.checkpoint_count;
  $("kpiRewards").textContent = status.reward_files.length;
  $("stTorch").textContent = status.torch_available ? `可用 · ${status.torch_version}` : "未安装（启发式回退）";

  const table = $("rlCkptTable");
  const empty = $("rlCkptEmpty");
  if (status.checkpoints.length) {
    table.style.display = "";
    empty.style.display = "none";
    $("rlCkptBody").innerHTML = status.checkpoints
      .map(
        (c) =>
          `<tr><td class="strong">agent_${c.agent_id}</td><td>${c.algorithm}</td><td>step_${c.step_k}k</td><td>${c.path.split(/[\\/]/).pop()}</td></tr>`
      )
      .join("");
  } else {
    table.style.display = "none";
    empty.style.display = "";
  }

  const rlBadge = $("rlSourceBadge");
  if (rlBadge) {
    rlBadge.className = `badge ${status.policy_loaded ? "ok" : "warn"}`;
    rlBadge.textContent = status.policy_source;
  }
}

/* ---------------- 概览 ---------------- */

function initOverview() {
  $("btnQuickSim").addEventListener("click", () => {
    switchView("rl");
    $("btnRLSim").click();
  });
  $("btnQuickPanel").addEventListener("click", () => {
    switchView("flow");
    $("btnPanel").click();
  });
  $("btnQuickPolicy").addEventListener("click", () =>
    withBusy($("btnQuickPolicy"), " 加载中", async () => {
      await loadLatestPolicy();
      toast("策略权重已刷新");
    })
  );
  $("btnQuickReports").addEventListener("click", () => {
    switchView("reports");
    $("btnReportsAll").click();
  });
}

/* ---------------- 枢纽导航 ---------------- */

async function loadGrid() {
  state.nav.grid = await api.get("/api/navigation/grid");
  buildNavControls();
  repaintNav();
  $("stNav").textContent = state.nav.grid.file ? state.nav.grid.file.split(/[\\/]/).pop() : "已加载";
}

function buildNavControls() {
  const grid = state.nav.grid;
  const options = Object.entries(grid.landmarks)
    .map(([key, [x, y]]) => `<option value="${x},${y}">${lmName(key)} (${x},${y})</option>`)
    .join("");
  $("navStart").innerHTML = `<option value="">— 点击网格选取 —</option>${options}`;
  $("navGoal").innerHTML = `<option value="">— 点击网格选取 —</option>${options}`;

  $("navStart").addEventListener("change", () => {
    const v = $("navStart").value;
    state.nav.start = v ? v.split(",").map(Number) : null;
    repaintNav();
  });
  $("navGoal").addEventListener("change", () => {
    const v = $("navGoal").value;
    state.nav.goal = v ? v.split(",").map(Number) : null;
    repaintNav();
  });

  const viaKeys = Object.keys(grid.landmarks).filter((k) => k.includes("security"));
  $("viaChips").innerHTML = viaKeys.map((k) => `<button class="chip" data-lm="${k}">${lmName(k)}</button>`).join("");
  $("viaChips")
    .querySelectorAll(".chip")
    .forEach((chip) =>
      chip.addEventListener("click", () => {
        const key = chip.dataset.lm;
        if (state.nav.via.has(key)) state.nav.via.delete(key);
        else state.nav.via.add(key);
        chip.classList.toggle("active");
      })
    );
}

function repaintNav() {
  drawGrid($("navCanvas"), state.nav.grid, state.nav);
}

function animateRoute() {
  const view = state.nav;
  view.progress = 0;
  const started = performance.now();
  const duration = 620;
  function frame(now) {
    view.progress = Math.min(1, (now - started) / duration);
    repaintNav();
    if (view.progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function initNavigation() {
  const canvas = $("navCanvas");
  canvas.addEventListener("click", (event) => {
    const grid = state.nav.grid;
    if (!grid) return;
    const rect = canvas.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) * canvas.width) / rect.width;
    const cy = ((event.clientY - rect.top) * canvas.height) / rect.height;
    const cell = Math.min((canvas.width - 30) / grid.width, (canvas.height - 30) / grid.height);
    const ox = (canvas.width - cell * grid.width) / 2;
    const oy = (canvas.height - cell * grid.height) / 2;
    const gx = Math.floor((cx - ox) / cell);
    const gy = Math.floor((cy - oy) / cell);
    if (gx < 0 || gy < 0 || gx >= grid.width || gy >= grid.height) return;
    if (grid.blocked.some(([bx, by]) => bx === gx && by === gy)) {
      toast("该格为阻挡区域，无法选择", "err");
      return;
    }

    const point = [gx, gy];
    const label = `网格 (${gx},${gy})`;
    if (!state.nav.start || (state.nav.start && state.nav.goal)) {
      state.nav.start = point;
      state.nav.goal = null;
      state.nav.route = null;
      $("navStart").insertAdjacentHTML("beforeend", `<option value="${point.join(",")}">${label}</option>`);
      $("navStart").value = point.join(",");
      $("navGoal").value = "";
    } else {
      state.nav.goal = point;
      $("navGoal").insertAdjacentHTML("beforeend", `<option value="${point.join(",")}">${label}</option>`);
      $("navGoal").value = point.join(",");
    }
    repaintNav();
  });

  $("btnNavReload").addEventListener("click", () =>
    withBusy($("btnNavReload"), " 加载中", async () => {
      await api.post("/api/navigation/load", { file_path: state.nav.grid.file });
      await loadGrid();
      toast("导航图已重新加载");
    })
  );

  $("btnNavPlan").addEventListener("click", () =>
    withBusy($("btnNavPlan"), " 规划中", async () => {
      if (!state.nav.start || !state.nav.goal) {
        toast("请先设定起点与终点", "err");
        return;
      }
      const result = await api.post("/api/navigation/plan", {
        start: state.nav.start,
        goal: state.nav.goal,
        via: [...state.nav.via],
      });
      state.nav.route = result.route;
      animateRoute();
      $("navResultEmpty").style.display = "none";
      $("navResult").style.display = "";
      $("navLen").textContent = `${result.length} 格`;
      $("navVia").textContent = state.nav.via.size ? [...state.nav.via].map(lmName).join("、") : "无";
      showRaw("navRaw", result.route);
      toast(`规划完成：路径长度 ${result.length} 格`);
    })
  );
}

/* ---------------- RL 智能体 ---------------- */

async function loadLatestPolicy() {
  const status = await api.post("/api/rl/load_policy", {});
  updatePolicyUI(status);
  return status;
}

async function refreshRewards() {
  const data = await api.get("/api/rl/rewards");
  const series = (data.series || []).map((s, i) => ({
    name: s.file.replace(/\.npy$/, ""),
    x: s.x,
    y: s.y,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));
  const canvas = $("rewardCanvas");
  if (!series.length) {
    drawChartEmpty(canvas, "暂无 *_env_*.npy 奖励数据（训练后自动出现）");
    $("rewardLegend").innerHTML = "";
    return;
  }
  drawLineChart(canvas, series);
  $("rewardLegend").innerHTML = series
    .map(
      (s, i) =>
        `<span class="key"><span class="swatch" style="background:${CHART_COLORS[i % CHART_COLORS.length]}"></span>${s.name}</span>`
    )
    .join("");
}

function initRL() {
  $("btnRLRefresh").addEventListener("click", () =>
    withBusy($("btnRLRefresh"), " 刷新中", async () => {
      updatePolicyUI(await api.get("/api/rl/status"));
      toast("状态已刷新");
    })
  );

  $("btnRLPolicy").addEventListener("click", () =>
    withBusy($("btnRLPolicy"), " 加载中", async () => {
      const status = await loadLatestPolicy();
      toast(
        status.policy_loaded ? `已加载 ${status.checkpoint_count} 个检查点` : status.load_error || "无权重，启发式回退",
        status.policy_loaded ? "ok" : "err"
      );
    })
  );

  $("btnRLRewards").addEventListener("click", () => withBusy($("btnRLRewards"), " 刷新中", refreshRewards));

  $("btnRLAct").addEventListener("click", () =>
    withBusy($("btnRLAct"), " 推理中", async () => {
      let observations;
      try {
        observations = JSON.parse($("rlObs").value);
      } catch (_) {
        throw new Error("观测 JSON 格式错误");
      }
      const result = await api.post("/api/rl/act", { observations });
      $("rlActResult").innerHTML = result.actions
        .map(
          (a, i) =>
            `<span class="badge info" style="margin:3px 6px 3px 0">agent_${i} → 前进 ${a[0].toFixed(3)} · 转向 ${a[1].toFixed(3)}（${result.policy_used[i]}）</span>`
        )
        .join("");
    })
  );

  $("btnRLSim").addEventListener("click", () =>
    withBusy($("btnRLSim"), " 仿真中", async () => {
      const result = await api.post("/api/rl/simulate", {
        config: {
          max_steps: Number($("simSteps").value) || 240,
          agents_per_group: Number($("simAgents").value) || 6,
          seed: Number($("simSeed").value) || 42,
        },
      });
      $("simEmpty").style.display = "none";
      $("simResult").style.display = "";
      const rate = Math.round((result.agents_arrived / result.agents_total) * 100);
      $("simArrive").textContent = `${rate}%`;
      $("simAvg").textContent = result.avg_transfer_steps ?? "未到达";
      $("simBase").textContent = result.free_flow_baseline_steps;
      $("simCong").textContent = result.congestion_peak;
      $("simColl").textContent = result.collision_events;
      $("simExec").textContent = result.steps_executed;
      $("simPolicy").textContent = result.policy_source;
      setImage("simImage", "simImageFrame", result.image_url);
      toast(`仿真完成：到达 ${result.agents_arrived}/${result.agents_total}`);
    })
  );
}

/* ---------------- 客流面板 ---------------- */

let flowGroups = [];

function renderGroupRows() {
  $("groupBody").innerHTML = flowGroups
    .map(
      (g, i) => `<tr>
        <td class="strong">${g.name}</td>
        <td>${g.start} → ${g.goal}${g.via_landmarks && g.via_landmarks.length ? `（经 ${g.via_landmarks.join("、")}）` : ""}</td>
        <td><input type="number" min="0" value="${g.release_time}" data-i="${i}" data-k="release_time" style="width:74px" /></td>
        <td><input type="number" min="1" value="${g.passengers}" data-i="${i}" data-k="passengers" style="width:74px" /></td>
      </tr>`
    )
    .join("");
  $("groupBody")
    .querySelectorAll("input")
    .forEach((input) =>
      input.addEventListener("change", () => {
        flowGroups[Number(input.dataset.i)][input.dataset.k] = Number(input.value);
      })
    );
}

function initFlow() {
  $("btnGroupReset").addEventListener("click", async () => {
    flowGroups = await api.get("/api/scenarios");
    renderGroupRows();
    toast("已恢复默认场景");
  });

  $("btnPanel").addEventListener("click", () =>
    withBusy($("btnPanel"), " 生成中", async () => {
      const result = await api.post("/api/dashboard/run", {
        groups: flowGroups,
        title: "智枢星 · 动态客流面板",
      });
      $("panelEmpty").style.display = "none";
      $("panelResult").style.display = "";
      $("panelGuidance").textContent = result.guidance;
      setImage("panelImage", "panelImage", result.image_url);
      $("panelMeta").textContent = `客流均值 ${result.flow_mean.toFixed(4)} · 峰值 ${result.flow_peak.toFixed(2)}`;
      toast("客流面板已生成");
    })
  );
}

/* ---------------- 路线规划 ---------------- */

function initPlan() {
  document.querySelectorAll("#strategyChips .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#strategyChips .chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.plan.strategy = chip.dataset.v;
    });
  });
  $("chipWalk").addEventListener("click", () => {
    state.plan.walkShort = !state.plan.walkShort;
    $("chipWalk").classList.toggle("active", state.plan.walkShort);
  });
  $("chipCrowd").addEventListener("click", () => {
    state.plan.crowdAvoid = !state.plan.crowdAvoid;
    $("chipCrowd").classList.toggle("active", state.plan.crowdAvoid);
  });

  $("btnRealPlan").addEventListener("click", () =>
    withBusy($("btnRealPlan"), " 规划中", async () => {
      const result = await api.post("/api/plan", {
        question: $("planQuestion").value.trim(),
        engine: $("planEngine").value,
        prefs: {
          strategy: state.plan.strategy,
          walk: state.plan.walkShort ? "short" : "normal",
          crowd: state.plan.crowdAvoid ? "avoid" : "normal",
        },
      });
      $("planEmpty").style.display = "none";
      $("planResult").style.display = "";
      $("planDur").textContent = result.duration_sec ? `${Math.round(result.duration_sec / 60)} 分` : "—";
      $("planCost").textContent = result.cost !== undefined && result.cost !== null ? `${result.cost} 元` : "—";
      $("planSeg").textContent = result.segments !== undefined ? `${result.segments} 次` : "—";
      $("planEngineUsed").textContent = `${result.engine === "amap" ? "高德" : "枢纽"} · ${result.od_source}`;
      $("planOD").textContent = `${result.origin_text} → ${result.destination_text}${result.city ? `（${result.city}）` : ""}`;

      const details = result.details && result.details.length ? result.details : ["暂无分段详情"];
      $("planDetails").innerHTML = details.map((d) => `<li>${d}</li>`).join("");
      $("planTips").innerHTML = (result.tips || []).map((t) => `<li>${t}</li>`).join("");
      showRaw("planRaw", result);

      const mapBox = $("planMapBox");
      if (result.engine === "amap" && result.polyline && result.polyline.length > 1) {
        mapBox.style.display = "";
        drawPolyline($("planCanvas"), result.polyline, {
          origin: result.origin_text,
          destination: result.destination_text,
        });
      } else {
        mapBox.style.display = "none";
      }
      toast("路线规划完成");
    })
  );
}

/* ---------------- LLM 引擎 ---------------- */

function initLLM() {
  $("chipReal").addEventListener("click", () => {
    state.llm.preferReal = !state.llm.preferReal;
    $("chipReal").classList.toggle("active", state.llm.preferReal);
  });

  $("btnLLMLoad").addEventListener("click", () =>
    withBusy($("btnLLMLoad"), " 加载中", async () => {
      const result = await api.post("/api/llm/load", {
        model_id: $("llmModel").value.trim(),
        prefer_real: state.llm.preferReal,
      });
      const realError = result.real_adapter_error;
      $("llmLoadResult").innerHTML = `<div class="callout ${realError ? "warn" : ""}">模型 <b>${result.model_id}</b> 已就绪（${
        realError ? "Mock 回退：" + realError : state.llm.preferReal ? "真实适配器" : "Mock 适配器"
      }）</div>`;
      $("stLLM").textContent = result.model_id;
      toast(realError ? "真实适配器不可用，已回退 Mock" : "模型已加载", realError ? "err" : "ok");
    })
  );

  $("btnLLMTune").addEventListener("click", () =>
    withBusy($("btnLLMTune"), " 执行中", async () => {
      let config;
      try {
        config = JSON.parse($("llmTuneConfig").value);
      } catch (_) {
        throw new Error("配置 JSON 格式错误");
      }
      const result = await api.post("/api/llm/fine_tune", { config });
      $("llmTuneResult").innerHTML = `<div class="callout">状态：<b>${result.status}</b><br />产物：<code>${result.artifact.split(
        /[\\/]/
      ).pop()}</code></div>`;
      toast("微调接口执行完成");
    })
  );

  $("btnFT").addEventListener("click", () =>
    withBusy($("btnFT"), " 生成中", async () => {
      const result = await api.post("/api/llm/simulate_metrics", {
        config: { epochs: Number($("ftEpochs").value) || 36, seed: Number($("ftSeed").value) || 42 },
      });
      $("ftEmpty").style.display = "none";
      $("ftResult").style.display = "";
      $("ftLoss").textContent = result.final.loss;
      $("ftBleu").textContent = result.final.bleu4;
      $("ftR1").textContent = result.final.rouge1;
      $("ftRL").textContent = result.final.rougeL;

      const epochs = result.series.epoch;
      drawLineChart($("ftLossCanvas"), [{ x: epochs, y: result.series.loss, color: "#f87171" }]);
      drawLineChart(
        $("ftMetricCanvas"),
        [
          { x: epochs, y: result.series.bleu4, color: "#38bdf8" },
          { x: epochs, y: result.series.rouge1, color: "#34d399" },
          { x: epochs, y: result.series.rougeL, color: "#f59e0b" },
        ],
        { yMin: 0, yMax: 1 }
      );
      toast("微调指标报告已生成");
    })
  );
}

/* ---------------- 分析报告 ---------------- */

function initReports() {
  $("btnReportsAll").addEventListener("click", () =>
    withBusy($("btnReportsAll"), " 运行中", async () => {
      $("reportProgress").textContent = "运行中…";
      document.querySelectorAll("#reportBody .rstate").forEach((td) => {
        td.innerHTML = '<span class="badge">运行中</span>';
      });
      const result = await api.post("/api/features/run_existing", {});
      const reports = result.reports || {};
      let okCount = 0;
      const images = [];
      Object.entries(reports).forEach(([name, report]) => {
        const row = document.querySelector(`#reportBody tr[data-r="${name}"] .rstate`);
        if (report.ok) okCount += 1;
        if (row) {
          row.innerHTML = report.ok
            ? `<span class="badge ok">完成 · ${report.files.length} 文件</span>`
            : `<span class="badge warn" title="${(report.error || "").replace(/"/g, "")}">失败</span>`;
        }
        (report.files || []).forEach((f) => {
          if (/\.(png|gif)$/i.test(f)) images.push(f);
        });
      });
      const total = Object.keys(reports).length;
      $("reportProgress").className = `badge ${okCount === total ? "ok" : "warn"}`;
      $("reportProgress").textContent = `${okCount}/${total} 报告完成`;

      const gallery = $("gallery");
      if (images.length) {
        $("galleryEmpty").style.display = "none";
        gallery.style.display = "";
        gallery.innerHTML = images
          .map(
            (f) =>
              `<figure class="output-frame show" style="margin:0"><img src="/outputs/${f.split(/[\\/]/).pop()}?t=${Date.now()}" loading="lazy" /><figcaption>${f.split(
                /[\\/]/
              ).pop()}</figcaption></figure>`
          )
          .join("");
      }
      toast(`报告完成：${okCount}/${total}`);
    })
  );
}

/* ---------------- 启动 ---------------- */

async function boot() {
  switchView("overview");
  initOverview();
  initNavigation();
  initRL();
  initFlow();
  initPlan();
  initLLM();
  initReports();

  try {
    await Promise.all([refreshTopbar(), loadGrid()]);
    flowGroups = await api.get("/api/scenarios");
    renderGroupRows();
    await refreshRewards();
  } catch (error) {
    toast(`初始化失败：${error.message}`, "err");
  }
}

document.addEventListener("DOMContentLoaded", boot);
