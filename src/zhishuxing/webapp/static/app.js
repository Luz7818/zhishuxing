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

/* ---------------- 视图与主题(仿 harness:欢迎页 → 应用壳,tab 切换) ---------------- */

const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  overview: ["概览", "系统状态总览与快捷入口"],
  navigation: ["枢纽导航", "网格可视化 · 点击选取起终点 · 偏好感知 A* 路径规划"],
  rl: ["RL 智能体", "MADDPG 策略权重 · 推理 · 多智能体引导仿真"],
  flow: ["客流面板", "乘客分组 · 动态客流热力与引导路径"],
  plan: ["路线规划", "自然语言诉求 + 个性化偏好 → 真实换乘方案"],
  reports: ["分析报告", "七类可视化报告一键生成"],
};

function showView(name) {
  $("view-welcome").classList.toggle("hidden", name !== "welcome");
  $("app").classList.toggle("hidden", name !== "app");
}

function switchTab(tab) {
  $$(".nav-item[data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$("main section").forEach((s) => s.classList.toggle("active", s.id === "tab-" + tab));
  const [t, sub] = TITLES[tab] || [tab, ""];
  $("page-title").textContent = t;
  $("page-sub").textContent = sub;
  $("scroll-area").scrollTo({ top: 0 });
}

function enterApp() {
  showView("app");
  initAppOnce();
}

/* ---------------- 主题(明/暗,记忆偏好) ---------------- */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("zsx_theme", theme); } catch (_) { /* 隐私模式忽略 */ }
  /* 日/月图标由 CSS 依 data-theme 切换,无需改按钮内容 */
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("zsx_theme"); } catch (_) { /* 忽略 */ }
  applyTheme(saved || "light");
  $("theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
}

/* ---------------- Canvas 工具 ---------------- */

const LANDMARK_ALIAS = {
  entry_a: "A口进站",
  entry_b: "B口进站",
  security: "主安检",
  security_backup: "备用安检",
  metro_gate: "地铁闸机",
  rail_gate: "高铁闸机",
  bus_gate: "公交闸机",
  restroom_a: "卫生间A",
  restroom_b: "卫生间B",
  elevator_a: "无障碍直梯",
  stairs_passage: "楼梯通道",
  escalator_passage: "扶梯通道",
  nursing_room: "母婴室",
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

  // 设施语义层（楼梯/扶梯/直梯/拥挤区）
  const TAG_STYLE = {
    stairs: "rgba(249,115,22,0.55)",
    escalator: "rgba(251,191,36,0.5)",
    elevator: "rgba(52,211,153,0.55)",
    crowd: "rgba(248,113,113,0.28)",
  };
  Object.entries(grid.cell_tags || {}).forEach(([tag, cells]) => {
    const color = TAG_STYLE[tag];
    if (!color) return;
    ctx.fillStyle = color;
    cells.forEach(([tx, ty]) => {
      roundRect(ctx, ox + tx * cell + 2, oy + ty * cell + 2, cell - 4, cell - 4, 4);
      ctx.fill();
    });
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

/* ---------------- 高德 JS API 底图 ---------------- */

const amapState = { scriptLoading: false, loaded: false, map: null, overlays: [] };

function loadAmapScript() {
  return new Promise((resolve, reject) => {
    if (amapState.loaded) return resolve();
    const key = window.ZSX_CONFIG && window.ZSX_CONFIG.amapJsKey;
    if (!key) return reject(new Error("未配置 AMAP_JS_KEY"));

    if (amapState.scriptLoading) {
      const started = Date.now();
      const timer = setInterval(() => {
        if (amapState.loaded) {
          clearInterval(timer);
          resolve();
        } else if (Date.now() - started > 8000) {
          clearInterval(timer);
          reject(new Error("高德地图脚本加载超时"));
        }
      }, 120);
      return;
    }
    amapState.scriptLoading = true;

    // 2021-12 之后申请的 Key 需要配合安全密钥使用
    if (window.ZSX_CONFIG.amapSecurityCode) {
      window._AMapSecurityConfig = { securityJsCode: window.ZSX_CONFIG.amapSecurityCode };
    }
    const script = document.createElement("script");
    script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}&plugin=AMap.ToolBar,AMap.Scale`;
    script.onload = () => {
      // v2.0 脚本加载成功不代表初始化完成：Key 无效时 AMap 全局可能缺失
      const started = Date.now();
      const timer = setInterval(() => {
        if (window.AMap) {
          clearInterval(timer);
          amapState.loaded = true;
          resolve();
        } else if (Date.now() - started > 2500) {
          clearInterval(timer);
          amapState.scriptLoading = false;
          reject(new Error("高德地图初始化失败（Key 无效或缺少 AMAP_SECURITY_CODE 安全密钥）"));
        }
      }, 100);
    };
    script.onerror = () => {
      amapState.scriptLoading = false;
      reject(new Error("高德地图脚本加载失败（网络不可用）"));
    };
    document.head.appendChild(script);
  });
}

async function renderAmapMap(polyline, labels) {
  await loadAmapScript();
  const container = $("amapContainer");
  // 必须先让容器可见（有尺寸）再创建地图，否则瓦片按 0×0 初始化后一片空白
  container.style.display = "block";
  if (!amapState.map) {
    const mid = polyline[Math.floor(polyline.length / 2)];
    amapState.map = new AMap.Map(container, {
      zoom: 12,
      center: mid,
      resizeEnable: true,
      mapStyle: "amap://styles/dark",
    });
    amapState.map.addControl(new AMap.ToolBar());
    amapState.map.addControl(new AMap.Scale());
  } else {
    amapState.map.resize();
  }

  amapState.overlays.forEach((o) => amapState.map.remove(o));
  amapState.overlays = [];

  const path = polyline.map(([lng, lat]) => new AMap.LngLat(lng, lat));
  const line = new AMap.Polyline({
    path,
    strokeColor: "#38bdf8",
    strokeWeight: 6,
    strokeOpacity: 0.92,
    showDir: true,
    lineJoin: "round",
  });
  amapState.map.add(line);
  amapState.overlays.push(line);

  const marker = (position, label) => {
    const m = new AMap.Marker({
      position,
      label: { content: `<span class="amap-label">${label}</span>`, direction: "top" },
    });
    amapState.map.add(m);
    amapState.overlays.push(m);
  };
  marker(path[0], labels.origin || "起点");
  marker(path[path.length - 1], labels.destination || "终点");

  amapState.map.setFitView(amapState.overlays, false, [42, 42, 42, 42]);
}

/* ---------------- 全局状态 ---------------- */

const state = {
  nav: { grid: null, start: null, goal: null, via: new Set(), route: null, progress: 1 },
  plan: { strategy: "balanced", walkShort: false, crowdAvoid: false, preferElevator: false, needRestroom: false },
  llm: { preferReal: false },
  chat: { sessionId: null, busy: false },
};

/* ---------------- 顶栏 / 健康检查 ---------------- */

async function refreshTopbar() {
  try {
    const health = await api.get("/health");
    $("health-dot").classList.remove("off");
    $("health-text").textContent = "服务在线";
    const navFile = (health && health.navigation ? health.navigation : "").split(/[\\/]/).pop();
    $("health-chip").textContent = `导航图:${navFile || "已加载"}`;
    const ws = $("welcome-stats");
    if (ws && !ws.dataset.loaded) {
      ws.dataset.loaded = "1";
      ws.innerHTML = `
        <div class="hstat"><b>🧠<span class="num">4 维</span></b>优先级 · 硬约束 · 软偏好 · 画像</div>
        <div class="hstat"><b>🛗<span class="num">4 类</span></b>直梯/扶梯/楼梯/拥挤设施建模</div>
        <div class="hstat"><b>📚<span class="num">22 篇</span></b>站内换乘经验语料</div>
        <div class="hstat"><b>🤖<span class="num">2 引擎</span></b>枢纽偏好规划 + 高德真实路线</div>`;
    }
  } catch (_) {
    $("health-dot").classList.add("off");
    $("health-text").textContent = "服务离线";
    $("health-chip").textContent = "服务离线";
  }

  const status = await api.get("/api/rl/status").catch(() => null);
  if (status) {
    updatePolicyUI(status);
  }
  // 侧栏 LLM 模式标注
  const llmMode = state.llm.preferReal ? "真实适配器" : "Mock";
  $("side-llm").textContent = `LLM:${llmMode}`;
}

function updatePolicyUI(status) {
  $("policyChip").textContent = `策略:${status.policy_source}`;

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
    switchTab("rl");
    $("btnRLSim").click();
  });
  $("btnQuickPanel").addEventListener("click", () => {
    switchTab("flow");
    $("btnPanel").click();
  });
  $("btnQuickPolicy").addEventListener("click", () =>
    withBusy($("btnQuickPolicy"), " 加载中", async () => {
      await loadLatestPolicy();
      toast("策略权重已刷新");
    })
  );
  $("btnQuickReports").addEventListener("click", () => {
    switchTab("reports");
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

function selectPoint(selectEl, point, label) {
  const value = point.join(",");
  let opt = [...selectEl.options].find((o) => o.value === value);
  if (!opt) {
    selectEl.insertAdjacentHTML("beforeend", `<option value="${value}">${label}</option>`);
    opt = [...selectEl.options].find((o) => o.value === value);
  }
  selectEl.value = value;
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
      selectPoint($("navStart"), point, label);
      $("navGoal").value = "";
    } else {
      state.nav.goal = point;
      selectPoint($("navGoal"), point, label);
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
  const bindToggle = (id, key) => {
    $(id).addEventListener("click", () => {
      state.plan[key] = !state.plan[key];
      $(id).classList.toggle("active", state.plan[key]);
    });
  };
  bindToggle("chipWalk", "walkShort");
  bindToggle("chipCrowd", "crowdAvoid");
  bindToggle("chipElevator", "preferElevator");
  bindToggle("chipRestroom", "needRestroom");

  $("btnRealPlan").addEventListener("click", () =>
    withBusy($("btnRealPlan"), " 规划中", async () => {
      const result = await api.post("/api/plan", {
        question: $("planQuestion").value.trim(),
        engine: $("planEngine").value,
        prefs: {
          strategy: state.plan.strategy,
          walk: state.plan.walkShort ? "short" : "normal",
          crowd: state.plan.crowdAvoid ? "avoid" : "normal",
          prefer_elevator: state.plan.preferElevator || false,
          need_restroom: state.plan.needRestroom || false,
        },
      });
      $("planEmpty").style.display = "none";
      $("planResult").style.display = "";
      $("planDur").textContent = result.duration_sec ? `${Math.round(result.duration_sec / 60)} 分` : "—";
      $("planCost").textContent = result.cost !== undefined && result.cost !== null ? `${result.cost} 元` : "—";
      $("planSeg").textContent = result.segments !== undefined ? `${result.segments} 次` : "—";
      $("planEngineUsed").textContent = `${result.engine === "amap" ? "高德" : "枢纽"} · ${result.od_source}`;
      $("planOD").textContent = `${result.origin_text} → ${result.destination_text}${result.city ? `（${result.city}）` : ""}`;
      renderPlanProfile(result.profile);

      const details = result.details && result.details.length ? result.details : ["暂无分段详情"];
      $("planDetails").innerHTML = details.map((d) => `<li>${d}</li>`).join("");
      $("planTips").innerHTML = (result.tips || []).map((t) => `<li>${t}</li>`).join("");
      showRaw("planRaw", result);

      const mapBox = $("planMapBox");
      if (result.engine === "amap" && result.polyline && result.polyline.length > 1) {
        mapBox.style.display = "";
        const container = $("amapContainer");
        const canvas = $("planCanvas");
        const note = $("planMapNote");
        const labels = { origin: result.origin_text, destination: result.destination_text };
        const drawFallback = (reason) => {
          container.style.display = "none";
          canvas.style.display = "";
          drawPolyline(canvas, result.polyline, labels);
          note.textContent = `离线折线示意 · ${reason}`;
        };
        if (window.ZSX_CONFIG && window.ZSX_CONFIG.amapJsKey) {
          try {
            await renderAmapMap(result.polyline, labels);
            canvas.style.display = "none";
            note.textContent = "底图：高德地图 JS API · 折线沿真实道路";
          } catch (error) {
            drawFallback(error.message);
          }
        } else {
          drawFallback("未配置 AMAP_JS_KEY");
        }
      } else {
        mapBox.style.display = "none";
      }
      toast("路线规划完成");
    })
  );
}

/* ---------------- 智能换乘助手 ---------------- */

const PRIORITY_LABELS = { time: "时间优先", distance: "距离优先", comfort: "舒适优先", crowd: "少拥挤优先" };
const PROFILE_BADGE = { hard: "danger", soft: "info", persona: "warn" };

function profileChipHtml(profile) {
  const bits = [];
  const top = Object.entries(profile.priorities || {}).sort((a, b) => b[1] - a[1])[0];
  if (top && top[1] > 0.3) {
    bits.push(`<span class="badge info">${PRIORITY_LABELS[top[0]] || top[0]}</span>`);
  }
  ["hard", "soft", "persona"].forEach((cat) => {
    (profile.labels && profile.labels[cat] || []).forEach((item) => {
      bits.push(`<span class="badge ${PROFILE_BADGE[cat]}">${esc(item.label)}</span>`);
    });
  });
  return bits;
}

function appendChatMessage(role, text, pending) {
  const box = $("chatMessages");
  const el = document.createElement("div");
  el.className = `chat-msg ${role}${pending ? " pending" : ""}`;
  el.textContent = text;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

function renderProfileChips(profile) {
  const row = $("chatProfileRow");
  const host = $("chatProfileChips");
  if (!profile) {
    row.style.display = "none";
    return;
  }
  const chips = profileChipHtml(profile);
  if (!chips.length) {
    row.style.display = "none";
    return;
  }
  if (profile.source === "merged" || profile.source === "rule") {
    chips.push(`<span class="badge">${profile.source === "merged" ? "多轮累积" : "规则解析"}</span>`);
  }
  host.innerHTML = chips.join(" ");
  row.style.display = "";
}

function renderPlanProfile(profile) {
  const host = $("planProfileChips");
  if (!profile) {
    host.style.display = "none";
    return;
  }
  const chips = profileChipHtml(profile);
  if (!chips.length) {
    host.style.display = "none";
    return;
  }
  chips.push(`<span class="badge">${profile.source === "llm" ? "LLM 解析" : "规则解析"}</span>`);
  host.innerHTML = chips.join(" ");
  host.style.display = "";
}

function renderChatAnalysis(data) {
  const host = $("chatAnalysis");
  const parts = [];

  if (data.route_error) {
    parts.push(`<div class="callout warn">路线获取失败：${data.route_error}</div>`);
  }

  if (data.route) {
    const r = data.route;
    const engineBadge =
      r.engine === "hub"
        ? `<span class="badge ok">枢纽内偏好规划</span>`
        : `<span class="badge info">高德市际规划</span>`;
    const needs = data.profile && data.profile.summary && data.profile.summary !== "无特殊需求"
      ? `<span class="badge">需求：${data.profile.summary}</span>`
      : "";
    const kvBits = [];
    if (r.duration_sec) kvBits.push(`<span class="item">预计用时<b>${Math.round(r.duration_sec / 60)} 分</b></span>`);
    if (r.cost !== undefined && r.cost !== null) kvBits.push(`<span class="item">预计费用<b>${r.cost} 元</b></span>`);
    if (r.segments !== undefined) kvBits.push(`<span class="item">换乘<b>${r.segments} 次</b></span>`);
    if (r.meters !== undefined) kvBits.push(`<span class="item">步行距离<b>${r.meters} 米</b></span>`);
    parts.push(`
      <div class="card">
        <div class="row" style="justify-content:space-between;margin-bottom:10px">
          <div>${engineBadge} ${needs}</div>
          <details class="raw"><summary>原始数据</summary><pre>${JSON.stringify(data, null, 2)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")}</pre></details>
        </div>
        ${kvBits.length ? `<div class="kv" style="margin-bottom:12px">${kvBits.join("")}</div>` : ""}
        <div class="grid-2">
          <div>
            <h3 style="margin-top:0">🛣️ 路线方案</h3>
            <ol class="list">${(r.details || ["暂无分段详情"]).map((d) => `<li>${d}</li>`).join("")}</ol>
            ${r.engine === "hub" && Array.isArray(r.route) && r.route.length > 1 && state.nav.grid
              ? `<div class="canvas-box"><canvas id="chatRouteCanvas" width="620" height="290"></canvas></div>`
              : ""}
          </div>
          <div>
            <h3 style="margin-top:0">💡 出行提醒</h3>
            <ul class="list">${(r.tips || ["无"]).map((t) => `<li>${t}</li>`).join("")}</ul>
            ${
              data.kb_refs && data.kb_refs.length
                ? `<h3 style="margin-top:16px">📚 站内换乘经验引用</h3><ul class="list">${data.kb_refs
                    .map(
                      (k) =>
                        `<li>《${k.title}》<span style="color:var(--ink-3);font-size:12px"> · 相关度 ${k.score} · ${k.source}</span></li>`
                    )
                    .join("")}</ul>`
                : ""
            }
          </div>
        </div>
      </div>`);
  } else if (data.od_incomplete) {
    parts.push(`<div class="callout">需要更多信息：告诉我<b>从哪儿出发、到哪儿去</b>(例如「从A口到地铁闸机」),我就能给出完整偏好路线。</div>`);
  }

  host.innerHTML = parts.join("");

  const canvas = $("chatRouteCanvas");
  if (canvas) {
    drawGrid(canvas, state.nav.grid, {
      route: data.route.route,
      start: data.route.route[0],
      goal: data.route.route[data.route.route.length - 1],
      progress: 1,
    });
  }
}

async function sendChat(message) {
  if (state.chat.busy) return;
  const text = (message !== undefined ? message : $("chatInput").value).trim();
  if (!text) return;
  $("chatInput").value = "";
  state.chat.busy = true;

  appendChatMessage("user", text);
  const pendingEl = appendChatMessage("assistant", "正在理解您的需求并规划路线…", true);

  try {
    const data = await api.post("/api/chat", {
      message: text,
      session_id: state.chat.sessionId,
      prefs: {
        strategy: state.plan.strategy,
        walk: state.plan.walkShort ? "short" : "normal",
        crowd: state.plan.crowdAvoid ? "avoid" : "normal",
      },
    });
    pendingEl.remove();
    state.chat.sessionId = data.session_id;
    appendChatMessage("assistant", data.reply);
    renderProfileChips(data.profile);
    renderChatAnalysis(data);
  } catch (error) {
    pendingEl.remove();
    appendChatMessage("assistant", `出错了:${error.message}`);
  } finally {
    state.chat.busy = false;
  }
}

function initChat() {
  $("btnChatSend").addEventListener("click", () => sendChat());
  $("chatInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") sendChat();
  });

  document.querySelectorAll("#chatExamples .chip").forEach((chip) => {
    chip.addEventListener("click", () => sendChat(chip.dataset.q));
  });

  $("btnChatReset").addEventListener("click", async () => {
    if (state.chat.sessionId) {
      await api.post("/api/chat/reset", { session_id: state.chat.sessionId }).catch(() => null);
    }
    state.chat.sessionId = null;
    $("chatMessages").innerHTML =
      '<div class="chat-msg assistant">已开启新会话。直接说出您的换乘需求即可,例如:<b>「带老人行李多,优先直梯,去地铁前先上趟卫生间,从A口出发」</b>。</div>';
    $("chatProfileRow").style.display = "none";
    $("chatAnalysis").innerHTML = "";
    toast("已开启新会话");
  });
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
      $("llmLoadResult").innerHTML = `<div class="callout ${realError ? "warn" : "ok"}">模型 <b>${esc(result.model_id)}</b> 已就绪（${
        realError ? "Mock 回退：" + esc(realError) : state.llm.preferReal ? "真实适配器" : "Mock 适配器"
      }）</div>`;
      $("stLLM").textContent = result.model_id;
      const modeChip = $("ai-mode-chip");
      if (modeChip) {
        const real = state.llm.preferReal && !realError;
        modeChip.className = `badge ${real ? "ok" : ""}`;
        modeChip.textContent = real ? "真实端点" : "离线 Mock";
      }
      $("side-llm").textContent = `LLM:${realError ? "Mock" : state.llm.preferReal ? result.model_id : "Mock"}`;
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

/* ---------------- 右侧 AI 助手面板(仿 harness,记忆开合偏好) ---------------- */

function setAiOpen(open) {
  const panel = $("aipanel");
  const willOpen = open === undefined ? !panel.classList.contains("open") : !!open;
  panel.classList.toggle("open", willOpen);
  $("ai-rail").classList.toggle("on", willOpen);
  $("app").classList.toggle("ai-open", willOpen);
  const nav = $("nav-assistant");
  if (nav) nav.classList.toggle("active", willOpen);
  try { localStorage.setItem("zsx_ai_open", willOpen ? "1" : "0"); } catch (_) { /* 忽略 */ }
}

function initAiPanel() {
  let pref = null;
  try { pref = localStorage.getItem("zsx_ai_open"); } catch (_) { /* 忽略 */ }
  setAiOpen(pref === null ? window.innerWidth >= 1440 : pref === "1");
  $("ai-rail").addEventListener("click", () => setAiOpen());
}

/* ---------------- 命令面板(Ctrl/⌘+K) ---------------- */

let cmdkItems = [];
let cmdkSel = 0;

async function cmdkActions() {
  const navs = [
    ["overview", "概览", "📊"],
    ["plan", "路线规划", "🧭"],
    ["navigation", "枢纽导航", "🗺️"],
    ["flow", "客流面板", "🌡️"],
    ["rl", "RL 智能体", "🤖"],
    ["reports", "分析报告", "📑"],
  ].map(([tab, label, ico]) => ({
    ico, label, cat: "页面",
    run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab(tab); },
  }));
  const cmds = [
    { ico: "💬", label: "智能换乘助手面板", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); setAiOpen(true); } },
    { ico: "☾", label: "切换明暗主题", cat: "命令",
      run: () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark") },
    { ico: "⤓", label: "加载最新策略权重", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab("rl"); setTimeout(() => $("btnRLPolicy").click(), 250); } },
    { ico: "📑", label: "运行全部报告", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab("reports"); setTimeout(() => $("btnReportsAll").click(), 250); } },
    { ico: "📱", label: "打开移动端 PWA", cat: "命令", run: () => window.open("/mobile", "_blank") },
  ];
  return [...navs, ...cmds];
}

function renderCmdk(q = "") {
  const kw = q.trim().toLowerCase();
  const list = cmdkItems.filter((it) => !kw
    || it.label.toLowerCase().includes(kw) || it.cat.toLowerCase().includes(kw));
  cmdkSel = 0;
  const host = $("cmdk-list");
  host._filtered = list;
  host.innerHTML = list.length ? list.map((it, i) => `
    <div class="cmdk-item ${i === cmdkSel ? "sel" : ""}" data-i="${cmdkItems.indexOf(it)}">
      <span class="ck-ico">${it.ico}</span>${esc(it.label)}<small>${esc(it.cat)}</small>
    </div>`).join("") : `<div class="cmdk-empty">没有匹配项</div>`;
  $$(".cmdk-item", host).forEach((el) =>
    el.addEventListener("click", () => runCmdkItem(Number(el.dataset.i))));
}

function runCmdkItem(i) {
  const item = cmdkItems[i];
  if (!item) return;
  closeCmdk();
  item.run();
}

function openCmdk() {
  $("cmdk-mask").classList.remove("hidden");
  const input = $("cmdk-input");
  input.value = "";
  cmdkActions().then((items) => {
    cmdkItems = items;
    renderCmdk("");
    input.focus();
  });
}

function closeCmdk() {
  $("cmdk-mask").classList.add("hidden");
}

function initCmdk() {
  $("cmdk-input").addEventListener("input", (e) => renderCmdk(e.target.value));
  $("cmdk-input").addEventListener("keydown", (e) => {
    const host = $("cmdk-list");
    const list = host._filtered || cmdkItems;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!list.length) return;
      cmdkSel = (cmdkSel + (e.key === "ArrowDown" ? 1 : list.length - 1)) % list.length;
      const items = $$(".cmdk-item", host);
      items.forEach((el, i) => el.classList.toggle("sel", i === cmdkSel));
      if (items[cmdkSel]) items[cmdkSel].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      const item = list[cmdkSel];
      if (item) { closeCmdk(); item.run(); }
    } else if (e.key === "Escape") {
      closeCmdk();
    }
  });
  $("cmdk-mask").addEventListener("click", (e) => {
    if (e.target === $("cmdk-mask")) closeCmdk();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if ($("cmdk-mask").classList.contains("hidden")) openCmdk();
      else closeCmdk();
    }
  });
}

/* ---------------- 面板宽度拖拽(侧栏 / AI 面板) ---------------- */

function initPaneResize() {
  $$(".pane-resizer").forEach((handle) => {
    const kind = handle.dataset.resize;
    const varName = kind === "side" ? "--pane-side" : "--pane-ai";
    const currentWidth = () => (kind === "side"
      ? document.querySelector(".sidebar").getBoundingClientRect().width
      : $("aipanel").getBoundingClientRect().width);

    handle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      handle.classList.add("dragging");
      document.body.classList.add("col-resizing");
      const startX = event.clientX;
      const startW = currentWidth();
      const onMove = (e) => {
        const delta = kind === "side" ? e.clientX - startX : startX - e.clientX;
        const w = Math.min(560, Math.max(210, Math.round(startW + delta)));
        document.documentElement.style.setProperty(varName, `${w}px`);
      };
      const onUp = () => {
        handle.classList.remove("dragging");
        document.body.classList.remove("col-resizing");
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
    handle.addEventListener("dblclick", () => {
      document.documentElement.style.removeProperty(varName);
    });
  });
}

/* ---------------- 启动:欢迎页 → 应用壳 ---------------- */

let appInited = false;

function initAppOnce() {
  if (appInited) return;
  appInited = true;
  initOverview();
  initNavigation();
  initRL();
  initFlow();
  initPlan();
  initChat();
  initLLM();
  initReports();
  initAiPanel();
  initCmdk();
  initPaneResize();
  switchTab("overview");

  (async () => {
    try {
      await Promise.all([refreshTopbar(), loadGrid()]);
      flowGroups = await api.get("/api/scenarios");
      renderGroupRows();
      await refreshRewards();
    } catch (error) {
      toast(`初始化失败：${error.message}`, "err");
    }
  })();
}

function boot() {
  initTheme();
  $$(".nav-item[data-tab]").forEach((item) =>
    item.addEventListener("click", () => switchTab(item.dataset.tab)));
  showView("welcome");
  refreshTopbar(); // 欢迎页统计与服务状态(应用壳元素隐藏但已存在)
}

document.addEventListener("DOMContentLoaded", boot);
