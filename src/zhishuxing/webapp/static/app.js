/* ============================================================
   智枢星 Web 控制台交互 v2
   视图路由 / API 客户端 / Canvas 网格与图表 / 全流程反馈

   章节索引(单文件 PWA 有意取舍,不引打包器;按标记检索):
   ① API 客户端 → ② 反馈组件 → ③ 视图与主题/侧栏药丸 → ④ Canvas 工具
   → ⑤ 高德底图 → ⑥ 全局状态/顶栏健康 → ⑦ 概览/枢纽导航/RL/客流/路线规划
   → ⑧ 智能换乘助手(发送/档案/分析/悬浮模式) → ⑨ 引擎与微调面板
   → ⑩ 分析报告/设置 → ⑪ Dock 开合/侧栏收起 → ⑫ 命令面板 → ⑬ 启动
   ============================================================ */

"use strict";

/* 桌面壳适配(?desktop=1,由 packaging/desktop_entry.py 注入):
   关闭浏览器专属行为,让 WebView2 窗口表现得像原生应用 */
const IS_DESKTOP = new URLSearchParams(location.search).get("desktop") === "1";
if (IS_DESKTOP) {
  document.addEventListener("contextmenu", (e) => e.preventDefault());
  document.addEventListener(
    "wheel",
    (e) => {
      if (e.ctrlKey) e.preventDefault(); // 阻止 Ctrl+滚轮页面缩放破坏布局
    },
    { passive: false },
  );
  document.addEventListener("keydown", (e) => {
    // 阻止 Ctrl +/-/0 缩放快捷键;F12 留给排障
    if (e.ctrlKey && ["+", "-", "=", "0"].includes(e.key)) e.preventDefault();
  });
}

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
    const message = (payload && payload.error) || `请求失败: ${response.status}`;
    // 403 等拒绝响应会带 detail 说明原因与替代做法，一并抛出便于用户理解
    throw new Error(payload && payload.detail ? `${message}（${payload.detail}）` : message);
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
  const icon = type === "err" ? "i-alert" : "i-check";
  el.innerHTML = `<span class="t-ico ${type === "err" ? "err" : "ok"}"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#${icon}"/></svg></span><span></span>`;
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

/* 数值换位:文本变化时轻上浮淡入一次(与顶栏标题 swap 同语言) */
function setTick(el, text) {
  if (!el || el.textContent === String(text)) return;
  el.textContent = text;
  el.classList.remove("tick");
  void el.offsetWidth;
  el.classList.add("tick");
}

function showRaw(preId, data) {
  $(preId).textContent = JSON.stringify(data, null, 2);
}

function setImage(imgId, frameId, url) {
  const img = $(imgId);
  img.src = `${url}?t=${Date.now()}`;
  $(frameId).classList.add("show");
  bindImgFade(img);
}

/* 输出图加载淡入:缓存命中直接显示,网络加载时避免破图/半图突兀 */
function bindImgFade(img) {
  if (img.complete && img.naturalWidth) { img.classList.remove("img-fade", "img-in"); return; }
  img.classList.add("img-fade");
  img.classList.remove("img-in");
  img.addEventListener("load", () => img.classList.add("img-in"), { once: true });
  img.addEventListener("error", () => img.classList.remove("img-fade"), { once: true });
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
  settings: ["设置", "本地 .env 密钥配置 · 状态掩码 · 降级链路说明"],
};

function showView(name) {
  const swap = () => {
    $("view-welcome").classList.toggle("hidden", name !== "welcome");
    $("app").classList.toggle("hidden", name !== "app");
  };
  if (name === "app" && document.startViewTransition && !reduceMotion()) {
    /* 进入工作台:新视图自下方上浮淡入(View Transition 快照栈)。
       initAppOnce 在交换前同步执行,交换完成后立即补定位(getBoundingClientRect 强制同步布局) */
    const root = document.documentElement;
    root.classList.add("vt-enter");
    const transition = document.startViewTransition(() => {
      swap();
      moveNavPill(true);
    });
    transition.finished.then(() => {
      root.classList.remove("vt-enter");
      /* 快照期间 getBoundingClientRect 可能读到过渡态布局(如窄屏断点宽度),
         按交换完成后的真实布局再定位一次,覆盖任何失真值 */
      moveNavPill(true);
    }, () => root.classList.remove("vt-enter"));
  } else {
    swap();
    /* 无 View Transition 的浏览器:给工作台一次轻入场作为回退(reduced-motion 下保持瞬时) */
    if (name === "app" && !reduceMotion()) document.documentElement.classList.add("no-vt");
  }
}

function switchTab(tab) {
  $$(".nav-item[data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  /* 旧页「发车」:淡出 + 轻微上行(--dur-2,比新页 pageIn --dur-3 快约 20%),
     期间 absolute + pointer-events:none 豁免布局回流,过渡结束后摘类落 display:none。
     定时摘类不依赖 transitionend;reduceMotion 或应用壳未显示(首次进入)时不加
     退场类,保持瞬时切换;快速连切先把上一退场页立即落定,避免两页同时离站 */
  const prev = document.querySelector("main section.page.active");
  if (prev && prev.id !== "tab-" + tab && !$("app").classList.contains("hidden") && !reduceMotion()) {
    clearTimeout(switchTab._leaving);
    $$("main section.leaving").forEach((s) => s.classList.remove("leaving"));
    prev.classList.add("leaving");
    switchTab._leaving = setTimeout(() => prev.classList.remove("leaving"), 200);   /* --dur-2 + 余量 */
  }
  $$("main section").forEach((s) => s.classList.toggle("active", s.id === "tab-" + tab));
  const [t, sub] = TITLES[tab] || [tab, ""];
  $("page-title").textContent = t;
  $("page-sub").textContent = sub;
  /* 标题随页面轻换位 + 侧栏高亮药丸滑动,给出空间连续性 */
  const head = document.querySelector(".topbar-left > div");
  if (head) { head.classList.remove("swap"); void head.offsetWidth; head.classList.add("swap"); }
  moveNavPill();
  $("scroll-area").scrollTo({ top: 0 });
  try { localStorage.setItem("zsx_tab", tab); } catch (_) { /* 隐私模式忽略 */ }
}

/* ---------------- 侧栏滑动高亮:激活背景由独立药丸承载,随切换滑动 ---------------- */

let navPill = null;

function initNavPill() {
  const nav = document.querySelector(".nav-scroll");
  if (!nav || navPill) return;
  navPill = document.createElement("span");
  navPill.className = "nav-pill";
  navPill.setAttribute("aria-hidden", "true");
  nav.appendChild(navPill);
  moveNavPill(true);
  window.addEventListener("resize", () => moveNavPill(true));
}

function moveNavPill(instant) {
  if (!navPill) return;
  const nav = document.querySelector(".nav-scroll");
  const item = document.querySelector(".nav-item[data-tab].active");
  if (!nav || !item) {
    navPill.style.opacity = "0";
    return;
  }
  const ir = item.getBoundingClientRect();
  if (!ir.width) {
    /* 应用壳尚未显示(View Transition 交换前)不定位,showView 在交换完成后会补一次 */
    navPill.style.opacity = "0";
    return;
  }
  const nr = nav.getBoundingClientRect();
  if (instant) navPill.style.transition = "none";
  navPill.style.opacity = "1";
  navPill.style.width = `${ir.width}px`;
  navPill.style.height = `${ir.height}px`;
  navPill.style.transform =
    `translate(${ir.left - nr.left + nav.scrollLeft}px, ${ir.top - nr.top + nav.scrollTop}px)`;
  /* 同步落地 instant 定位:强制重排后立即恢复过渡,不依赖 rAF(部分环境不触发帧回调) */
  if (instant) { void navPill.offsetWidth; navPill.style.transition = ""; }
}

function lastTab() {
  try { return localStorage.getItem("zsx_tab"); } catch (_) { return null; }
}

function enterApp() {
  showView("app");
  initAppOnce();
}

/* ---------------- 主题(明/暗,记忆偏好) ---------------- */

function reduceMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function applyTheme(theme) {
/* 主题切换:无页面级动画(GitHub/Linear 同款)。
   切换瞬间挂 theme-fading 类,让全站颜色以统一曲线滑入新配色,260ms 后移除;
   reduced-motion 下全局规则会抑制过渡,表现为瞬时切换 */
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("zsx_theme", theme); } catch (_) { /* 隐私模式忽略 */ }
  if (!reduceMotion()) {
    const root = document.documentElement;
    root.classList.add("theme-fading");
    clearTimeout(applyTheme._t);
    applyTheme._t = setTimeout(() => root.classList.remove("theme-fading"), 260);
  }
  /* 画布取色随主题变(F5):可见画布立即按新令牌重绘(瞬时路径,不重放描线动画) */
  repaintThemedCanvases();
  /* 日/月图标由 CSS 依 data-theme 切换,无需改按钮内容 */
}

/* 主题切换后按新令牌重绘可见画布:全部走既有瞬时路径(drawLineChart 传
   opts.animate=false、drawGrid/drawPolyline 直接画终态),绝不重放 560ms 描线;
   canvas 像素不参与 theme-fading 颜色过渡,重绘即终态。工作台未初始化时跳过
   (画布由 initAppOnce 首绘);对话/微调/折线画布的数据在各自渲染处留有缓存 */
function repaintThemedCanvases() {
  if (!appInited) return;
  if (state.nav.grid) repaintNav();
  if (rewardSeriesCache.length) {
    const colors = chartColors();
    rewardSeriesCache = rewardSeriesCache.map((s, i) => ({ ...s, color: colors[i % colors.length] }));
    drawLineChart($("rewardCanvas"), rewardSeriesCache, { animate: false });
    $$("#rewardLegend .swatch").forEach((sw, i) => { sw.style.background = colors[i % colors.length]; });
  }
  if (ftSeriesCache) {
    drawLineChart($("ftLossCanvas"), ftSeriesCache.loss, { animate: false });
    drawLineChart($("ftMetricCanvas"), ftSeriesCache.metric, { animate: false, yMin: 0, yMax: 1 });
  }
  if (chatRouteView && $("chatRouteCanvas") && state.nav.grid) {
    drawGrid($("chatRouteCanvas"), state.nav.grid, chatRouteView);
  }
  if (polylineCache) drawPolyline($("planCanvas"), polylineCache.coords, polylineCache.labels);
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("zsx_theme"); } catch (_) { /* 忽略 */ }
  /* 首访(无记忆偏好)跟随系统明暗,与移动端三态「auto」同口径;已有偏好则尊重记忆 */
  const system = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(saved || system);
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

/* 画布主题取色(F5):网格/图表底色与文字从 CSS 令牌读取,随明暗主题重绘
   (移动端 mobile_app.html drawGrid 为同款先例);令牌组见 style.css --canvas-* */
const cvar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawGrid(canvas, grid, view) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cell = Math.min((W - 30) / grid.width, (H - 30) / grid.height);
  const ox = (W - cell * grid.width) / 2;
  const oy = (H - cell * grid.height) / 2;

  // 底色网格(令牌取色,随明暗主题)
  ctx.fillStyle = cvar("--canvas-bg");
  ctx.fillRect(ox, oy, cell * grid.width, cell * grid.height);
  ctx.strokeStyle = cvar("--canvas-line");
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
  ctx.fillStyle = cvar("--canvas-blocked");
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

  // 悬停准星格:拾取起终点的瞄准提示(鼠标悬停时由 repaintNav 重绘)
  if (view && view.hover) {
    const [hx, hy] = view.hover;
    if (hx >= 0 && hy >= 0 && hx < grid.width && hy < grid.height) {
      ctx.fillStyle = "rgba(56,189,248,.13)";
      roundRect(ctx, ox + hx * cell + 1.5, oy + hy * cell + 1.5, cell - 3, cell - 3, 4);
      ctx.fill();
      ctx.strokeStyle = "rgba(56,189,248,.5)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // 地标
  Object.entries(grid.landmarks).forEach(([key, [lx, ly]]) => {
    const cx = ox + lx * cell + cell / 2;
    const cy = oy + ly * cell + cell / 2;
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(4, cell * 0.22), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = cvar("--canvas-text");
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

/* 图表序列色(F5):明暗两组,亮色取 600 系保证浅底对比度;随主题取用 */
const CHART_COLORS = {
  dark: ["#38bdf8", "#f59e0b", "#34d399", "#a78bfa", "#f87171"],
  light: ["#0284c7", "#b45309", "#059669", "#7c3aed", "#dc2626"],
};
const chartColors = () => CHART_COLORS[document.documentElement.dataset.theme === "dark" ? "dark" : "light"];

/* 同一画布的描线动画互斥:新数据到达即取消旧帧,不叠加 */
const chartAnims = new WeakMap();

function drawLineChart(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

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

  const render = (progress) => {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = cvar("--canvas-line");
    ctx.fillStyle = cvar("--canvas-text-dim");
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

    const colors = chartColors();
    series.forEach((s, si) => {
      const color = s.color || colors[si % colors.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = s.width || 2;
      ctx.beginPath();
      const n = s.y.length;
      const upto = Math.max(2, Math.ceil(n * progress));
      for (let i = 0; i < Math.min(n, upto); i++) {
        const xi = s.x ? s.x[i] : i;
        const [px, py] = toPx(xi, s.y[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
  };

  const prev = chartAnims.get(canvas);
  if (prev) cancelAnimationFrame(prev);

  /* 描线入场:约 0.56s ease-out,坐标轴常驻、折线自左向右落笔;
     reduced-motion 或显式 animate:false 时直接画终态 */
  if (opts.animate === false || reduceMotion()) { render(1); return; }
  const t0 = performance.now();
  const dur = 560;
  const frame = (now) => {
    const t = Math.min(1, (now - t0) / dur);
    render(1 - Math.pow(1 - t, 3));
    if (t < 1) chartAnims.set(canvas, requestAnimationFrame(frame));
    else chartAnims.delete(canvas);
  };
  chartAnims.set(canvas, requestAnimationFrame(frame));
}

function ctxClear(canvas) {
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

function drawChartEmpty(canvas, text) {
  ctxClear(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = cvar("--canvas-text-dim");
  ctx.font = '13px "Microsoft YaHei"';
  ctx.textAlign = "center";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

function drawPolyline(canvas, coords, labels = {}, progress = 1) {
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

  ctx.fillStyle = cvar("--canvas-bg");
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(56,189,248,0.6)";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  const upto = Math.max(2, Math.ceil(coords.length * progress));
  coords.slice(0, upto).forEach(([lon, lat], i) => {
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
      ctx.fillStyle = cvar("--canvas-text");
      ctx.font = '12px "Microsoft YaHei"';
      ctx.textAlign = "center";
      ctx.fillText(label, x, y - 13);
    }
  };
  pin(coords[0], "#22d3ee", labels.origin || "起点");
  pin(coords[coords.length - 1], "#34d399", labels.destination || "终点");
}

/* 回退折线的描线入场:与 drawLineChart 同一语言(0.56s ease-out,新请求取消旧帧) */
const polylineAnims = new WeakMap();
let polylineCache = null;   // 最近一次折线数据,供主题切换重绘终态

function animatePolyline(canvas, coords, labels) {
  polylineCache = { coords, labels };
  const prev = polylineAnims.get(canvas);
  if (prev) cancelAnimationFrame(prev);
  if (reduceMotion()) { drawPolyline(canvas, coords, labels, 1); return; }
  const t0 = performance.now();
  const dur = 560;
  const frame = (now) => {
    const t = Math.min(1, (now - t0) / dur);
    drawPolyline(canvas, coords, labels, 1 - Math.pow(1 - t, 3));
    if (t < 1) polylineAnims.set(canvas, requestAnimationFrame(frame));
    else polylineAnims.delete(canvas);
  };
  polylineAnims.set(canvas, requestAnimationFrame(frame));
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
    $("health-chip").classList.remove("off");
    const navFile = (health && health.navigation ? health.navigation : "").split(/[\\/]/).pop();
    $("health-chip").textContent = `导航图:${navFile || "已加载"}`;
    const ws = $("welcome-stats");
    if (ws && !ws.dataset.loaded) {
      ws.dataset.loaded = "1";
      ws.innerHTML = `
        <div class="hstat"><span class="hs-ico"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-idcard"/></svg></span><div class="hs-txt"><b class="num">4 维</b><span>优先级 · 硬约束 · 软偏好 · 画像</span></div></div>
        <div class="hstat"><span class="hs-ico"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-elevator"/></svg></span><div class="hs-txt"><b class="num">4 类</b><span>直梯/扶梯/楼梯/拥挤设施建模</span></div></div>
        <div class="hstat"><span class="hs-ico"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-book"/></svg></span><div class="hs-txt"><b class="num">22 篇</b><span>站内换乘经验语料</span></div></div>
        <div class="hstat"><span class="hs-ico"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-hub"/></svg></span><div class="hs-txt"><b class="num">2 引擎</b><span>枢纽偏好规划 + 高德真实路线</span></div></div>`;
    }
  } catch (_) {
    $("health-dot").classList.add("off");
    $("health-text").textContent = "服务离线";
    $("health-chip").textContent = "服务离线";
    $("health-chip").classList.add("off");
  }

  const status = await api.get("/api/rl/status").catch(() => null);
  if (status) {
    updatePolicyUI(status);
  }
  // 侧栏 LLM 模式标注
  const llmMode = state.llm.preferReal ? "真实适配器" : "Mock";
  $("side-llm").textContent = `LLM:${llmMode}`;

  // 授权状态 chip:试用/临期/宽限/授权信息如实展示(取自 /api/settings 的只读状态)
  const settingsState = await api.get("/api/settings").catch(() => null);
  const licChip = $("lic-chip");
  if (licChip && settingsState && settingsState.license) {
    const lic = settingsState.license;
    licChip.hidden = false;
    licChip.title = lic.message || "";
    if (lic.status === "missing" || lic.status === "invalid") {
      licChip.textContent = "授权:试用版";
      licChip.classList.add("off");
    } else if (lic.status === "grace") {
      licChip.textContent = `授权:宽限 ${-lic.days_left} 天`;
      licChip.classList.remove("off");
    } else if (lic.status === "expiring") {
      licChip.textContent = `授权:剩 ${lic.days_left} 天`;
      licChip.classList.remove("off");
    } else {
      licChip.textContent =
        lic.expires_at && lic.expires_at !== "永久"
          ? `授权:${lic.customer} · ${lic.expires_at}`
          : `授权:${lic.customer} · 永久`;
      licChip.classList.remove("off");
    }
  }
}

function updatePolicyUI(status) {
  $("policyChip").textContent = `策略:${status.policy_source}`;

  setTick($("kpiPolicy"), status.policy_source);
  setTick($("kpiSteps"), status.latest_step_k ? `${status.latest_step_k}k` : "—");
  $("kpiStepsFoot").textContent = status.latest_step_k ? `检查点 step_${status.latest_step_k}k` : "检查点加载后显示";
  setTick($("kpiCheckpoints"), status.checkpoint_count);
  setTick($("kpiRewards"), status.reward_files.length);
  $("stTorch").textContent = status.torch_available ? `可用 · ${status.torch_version}` : "未安装（启发式回退）";

  const table = $("rlCkptTable");
  const empty = $("rlCkptEmpty");
  if (status.checkpoints.length) {
    table.style.display = "";
    empty.style.display = "none";
    $("rlCkptBody").innerHTML = status.checkpoints
      .map(
        (c) =>
          `<tr><td class="strong">agent_${esc(c.agent_id)}</td><td>${esc(c.algorithm)}</td><td>step_${esc(c.step_k)}k</td><td>${esc(c.path.split(/[\\/]/).pop())}</td></tr>`
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
    .map(([key, [x, y]]) => `<option value="${x},${y}">${esc(lmName(key))} (${x},${y})</option>`)
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
  $("viaChips").innerHTML = viaKeys.map((k) => `<button class="chip" data-lm="${esc(k)}">${esc(lmName(k))}</button>`).join("");
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
    const t = Math.min(1, (now - started) / duration);
    view.progress = 1 - Math.pow(1 - t, 3);   // easeOutCubic:落笔先快后慢
    repaintNav();
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* 任意网格画布的描线动画(对话分析卡的小图用);卡片被替换后自动停笔 */
function animateGridCanvas(canvas, view, duration = 560) {
  if (reduceMotion()) { view.progress = 1; drawGrid(canvas, state.nav.grid, view); return; }
  const t0 = performance.now();
  const frame = (now) => {
    if (!canvas.isConnected) return;
    const t = Math.min(1, (now - t0) / duration);
    view.progress = 1 - Math.pow(1 - t, 3);
    drawGrid(canvas, state.nav.grid, view);
    if (t < 1) requestAnimationFrame(frame);
  };
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

/* 指针事件 → 网格坐标(点击与悬停准星共用);不在界内返回 null */
function cellFromEvent(event, canvas, grid) {
  const rect = canvas.getBoundingClientRect();
  const cx = ((event.clientX - rect.left) * canvas.width) / rect.width;
  const cy = ((event.clientY - rect.top) * canvas.height) / rect.height;
  const cell = Math.min((canvas.width - 30) / grid.width, (canvas.height - 30) / grid.height);
  const ox = (canvas.width - cell * grid.width) / 2;
  const oy = (canvas.height - cell * grid.height) / 2;
  const gx = Math.floor((cx - ox) / cell);
  const gy = Math.floor((cy - oy) / cell);
  if (gx < 0 || gy < 0 || gx >= grid.width || gy >= grid.height) return null;
  return [gx, gy];
}

function initNavigation() {
  const canvas = $("navCanvas");
  canvas.style.cursor = "crosshair";
  canvas.addEventListener("click", (event) => {
    const grid = state.nav.grid;
    if (!grid) return;
    const hit = cellFromEvent(event, canvas, grid);
    if (!hit) return;
    const [gx, gy] = hit;
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

  /* 悬停准星:高亮当前格,拾取起终点的瞄准感(触屏不触发,无副作用) */
  canvas.addEventListener("mousemove", (event) => {
    const grid = state.nav.grid;
    if (!grid) return;
    const hit = cellFromEvent(event, canvas, grid);
    const hover = state.nav.hover;
    const same = hit === hover || (hit && hover && hit[0] === hover[0] && hit[1] === hover[1]);
    if (!same) {
      state.nav.hover = hit;
      repaintNav();
    }
  });
  canvas.addEventListener("mouseleave", () => {
    if (state.nav.hover) {
      state.nav.hover = null;
      repaintNav();
    }
  });

  $("btnNavReload").addEventListener("click", () =>
    withBusy($("btnNavReload"), " 加载中", async () => {
      await api.post("/api/navigation/load", { file_path: state.nav.grid.file });
      await loadGrid();
      toast("导航图已重新加载");
    })
  );

  $("btnPlanParams").addEventListener("click", () =>
    $("plan-params-mask").classList.remove("hidden"));
  $("btnPlanParamsClose").addEventListener("click", () =>
    $("plan-params-mask").classList.add("hidden"));
  $("plan-params-mask").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) $("plan-params-mask").classList.add("hidden");
  });
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
      setTick($("navLen"), `${result.length} 格`);
      setTick($("navVia"), state.nav.via.size ? [...state.nav.via].map(lmName).join("、") : "无");
      showRaw("navRaw", result.route);
      toast(`规划完成：路径长度 ${result.length} 格`);
      $("plan-params-mask").classList.add("hidden");   // 规划成功即收起参数弹窗,结果看地图左下悬浮卡
    })
  );

  /* Escape 关闭规划参数弹窗(与 cmdk / LLM 面板同一习惯) */
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("plan-params-mask").classList.add("hidden");
  });
}

/* ---------------- RL 智能体 ---------------- */

async function loadLatestPolicy() {
  const status = await api.post("/api/rl/load_policy", {});
  updatePolicyUI(status);
  return status;
}

/* 奖励曲线最近一次绘制的数据,供图例悬停高亮重绘 */
let rewardSeriesCache = [];

function highlightRewardSeries(activeIdx) {
  const canvas = $("rewardCanvas");
  if (!canvas || !rewardSeriesCache.length) return;
  const series = rewardSeriesCache.map((s, i) => (activeIdx === -1 || i === activeIdx
    ? { ...s, width: i === activeIdx ? 3 : 2 }
    : { ...s, color: "rgba(110,127,160,.22)", width: 1.5 }));
  drawLineChart(canvas, series, { animate: false });
}

async function refreshRewards() {
  const data = await api.get("/api/rl/rewards");
  const colors = chartColors();
  const series = (data.series || []).map((s, i) => ({
    name: s.file.replace(/\.npy$/, ""),
    x: s.x,
    y: s.y,
    color: colors[i % colors.length],
  }));
  const canvas = $("rewardCanvas");
  if (!series.length) {
    rewardSeriesCache = [];
    drawChartEmpty(canvas, "暂无 *_env_*.npy 奖励数据（训练后自动出现）");
    $("rewardLegend").innerHTML = "";
    return;
  }
  rewardSeriesCache = series;
  drawLineChart(canvas, series);
  $("rewardLegend").innerHTML = series
    .map(
      (s, i) =>
        `<button class="key legend-key" data-series="${i}" type="button" title="悬停高亮该曲线"><span class="swatch" style="background:${colors[i % colors.length]}"></span>${esc(s.name)}</button>`
    )
    .join("");
  /* 图例悬停高亮对应曲线,移出恢复(多 seed 对比时快速分辨) */
  $$("#rewardLegend .legend-key").forEach((key) => {
    key.addEventListener("mouseenter", () => highlightRewardSeries(Number(key.dataset.series)));
    key.addEventListener("mouseleave", () => highlightRewardSeries(-1));
  });
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
            `<span class="badge info" style="margin:3px 6px 3px 0">agent_${i} → 前进 ${a[0].toFixed(3)} · 转向 ${a[1].toFixed(3)}（${esc(result.policy_used[i])}）</span>`
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
      setTick($("simArrive"), `${rate}%`);
      setTick($("simAvg"), result.avg_transfer_steps ?? "未到达");
      setTick($("simBase"), result.free_flow_baseline_steps);
      setTick($("simCong"), result.congestion_peak);
      setTick($("simColl"), result.collision_events);
      setTick($("simExec"), result.steps_executed);
      setTick($("simPolicy"), result.policy_source);
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
        <td class="strong">${esc(g.name)}</td>
        <td>${esc(g.start)} → ${esc(g.goal)}${g.via_landmarks && g.via_landmarks.length ? `（经 ${esc(g.via_landmarks.join("、"))}）` : ""}</td>
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
  $("btnGroupReset").addEventListener("click", () =>
    withBusy($("btnGroupReset"), " 恢复中", async () => {
      flowGroups = await api.get("/api/scenarios");
      renderGroupRows();
      toast("已恢复默认场景");
    })
  );

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
      setTick($("planDur"), result.duration_sec ? `${Math.round(result.duration_sec / 60)} 分` : "—");
      setTick($("planCost"), result.cost !== undefined && result.cost !== null ? `${result.cost} 元` : "—");
      setTick($("planSeg"), result.segments !== undefined ? `${result.segments} 次` : "—");
      setTick($("planEngineUsed"), `${result.engine === "amap" ? "高德" : "枢纽"} · ${result.od_source}`);
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
          animatePolyline(canvas, result.polyline, labels);
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
    bits.push(`<span class="badge info">${esc(PRIORITY_LABELS[top[0]] || top[0])}</span>`);
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
  /* 回读保护:用户已向上翻阅历史(离底 >80px)时,新回复不再把视口拽到底部;
     用户自己发送的消息(role=user)始终滚底——那是主动行为,应当看到自己说的话 */
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  const el = document.createElement("div");
  el.className = `chat-msg ${role}${pending ? " pending" : ""}`;
  if (pending) {
    el.innerHTML = `<span class="spinner-inline"></span><span></span>`;
    el.lastElementChild.textContent = text;
  } else {
    el.textContent = text;
  }
  box.appendChild(el);
  if (nearBottom || role === "user") box.scrollTop = box.scrollHeight;
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

/* 最近一次对话路线小图的视图,供主题切换重绘(canvas 随分析卡内容替换) */
let chatRouteView = null;

function renderChatAnalysis(data) {
  const host = $("chatAnalysis");
  const parts = [];


  if (data.route_error) {
    parts.push(`<div class="callout warn">路线获取失败：${esc(data.route_error)}</div>`);
  }

  if (data.route) {
    const r = data.route;
    const engineBadge =
      r.engine === "hub"
        ? `<span class="badge ok">枢纽内偏好规划</span>`
        : `<span class="badge info">高德市际规划</span>`;
    const needs = data.profile && data.profile.summary && data.profile.summary !== "无特殊需求"
      ? `<span class="badge">需求：${esc(data.profile.summary)}</span>`
      : "";
    const kvBits = [];
    if (r.duration_sec) kvBits.push(`<span class="item">预计用时<b>${Math.round(r.duration_sec / 60)} 分</b></span>`);
    if (r.cost !== undefined && r.cost !== null) kvBits.push(`<span class="item">预计费用<b>${esc(r.cost)} 元</b></span>`);
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
            <h3 style="margin-top:0"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-road"/></svg>路线方案</h3>
            <ol class="list">${(r.details || ["暂无分段详情"]).map((d) => `<li>${esc(d)}</li>`).join("")}</ol>
            ${r.engine === "hub" && Array.isArray(r.route) && r.route.length > 1 && state.nav.grid
              ? `<div class="canvas-box"><canvas id="chatRouteCanvas" width="620" height="290" role="img" aria-label="对话路线在枢纽网格上的示意图"></canvas></div>`
              : ""}
          </div>
          <div>
            <h3 style="margin-top:0"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-bulb"/></svg>出行提醒</h3>
            <ul class="list">${(r.tips || ["无"]).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
            ${
              data.kb_refs && data.kb_refs.length
                ? `<h3 style="margin-top:16px"><svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#i-book"/></svg>站内换乘经验引用</h3><ul class="list">${data.kb_refs
                    .map(
                      (k) =>
                        `<li>《${esc(k.title)}》<span style="color:var(--t3);font-size:12px"> · 相关度 ${esc(k.score)} · ${esc(k.source)}</span></li>`
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
  /* 整块内容换新时轻淡入一次,替代硬切 teleport(reduced-motion 由全局规则抑制) */
  host.classList.remove("ca-in");
  void host.offsetWidth;
  host.classList.add("ca-in");

  const canvas = $("chatRouteCanvas");
  if (canvas) {
    chatRouteView = {
      route: data.route.route,
      start: data.route.route[0],
      goal: data.route.route[data.route.route.length - 1],
      progress: 0,
    };
    animateGridCanvas(canvas, chatRouteView);
  }
}

async function sendChat(message) {
  if (state.chat.busy) return;
  const text = (message !== undefined ? message : $("chatInput").value).trim();
  if (!text) return;
  $("chatInput").value = "";
  state.chat.busy = true;

  setDockOpen(true);   // 底部 Dock 随发送自动展开,对话历史立即可见
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
    if (data.action && data.action.type === "switch_tab") {
      const labels = { overview: "概览", plan: "路线规划", navigation: "枢纽导航",
                       flow: "客流面板", rl: "RL 智能体", reports: "分析报告", settings: "设置" };
      switchTab(data.action.tab);
      toast(`已为你打开「${labels[data.action.tab] || data.action.tab}」`);
    }
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

/* ---------------- 对话引擎 / 模型微调 面板(弹窗) ---------------- */

function setLLMPanel(open) {
  $("llm-panel-mask").classList.toggle("hidden", !open);
}

function initLLMPanel() {
  $("btnDockTools").addEventListener("click", () => setLLMPanel(true));
  $("btnLLMPanelClose").addEventListener("click", () => setLLMPanel(false));
  $("llm-panel-mask").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) setLLMPanel(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setLLMPanel(false);
  });
}

/* ---------------- LLM 引擎 ---------------- */

let ftSeriesCache = null;   // 最近一次微调指标序列,供主题切换重绘

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
      $("llmTuneResult").innerHTML = `<div class="callout">状态：<b>${esc(result.status)}</b><br />产物：<code>${esc(result.artifact.split(
        /[\\/]/
      ).pop())}</code></div>`;
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
      /* 序列留档:主题切换后微调画布按新令牌重绘(repaintThemedCanvases) */
      ftSeriesCache = {
        loss: [{ x: epochs, y: result.series.loss, color: "#f87171" }],
        metric: [
          { x: epochs, y: result.series.bleu4, color: "#38bdf8" },
          { x: epochs, y: result.series.rouge1, color: "#34d399" },
          { x: epochs, y: result.series.rougeL, color: "#f59e0b" },
        ],
      };
      drawLineChart($("ftLossCanvas"), ftSeriesCache.loss);
      drawLineChart($("ftMetricCanvas"), ftSeriesCache.metric, { yMin: 0, yMax: 1 });
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
            : `<span class="badge warn" title="${esc(report.error || "")}">失败</span>`;
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
          .map((f, i) => {
            const name = f.split(/[\\/]/).pop();
            return `<figure class="output-frame show" style="margin:0;--gi:${i}"><img src="/outputs/${esc(name)}?t=${Date.now()}" loading="lazy" title="点击放大查看" alt="${esc(name)}" tabindex="0" role="button" aria-label="放大查看 ${esc(name)}" /><figcaption>${esc(name)}</figcaption></figure>`;
          })
          .join("");
        /* 画廊图点击放大:新标签页打开原图(轻量,不引入 lightbox 依赖);Enter/Space 同效 */
        $$("#gallery img").forEach((img) => {
          const open = () => window.open(img.src, "_blank");
          img.addEventListener("click", open);
          img.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              open();
            }
          });
          bindImgFade(img);
        });
      }
      toast(`报告完成：${okCount}/${total}`);
    })
  );
}

/* ---------------- 设置：本地 .env 密钥配置（状态掩码 + 保存热重载） ---------------- */

/* 密钥类配置项用 password 输入框遮蔽；是否允许本机保存由服务端注入的 ZSX_CONFIG 决定 */
const isSecretKey = (key) => /_(KEY|SECRET|CODE|TOKEN)$/.test(key);

function settingsWritable() {
  return !!(window.ZSX_CONFIG && window.ZSX_CONFIG.settingsWritable !== false);
}

function settingsStatusBadge(state) {
  const configured = state.items.filter((item) => item.configured).length;
  const badge = $("settingsBadge");
  badge.className = `badge ${state.ready ? "ok" : "warn"}`;
  badge.textContent = `${configured}/${state.items.length} 项已配置`;
  $("settingsFlag").classList.toggle("hidden", state.missing.length === 0);
}

function settingRowHtml(item) {
  const badge = item.configured
    ? `<span class="badge ok">已配置</span>`
    : `<span class="badge ${item.required ? "warn" : ""}">未配置</span>`;
  const reused = item.reused_from ? `<span class="badge info">复用 ${esc(item.reused_from)}</span>` : "";
  const placeholder = item.configured
    ? `当前 ${esc(item.value_masked)} · 留空表示不修改`
    : item.default
      ? `留空则用默认值 ${esc(item.default)}`
      : "留空则该项走降级链路";
  const clear = item.configured
    ? `<button class="btn subtle sm" data-clear="${esc(item.key)}" title="清空并写入 .env（即时生效，无需重启）">清除</button>`
    : "";
  return `
    <div class="setting-row">
      <div class="setting-head">
        <b>${esc(item.label)}</b><code>${esc(item.key)}</code>${badge}${reused}
      </div>
      <p class="muted">${esc(item.purpose)}</p>
      <p class="muted"><b>申请：</b>${esc(item.apply_entry)}</p>
      <p class="muted"><b>留空降级：</b>${esc(item.degrades_to)}</p>
      <div class="setting-edit">
        <input type="${isSecretKey(item.key) ? "password" : "text"}" data-key="${esc(item.key)}"
               placeholder="${placeholder}" autocomplete="off" spellcheck="false" />
        ${clear}
      </div>
    </div>`;
}

function fallbackRowHtml(label, ready, readyText, degradedText) {
  const badge = ready ? `<span class="badge ok">在线</span>` : `<span class="badge warn">降级</span>`;
  return `<tr><td>${label}</td><td>${badge} ${esc(ready ? readyText : degradedText)}</td></tr>`;
}

function renderSettings(state) {
  $("settingsItems").innerHTML = state.items.map(settingRowHtml).join("");
  $$("#settingsItems [data-clear]").forEach((btn) =>
    btn.addEventListener("click", () => clearSetting(btn.dataset.clear))
  );
  $("settingsEnvFile").textContent =
    `写入目标：${state.env_file}` + (state.env_file_exists ? "" : "（尚不存在，保存时自动创建）") + ` · 模板见 ${state.template_file}`;

  const caps = state.capabilities;
  $("settingsFallbackBody").innerHTML = [
    fallbackRowHtml("LLM 对话 / 需求解析", caps.llm_real, "真实 OpenAI 兼容适配器", "Mock 确定性模板（规则解析仍生效）"),
    fallbackRowHtml("地图底图", caps.amap_map, "高德 JS API 真实底图", "Canvas 离线折线示意"),
    fallbackRowHtml("路线规划", caps.amap_plan, "高德真实路线 + 枢纽内规划", "内置枢纽引擎（偏好感知 A* + RL）"),
  ].join("");

  const notes = [];
  if (state.missing.length) notes.push(`待配置：${state.missing.join("、")}`);
  if (caps.amap_map_blocked_by_security_code) notes.push("已填 JS Key 但缺安全密钥，底图仍会回退 Canvas");
  if (!settingsWritable()) notes.push("当前服务禁用了密钥写入，请改用环境变量或手工编辑 .env");
  const warn = $("settingsWarn");
  warn.classList.toggle("hidden", notes.length === 0);
  warn.innerHTML = notes.map(esc).join("<br />");
  settingsStatusBadge(state);
}

async function refreshSettings() {
  try {
    renderSettings(await api.get("/api/settings"));
  } catch (error) {
    toast(`读取配置失败：${error.message}`, "err");
  }
}

async function postSettings(payload, successText) {
  try {
    const result = await api.post("/api/settings", payload);
    renderSettings(result.state);
    // JS Key 属前端注入项：服务端已热重载，浏览器要重新拉一次页面才拿到底图新配置
    $("settingsReloadHint").classList.toggle("hidden", !result.client_config_stale);
    toast(`${successText}已写入 ${result.env_file} 并热重载${result.backup_file ? "（旧值已备份）" : ""}`);
  } catch (error) {
    toast(`保存失败：${error.message}`, "err");
  }
}

function collectSettingEdits() {
  const payload = {};
  $$("#settingsItems input[data-key]").forEach((input) => {
    const value = input.value.trim();
    if (value) payload[input.dataset.key] = value; // 留空表示不改动该项
  });
  return payload;
}

/* 清除某一项：写空值 → 热重载后对应能力立即回到降级链路 */
function clearSetting(key) {
  if (!window.confirm(`确认清空 ${key} ？该项会立即回退到降级行为（无需重启服务）。`)) return;
  postSettings({ [key]: "" }, `${key} 已清空：`);
}

function initSettings() {
  $("btnReloadSettings").addEventListener("click", refreshSettings);
  $("btnReloadPage").addEventListener("click", () => location.reload());
  $("btnSaveSettings").addEventListener("click", () =>
    withBusy($("btnSaveSettings"), " 保存中", async () => {
      const payload = collectSettingEdits();
      if (!Object.keys(payload).length) {
        toast("没有需要保存的改动（输入框留空即表示不修改）", "err");
        return;
      }
      await postSettings(payload, "配置：");
    })
  );
  refreshSettings();
}

/* ---------------- 智能换乘助手:页面底部 Dock(记忆开合偏好) ---------------- */

function setDockOpen(open) {
  const body = $("chatDockBody");
  const willOpen = open === undefined ? body.hidden : !!open;
  body.hidden = !willOpen;
  $("btnChatDockToggle").classList.toggle("on", willOpen);
  $("app").classList.toggle("dock-open", willOpen);
  try { localStorage.setItem("zsx_ai_open", willOpen ? "1" : "0"); } catch (_) { /* 忽略 */ }
  if (willOpen) {
    const box = $("chatMessages");
    if (box) box.scrollTop = box.scrollHeight;
  }
}

/* ---------------- 悬浮模式:气泡 ↔ 浮动面板 ----------------
   docked(停靠)→ 拖拽对话按钮脱离(>24px)→ bubble(悬浮泡,可拖动)
   → 点击 → panel(浮动面板);× 收回气泡;图钉停靠回底部。
   形变动画以气泡位置为 transform-origin,纯 transform/opacity。 */

const floatState = { mode: "docked", x: 0, y: 0, bx: 0, by: 0 };
let suppressFabClick = false;
let floatBusy = false;

function persistFloatMode(on) {
  try { localStorage.setItem("zsx_float", on ? "1" : "0"); } catch (_) { /* 忽略 */ }
}

function setFloatPos(x, y, save) {
  const dock = $("chatDock");
  const w = dock.getBoundingClientRect().width || 56;
  const h = dock.getBoundingClientRect().height || 56;
  floatState.x = Math.min(Math.max(8, x), Math.max(8, innerWidth - w - 8));
  floatState.y = Math.min(Math.max(8, y), Math.max(8, innerHeight - h - 8));
  dock.style.left = `${floatState.x}px`;
  dock.style.top = `${floatState.y}px`;
  if (save && floatState.mode === "bubble") {
    try { localStorage.setItem("zsx_float_pos", JSON.stringify({ x: floatState.x, y: floatState.y })); } catch (_) { /* 忽略 */ }
  }
}

/* 悬浮状态标记到 #app:切换头部按钮的图标(弹出 ↔ 图钉)与提示 */
function syncFloatUi() {
  const app = $("app");
  const floating = floatState.mode !== "docked";
  app.classList.toggle("float-mode", floating);
  const btn = $("btnChatRedock");
  const title = floating ? "收回至底部" : "变成悬浮球";
  btn.title = title;
  btn.setAttribute("aria-label", title);
}

function enterFloat(cx, cy, opts = {}) {
  if (floatState.mode !== "docked" || floatBusy) return;
  floatState.mode = "bubble";
  const dock = $("chatDock");
  dock.classList.add("float", "float-bubble", "spawning");
  setTimeout(() => dock.classList.remove("spawning"), 300);
  setDockOpen(false);
  setFloatPos(cx - 28, cy - 28, true);
  persistFloatMode(true);
  syncFloatUi();
  if (!opts.silent && !localStorage.getItem("zsx_float_hint")) {
    toast("智能助手已悬浮:拖动可移动,点击展开");
    try { localStorage.setItem("zsx_float_hint", "1"); } catch (_) { /* 忽略 */ }
  }
}

function exitFloat() {
  floatState.mode = "docked";
  const dock = $("chatDock");
  dock.classList.remove("float", "float-bubble", "float-panel", "morphing", "collapsing", "docking-out");
  dock.style.left = dock.style.top = dock.style.transformOrigin = "";
  persistFloatMode(false);
  try { localStorage.removeItem("zsx_float_pos"); } catch (_) { /* 忽略 */ }
  syncFloatUi();
  setDockOpen(true);
  toast("已停靠到底部输入条");
}

function expandFloatPanel(immediate = false) {
  if (floatState.mode !== "bubble" || floatBusy) return;
  const dock = $("chatDock");
  floatState.bx = floatState.x;
  floatState.by = floatState.y;
  dock.classList.remove("float-bubble");
  dock.classList.add("float-panel");
  floatState.mode = "panel";
  const pw = dock.getBoundingClientRect().width || 400;
  const px = Math.min(Math.max(12, floatState.bx + 28 - pw), Math.max(12, innerWidth - pw - 12));
  const py = Math.min(Math.max(12, floatState.by + 28 - 500), Math.max(12, innerHeight - 500));
  setFloatPos(px, py, false);
  dock.style.transformOrigin = `${floatState.bx + 28 - px}px ${floatState.by + 28 - py}px`;
  if (!immediate) {
    dock.classList.add("morphing");
    setTimeout(() => dock.classList.remove("morphing"), 360);
  }
  setDockOpen(true);
}

function collapseFloatToBubble() {
  if (floatState.mode !== "panel" || floatBusy) return;
  floatBusy = true;
  const dock = $("chatDock");
  const fr = dock.getBoundingClientRect();
  dock.style.transformOrigin = `${fr.width - 28}px ${fr.height - 28}px`;
  dock.classList.add("collapsing");
  setTimeout(() => {
    dock.classList.remove("float-panel", "collapsing");
    dock.classList.add("float-bubble", "settling");
    floatState.mode = "bubble";
    dock.style.transformOrigin = "";
    setFloatPos(fr.right - 56, fr.bottom - 56, true);
    setTimeout(() => dock.classList.remove("settling"), 300);
    setDockOpen(false);
    floatBusy = false;
  }, 250);
}

function initFloatDrag() {
  const dock = $("chatDock");
  // 泡泡态:拖容器移动;面板态:拖头部移动
  dock.addEventListener("pointerdown", (e) => {
    if (floatState.mode === "docked" || e.button !== 0) return;
    e.preventDefault();                    // 阻止浏览器原生文字选择/拖拽
    suppressFabClick = false;               // 每次按下重置,避免上一次拖拽的标记误吞本次点击
    const onHead = !!e.target.closest(".dock-head");
    if (floatState.mode === "panel" && !onHead) return;
    const rect = dock.getBoundingClientRect();
    const offX = e.clientX - rect.left;
    const offY = e.clientY - rect.top;
    let moved = false;
    let inZone = false;
    const onMove = (ev) => {
      if (!moved && Math.hypot(ev.clientX - rect.left - offX, ev.clientY - rect.top - offY) < 5) return;
      if (!moved) { moved = true; document.body.classList.add("float-dragging"); }
      dock.classList.add("dragging-float");
      inZone = ev.clientY > innerHeight - 96;
      $("dock-drop-hint").classList.toggle("show", inZone);
      dock.classList.toggle("dock-hint", inZone && floatState.mode === "bubble");
      setFloatPos(ev.clientX - offX, ev.clientY - offY);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      dock.classList.remove("dragging-float", "dock-hint");
      document.body.classList.remove("float-dragging");
      $("dock-drop-hint").classList.remove("show");
      if (moved) suppressFabClick = true;   // 拖拽结束吞掉随后的 click
      if (moved && inZone) magnetDock();    // 在吸附区松手 → 融入底栏
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
  // 停靠态:拖拽对话按钮脱离成为悬浮泡
  $("btnChatDockToggle").addEventListener("pointerdown", (e) => {
    if (floatState.mode !== "docked" || e.button !== 0) return;
    e.preventDefault();
    suppressFabClick = false;
    const sx = e.clientX;
    const sy = e.clientY;
    let detached = false;
    let inZone = false;
    const onMove = (ev) => {
      if (!detached && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 24) return;
      if (!detached) {
        detached = true;
        suppressFabClick = true;
        enterFloat(ev.clientX, ev.clientY);
        document.body.classList.add("float-dragging");
      }
      setFloatPos(ev.clientX - 28, ev.clientY - 28);
      inZone = ev.clientY > innerHeight - 96;
      $("dock-drop-hint").classList.toggle("show", inZone);
      dock.classList.toggle("dock-hint", inZone);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("float-dragging");
      $("dock-drop-hint").classList.remove("show");
      dock.classList.remove("dock-hint");
      if (detached && inZone) magnetDock();  // 拖到底部松手 → 吸附回底栏
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
  // 窗口变化时把悬浮体收回视口
  window.addEventListener("resize", () => {
    if (floatState.mode !== "docked") setFloatPos(floatState.x, floatState.y);
  });
}

/* 磁性吸附:悬浮球滑向底部中央融入底栏;面板直接落回底栏 */
function magnetDock() {
  const dock = $("chatDock");
  dock.classList.remove("dragging-float", "dock-hint");
  document.body.classList.remove("float-dragging");
  $("dock-drop-hint").classList.remove("show");
  if (floatState.mode === "bubble") {
    dock.classList.add("gliding");                                 // 悬浮球滑向底部中央
    setFloatPos((innerWidth - 56) / 2, innerHeight - 76, false);
    setTimeout(() => {
      dock.classList.remove("gliding");
      exitFloat();
      dock.classList.add("redocking");
      setTimeout(() => dock.classList.remove("redocking"), 340);
    }, 250);
  } else {
    exitFloat();                                                   // 面板直接落回底栏
    dock.classList.add("redocking");
    setTimeout(() => dock.classList.remove("redocking"), 340);
  }
}

function initChatDock() {
  let pref = null;
  try { pref = localStorage.getItem("zsx_ai_open"); } catch (_) { /* 忽略 */ }
  let floatOn = false;
  let floatPos = null;
  try {
    floatOn = localStorage.getItem("zsx_float") === "1";
    floatPos = JSON.parse(localStorage.getItem("zsx_float_pos") || "null");
  } catch (_) { /* 忽略 */ }
  if (floatOn) {
    enterFloat(floatPos ? floatPos.x : innerWidth - 84, floatPos ? floatPos.y : innerHeight - 96, { silent: true });
  }
  setDockOpen(pref === "1");                       // 默认收起:只留底部输入条
  if (floatOn && pref === "1") expandFloatPanel(true);   // 恢复为浮动面板
  $("btnChatDockToggle").addEventListener("click", () => {
    if (suppressFabClick) { suppressFabClick = false; return; }
    if (floatState.mode === "bubble") { expandFloatPanel(); return; }
    if (floatState.mode === "panel") { collapseFloatToBubble(); return; }
    setDockOpen();
  });
  $("btnChatCollapse").addEventListener("click", () => {
    if (floatState.mode === "panel") { collapseFloatToBubble(); return; }
    setDockOpen(false);
  });
  $("btnChatRedock").addEventListener("click", () => {
    if (floatState.mode === "docked") {
      enterFloat(innerWidth - 96, innerHeight - 260);   // 悬浮球在落点原地生成脉冲
    } else if (!floatBusy) {
      // 收回至底部:面板向下潜入底栏方向,底栏随后滑入落位
      floatBusy = true;
      const dock = $("chatDock");
      dock.style.transformOrigin = "50% 100%";
      dock.classList.add("docking-out");
      setTimeout(() => {
        exitFloat();
        dock.classList.add("redocking");
        setTimeout(() => { dock.classList.remove("redocking"); floatBusy = false; }, 340);
      }, 240);
    }
  });
  $("chatInput").addEventListener("focus", () => { if ($("chatDockBody").hidden) setDockOpen(true); });
  initFloatDrag();
  initDockResize();
}

/* 面板高度拖拽:顶边手柄上下拖动,记忆到 localStorage,双击恢复默认 */
function initDockResize() {
  const handle = $("dockResizer");
  const root = document.documentElement;
  let saved = null;
  try { saved = Number(localStorage.getItem("zsx_dock_h")); } catch (_) { /* 忽略 */ }
  if (saved >= 240) root.style.setProperty("--dock-h", `${saved}px`);

  handle.addEventListener("mousedown", (event) => {
    event.preventDefault();
    handle.classList.add("dragging");
    document.body.classList.add("row-resizing");
    const startY = event.clientY;
    const startH = $("chatDockBody").getBoundingClientRect().height;
    const onMove = (e) => {
      const h = Math.min(window.innerHeight * 0.78,
                         Math.max(260, Math.round(startH + (startY - e.clientY))));
      root.style.setProperty("--dock-h", `${h}px`);
    };
    const onUp = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("row-resizing");
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      const h = parseInt(root.style.getPropertyValue("--dock-h"), 10);
      try { localStorage.setItem("zsx_dock_h", String(h)); } catch (_) { /* 忽略 */ }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
  handle.addEventListener("dblclick", () => {
    root.style.removeProperty("--dock-h");
    try { localStorage.removeItem("zsx_dock_h"); } catch (_) { /* 忽略 */ }
  });
}

/* ---------------- 侧栏收起 / 展开(图标栏形态,记忆状态) ---------------- */

function setSideCollapsed(collapsed) {
  $("app").classList.toggle("side-collapsed", collapsed);
  try { localStorage.setItem("zsx_side_collapsed", collapsed ? "1" : "0"); } catch (_) { /* 忽略 */ }
}

function initSideToggle() {
  let pref = null;
  try { pref = localStorage.getItem("zsx_side_collapsed"); } catch (_) { /* 忽略 */ }
  setSideCollapsed(pref === "1");
  $("side-toggle").addEventListener("click", () =>
    setSideCollapsed(!$("app").classList.contains("side-collapsed")));
}

/* ---------------- 命令面板(Ctrl/⌘+K) ---------------- */

let cmdkItems = [];
let cmdkSel = 0;

async function cmdkActions() {
  /* 图标与全站 sprite 体系统一(不再用 emoji);新增 symbol 时 icons.svg 需升版本号 */
  const ico = (id) => `<svg class="ic" aria-hidden="true"><use href="/static/assets/icons.svg?v=6.0#${id}"/></svg>`;
  const navs = [
    ["overview", "概览", "i-overview"],
    ["plan", "路线规划", "i-plan"],
    ["navigation", "枢纽导航", "i-hub"],
    ["flow", "客流面板", "i-flow"],
    ["rl", "RL 智能体", "i-rl"],
    ["reports", "分析报告", "i-reports"],
    ["settings", "设置（密钥配置）", "i-sliders"],
  ].map(([tab, label, iconId]) => ({
    ico: ico(iconId), label, cat: "页面",
    run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab(tab); },
  }));
  const cmds = [
    { ico: ico("i-chat"), label: "智能换乘助手(底部对话)", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); setDockOpen(true); $("chatInput").focus(); } },
    { ico: ico("i-moon"), label: "切换明暗主题", cat: "命令",
      run: () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark") },
    { ico: ico("i-download"), label: "加载最新策略权重", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab("rl"); setTimeout(() => $("btnRLPolicy").click(), 250); } },
    { ico: ico("i-reports"), label: "运行全部报告", cat: "命令",
      run: () => { if ($("app").classList.contains("hidden")) enterApp(); switchTab("reports"); setTimeout(() => $("btnReportsAll").click(), 250); } },
    { ico: ico("i-phone"), label: "打开移动端 PWA", cat: "命令", run: () => window.open("/mobile", "_blank") },
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
  initSettings();
  initChatDock();
  initSideToggle();
  initNavPill();
  initCmdk();
  initLLMPanel();
  switchTab(lastTab() || "overview");

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
  initWelcomeReveals();
  $$(".nav-item[data-tab]").forEach((item) =>
    item.addEventListener("click", () => switchTab(item.dataset.tab)));
  showView("welcome");
  refreshTopbar(); // 欢迎页统计与服务状态(应用壳元素隐藏但已存在)
  /* 顶栏滚动投影:主区内容滑到顶栏之下时加深(被动监听,阈值切换) */
  const scrollArea = $("scroll-area");
  const topbar = document.querySelector(".topbar");
  if (scrollArea && topbar) {
    scrollArea.addEventListener("scroll", () => {
      topbar.classList.toggle("scrolled", scrollArea.scrollTop > 8);
    }, { passive: true });
  }
}

/* 欢迎页下半屏内容滚动显现:只播一次,进过视口即标记,不打扰回读 */
function initWelcomeReveals() {
  const items = $$(".reveal");
  if (!items.length) return;
  const showAll = () => items.forEach((el) => el.classList.add("in"));
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) { showAll(); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      io.unobserve(entry.target);
    });
  }, { threshold: 0.15 });
  items.forEach((el, i) => {
    el.style.transitionDelay = `${Math.min(i * 70, 210)}ms`;
    io.observe(el);
  });
}

document.addEventListener("DOMContentLoaded", boot);
