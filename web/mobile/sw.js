const CACHE_NAME = "zhishuxing-mobile-v15";
const ASSETS = [
  "./mobile_app.html",
  "./manifest.webmanifest",
  "./map.png",
  "./icon-192.png",
  "./icon-512.png",
  "./icon-maskable-512.png",
  "./apple-touch-icon.png",
  "./logo-mark.svg",
  "./favicon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  // API 请求走网络优先，避免离线拿到过期规划结果
  const isApi = url.pathname.startsWith("/api/") || url.pathname.startsWith("/outputs/");
  // 壳文档(mobile_app.html/manifest)同样网络优先:发布新版后用户无需等 CACHE_NAME 手工 bump
  // 才能拿到新壳;离线时才回退缓存。图标等静态资产保持 cache-first(内容随版本名变化)。
  const isShell = url.pathname.endsWith("/mobile_app.html") || url.pathname.endsWith("/manifest.webmanifest");
  event.respondWith(
    (isApi || isShell
      ? fetch(event.request)
          .then((response) => {
            if (isShell && response.ok) {
              const copied = response.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copied));
            }
            return response;
          })
          .catch(() => caches.match(event.request))
      : caches.match(event.request).then(
          (cached) =>
            cached ||
            fetch(event.request).then((response) => {
              if (response.ok) {
                const copied = response.clone();
                caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copied));
              }
              return response;
            })
        )
    ).catch(() => caches.match("./mobile_app.html"))
  );
});
