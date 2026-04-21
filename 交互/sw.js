const CACHE_NAME = "zhishuxing-mobile-v1";
const ASSETS = [
  "./mobile_app.html",
  "./manifest.webmanifest",
  "./地图.png",
  "./VR1.png",
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
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request)
        .then((response) => {
          if (event.request.method === "GET" && response.ok) {
            const copied = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copied));
          }
          return response;
        })
        .catch(() => caches.match("./mobile_app.html"));
    })
  );
});
