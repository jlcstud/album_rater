(function () {
  if (typeof document === "undefined") {
    return;
  }

  function openSpotify(anchor) {
    const uri = anchor.getAttribute("data-spotify-uri");
    const web = anchor.getAttribute("data-spotify-web");
    if (!uri) {
      return;
    }

    const fallbackDelay = 1200;
    let fallbackTimer = null;

    const clearFallback = () => {
      if (fallbackTimer !== null) {
        clearTimeout(fallbackTimer);
        fallbackTimer = null;
      }
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        clearFallback();
      }
    };

    const launchFallback = () => {
      clearFallback();
      if (web) {
        window.open(web, "_blank", "noopener");
      }
    };

    fallbackTimer = window.setTimeout(launchFallback, fallbackDelay);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    const iframe = document.createElement("iframe");
    iframe.style.display = "none";
    iframe.setAttribute("aria-hidden", "true");
    iframe.src = uri;
    document.body.appendChild(iframe);

    window.setTimeout(() => {
      if (document.body.contains(iframe)) {
        document.body.removeChild(iframe);
      }
    }, fallbackDelay * 2);
  }

  document.addEventListener(
    "click",
    (event) => {
      const anchor = event.target.closest("a[data-spotify-uri]");
      if (!anchor) {
        return;
      }
      event.preventDefault();
      openSpotify(anchor);
    },
    false
  );
})();
