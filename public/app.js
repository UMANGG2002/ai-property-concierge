/* Shared shell: API base, helpers, icons, theme, sidebar, status. Loaded in <head> by every page. */
(() => {
  const API = (new URLSearchParams(location.search).get("api") || window.API_BASE || "").replace(/\/$/, "") + "/api";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);
  const fmt = (n) => (typeof n === "number" ? n.toLocaleString("en-US", { maximumFractionDigits: 0 }) : n);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { } },
  };

  // ---------------- icons (24px stroke, Lucide-style paths drawn here)
  const P = {
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    send: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    spark: '<path d="M12 3l1.8 4.9L18.7 9.7l-4.9 1.8L12 16.4l-1.8-4.9L5.3 9.7l4.9-1.8z"/><path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z"/>',
    chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    bed: '<path d="M3 18V7M3 13h18v5M21 13a3 3 0 0 0-3-3h-7v3"/><circle cx="7" cy="11" r="1.5"/>',
    area: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    home: '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    pin: '<path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    ext: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    chevL: '<path d="M15 18l-6-6 6-6"/>',
    chevR: '<path d="M9 18l6-6-6-6"/>',
    chevD: '<path d="M6 9l6 6 6-6"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
    building: '<path d="M4 21V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v16M15 9h4a1 1 0 0 1 1 1v11M2 21h20M8 8h3M8 12h3M8 16h3"/>',
    tag: '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="8" cy="8" r="1.5"/>',
    shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    trend: '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    star: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M17 6l3 3"/>',
  };
  const icon = (n, cls = "") => `<svg class="ic ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[n] || ""}</svg>`;
  const hydrateIcons = (root = document) => root.querySelectorAll("[data-icon]").forEach((el) => {
    if (el.dataset.iconDone) return; el.dataset.iconDone = 1; el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon));
  });

  // ---------------- theme (applied before paint)
  const applyTheme = (t) => { document.documentElement.dataset.theme = t; };
  const preferred = store.get("theme") || "light";
  applyTheme(preferred);

  function header(page) {
    const el = document.getElementById("site-head"); if (!el) return;
    const link = (href, label, key) => `<a href="${href}" class="${page === key ? "on" : ""}">${label}</a>`;
    el.innerHTML = `
      <a class="wordmark" href="/" aria-label="Concierge home"><span class="mono">C</span><span>Concierge</span></a>
      <nav class="site-nav" id="site-nav">
        ${link("/", "Concierge", "chat")}${link("/#collections", "Collections", "coll")}${link("/about.html", "Architecture", "about")}${link("/about.html#benchmark", "Benchmark", "bench")}
      </nav>
      <div class="head-actions">
        <button class="icon-btn" type="button" id="theme-btn" aria-label="Toggle dark mode"></button>
        <a class="btn-line only-desk" href="/" id="new-btn">New enquiry</a>
        <button class="icon-btn only-mobile" type="button" id="menu-btn" aria-label="Menu">${icon("menu")}</button>
      </div>`;
    const tb = el.querySelector("#theme-btn");
    const sync = () => { tb.innerHTML = icon(document.documentElement.dataset.theme === "dark" ? "sun" : "moon"); };
    sync();
    tb.onclick = () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; applyTheme(t); store.set("theme", t); sync(); };
    el.querySelector("#menu-btn").onclick = () => el.classList.toggle("open");
    const onScroll = () => el.classList.toggle("solid", (document.getElementById("scroller")?.scrollTop || window.scrollY) > 40 || document.body.classList.contains("chatting"));
    (document.getElementById("scroller") || window).addEventListener("scroll", onScroll, { passive: true }); window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
    header.sync = onScroll;
  }

  let healthP = null;
  const health = () => (healthP ||= fetch(`${API}/health`).then((r) => r.json()));

  function toast(msg) {
    const t = document.getElementById("toast"); if (!t) return;
    t.textContent = msg; t.hidden = false; t.classList.add("on");
    clearTimeout(toast._t); toast._t = setTimeout(() => { t.classList.remove("on"); setTimeout(() => (t.hidden = true), 200); }, 1800);
  }

  window.UI = { API, esc, safeUrl, fmt, icon, hydrateIcons, health, toast, store, header: () => header.sync?.() };
  document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.dataset.page;
    header(location.hash === "#benchmark" && page === "about" ? "bench" : page);
    hydrateIcons();
  });
})();
