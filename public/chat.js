/* Chat page: ask → /api/chat, render answer + property cards + question-aware highlights; detail drawer from /api/property/{id}. */
(() => {
  const { API, esc, safeUrl, fmt, icon, hydrateIcons, toast } = window.UI;
  const $ = (s, el = document) => el.querySelector(s);
  const KIND = { unit_listing: "Resale listing", development_project: "Development", one_of_one: "One of One", partner: "Brand partner", info_page: "DarGlobal", article: "Article" };
  const PROPERTY_KINDS = ["unit_listing", "development_project", "one_of_one"];
  const cards = new Map();   // property id -> card payload from /api/chat

  const DEBUG = new URLSearchParams(location.search).has("debug");

  // ---------------- landing: hero, prompts, collections
  const HERO = ["https://cdn.darglobal.co.uk/TTD_exterior_10_8a7711609b.webp", "https://cdn.darglobal.co.uk/coastal_2_f027071962.jpg", "https://cdn.darglobal.co.uk/p7_e5624544a4.jpg"];
  const hm = $("#hero-media");
  hm.innerHTML = HERO.map((u, i) => `<div class="slide ${i === 0 ? "on" : ""}" style="background-image:url('${u}')"></div>`).join("");
  let hs = 0; setInterval(() => { const sl = hm.querySelectorAll(".slide"); sl[hs].classList.remove("on"); hs = (hs + 1) % sl.length; sl[hs].classList.add("on"); }, 6500);

  const PROMPTS = ["Why invest in Marea?", "Villas in AIDA, Oman", "4 bedroom villa in North Riyadh under 3M", "Amenities at Les Vagues", "شقة للبيع في الرياض"];
  $("#hero-prompts").innerHTML = PROMPTS.map((t) => `<button type="button" data-ask="${esc(t)}">${esc(t)}</button>`).join("");

  const COLLECTIONS = [
    ["Trump International Hotel & Tower", "Dubai, UAE", "https://cdn.darglobal.co.uk/TTD_Executive_Suite_12ca223803.webp", "Tell me about Trump International Hotel and Tower Dubai"],
    ["Marea, Interiors by Missoni", "Costa del Sol, Spain", "https://cdn.darglobal.co.uk/p6_58dbba15d4.jpg", "Why should I invest in Marea?"],
    ["AIDA Coastal Villas", "Muscat, Oman", "https://cdn.darglobal.co.uk/Dar_Global_AIDA_Villa_Living_1_a66fbb87d7.jpg", "Tell me about AIDA Coastal Investment Villas"],
    ["Les Vagues by Elie Saab", "Dubai, UAE", "https://cdn.darglobal.co.uk/LV_1_71ea05cd0d.jpg", "What amenities does Les Vagues have?"],
    ["DaVinci Tower by Pagani", "Dubai, UAE", null, "Tell me about DaVinci Tower by Pagani"],
    ["Homes for sale in Riyadh", "Wasalt resale listings", null, "Show me villas for sale in North Riyadh"],
  ];
  $("#coll-grid").innerHTML = COLLECTIONS.map(([t, l, img, q], i) => `<button class="coll ${i === 0 ? "wide" : ""}" type="button" data-ask="${esc(q)}">
      <span class="coll-img" ${img ? `style="background-image:url('${img}')"` : ""}>${img ? "" : icon(i === 5 ? "home" : "building")}</span>
      <span class="coll-txt"><small>${esc(l)}</small><b>${esc(t)}</b><span class="coll-cta">Discover ${icon("chevR")}</span></span></button>`).join("");
  fetch(`${API}/stats`).then((r) => r.json()).then((s) => {
    $("#coll-stats").innerHTML = [[s.property_records, "Residences & listings"], [Object.keys(s.countries || {}).length, "Countries"], [s.by_source?.wasalt, "Riyadh homes for sale"], ["EN · عربي", "Ask in either language"]]
      .filter(([v]) => v != null).map(([v, l]) => `<div><b>${typeof v === "number" ? fmt(v) : esc(v)}</b><span>${esc(l)}</span></div>`).join("");
  }).catch(() => { });

  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-ask]");
    if (a) { e.preventDefault(); ask(a.dataset.ask); }
  });

  // ---------------- composers (hero + docked)
  const thread = $("#thread"), scroller = $("#scroller"), q = $("#q"), send = $("#send");
  document.querySelectorAll("[data-composer]").forEach((f) => {
    const ta = $("textarea", f), btn = $("button", f);
    const grow = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 180) + "px"; btn.classList.toggle("ready", !!ta.value.trim()); };
    ta.addEventListener("input", grow); ta._grow = grow;
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); f.requestSubmit(); } });
    f.addEventListener("submit", (e) => { e.preventDefault(); const v = ta.value.trim(); if (v && !send.disabled) { ta.value = ""; grow(); ask(v); } });
  });
  const grow = () => q._grow?.();
  document.addEventListener("keydown", (e) => { if (e.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { e.preventDefault(); (document.body.classList.contains("chatting") ? q : $(".hero-search textarea")).focus(); } });
  const toBottom = () => scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });

  async function ask(text) {
    if (!document.body.classList.contains("chatting")) {
      document.body.classList.add("chatting"); $("#intro")?.remove(); $("#collections")?.remove(); $("#dock").hidden = false; window.UI.header();
    }
    send.disabled = true;
    thread.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${esc(text)}</div></div>`);
    const holder = document.createElement("div");
    holder.className = "msg bot";
    holder.innerHTML = `<div class="avatar">C</div><div class="bot-body">
        <div class="thinking"><span></span><span></span><span></span><em>Reviewing residences…</em></div>
        <div class="sk-grid"><div class="sk"></div><div class="sk"></div></div></div>`;
    thread.appendChild(holder); toBottom();
    const body = { query: text, top_k: 5, use_llm: true, debug: DEBUG };
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 75000);
      const r = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      clearTimeout(timer);
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail ? (Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail) : `HTTP ${r.status}`);
      (data.properties || []).forEach((p) => cards.set(p.id, p));
      holder.innerHTML = `<div class="avatar">C</div><div class="bot-body">${renderAnswer(data, body.debug)}</div>`;
      wire(holder, data);
    } catch (err) {
      holder.innerHTML = `<div class="avatar err">${icon("info")}</div><div class="bot-body"><div class="alert">
        <b>We couldn't complete that request.</b> ${esc(err.name === "AbortError" ? "It took longer than expected." : "Please try again in a moment.")}
        <button class="btn sm" type="button">Try again</button></div></div>`;
      $("button", holder).onclick = () => { holder.previousElementSibling?.remove(); holder.remove(); ask(text); };
    } finally {
      send.disabled = false; q.focus();
      holder.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // ---------------- answer
  function mdLite(s, n) {
    const lines = esc(s).split(/\n/);
    let html = "", list = null;
    for (const raw of lines) {
      let l = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)_(.+?)_(?=\s|$)/g, "$1<i>$2</i>");
      l = l.replace(/\[(\d{1,2})\]/g, (m, d) => (+d >= 1 && +d <= n ? `<button class="cite" type="button" data-cite="${d}" title="Jump to source ${d}">${d}</button>` : m));
      const ul = /^\s*[-•*]\s+/.test(l), ol = /^\s*\d+[.)]\s+/.test(l);
      if (ul || ol) {
        const tag = ul ? "ul" : "ol";
        if (list !== tag) { if (list) html += `</${list}>`; html += `<${tag}>`; list = tag; }
        html += `<li>${l.replace(/^\s*([-•*]|\d+[.)])\s+/, "")}</li>`; continue;
      }
      if (list) { html += `</${list}>`; list = null; }
      if (/^#{1,4}\s/.test(l)) { html += `<h4>${l.replace(/^#+\s/, "")}</h4>`; continue; }
      if (l.trim()) html += `<p>${l}</p>`;
    }
    return html + (list ? `</${list}>` : "");
  }

  function renderAnswer(d, debug) {
    const props = d.properties || [];
    const relaxed = (d.relaxed || []).length ? `<div class="notice">${icon("info")}<span>Nothing matched every detail, so these are the closest alternatives.</span></div>` : "";
    const dbg = debug ? `<details class="dbg"><summary>Diagnostics</summary><pre>${esc(JSON.stringify({ mode: d.answer_mode, model: d.model, store: d.store, timings: d.timings, understood: d.understood, filters_used: d.filters_used, relaxed: d.relaxed, grounding: d.grounding, errors: d.errors }, null, 2))}</pre><pre>${esc(d.context || "")}</pre></details>` : "";
    const top = props[0];
    const hl = top && PROPERTY_KINDS.concat("partner").includes(top.record_kind) ? highlights(top) : "";
    return `
      ${relaxed}
      <div class="answer">${mdLite(d.answer || "", Math.min(props.length, 8))}</div>
      <div class="answer-tools"><button class="tool copy" type="button">${icon("copy")}Copy</button>${props.length ? `<span>${props.length} ${props.length === 1 ? "residence" : "residences"} referenced</span>` : ""}</div>
      ${props.length ? `<div class="cards">${props.map((p, i) => card(p, i + 1)).join("")}</div>` : ""}
      ${hl ? `<div class="hl-panel"><div class="hl-head"><div><small>In focus</small><b>${esc(top.title)}</b></div><button class="btn-line" type="button" data-open="${esc(top.id)}">View residence</button></div>${hl}</div>` : ""}
      ${dbg}`;
  }

  const priceText = (p) => (p.price_published && p.price ? `${fmt(p.price)} ${p.currency || "SAR"}` : null);
  const range = (a, b, unit) => (a == null ? null : (b != null && b !== a ? `${fmt(a)}–${fmt(b)}` : fmt(a)) + unit);
  const locText = (p) => [p.district, p.community, p.zone, p.city, p.country].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(", ");
  const srcName = (s) => (s === "wasalt" ? "Wasalt" : "DarGlobal");

  function placeholder(p) {
    return `<div class="ph">${icon(p.source === "wasalt" ? "home" : "building")}<span>${p.source === "wasalt" ? "No photos on this listing" : "No photos available"}</span></div>`;
  }

  function card(p, n) {
    const imgs = (p.images || []).map(safeUrl).filter(Boolean);
    const isProp = PROPERTY_KINDS.includes(p.record_kind);
    const price = priceText(p);
    const beds = range(p.bedrooms_min, p.bedrooms_max, " bd");
    const area = range(p.area_min, p.area_max, " m²") || (p.land_area ? `${fmt(p.land_area)} m² land` : null);
    const types = (p.property_types || []).map((t) => t.replace(/_/g, " ")).slice(0, 2).join(" · ");
    const loc = locText(p);
    return `<article class="pcard" id="card-${n}" data-id="${esc(p.id)}" tabindex="0" aria-label="${esc(p.title)}">
      <div class="pmedia" data-images='${esc(JSON.stringify(imgs))}' data-src="${esc(p.source)}">
        ${imgs.length ? `<img src="${esc(imgs[0])}" alt="${esc(p.title)}" loading="lazy">` : placeholder(p)}
        <span class="pnum">${n}</span>
        <span class="psrc ${esc(p.source)}">${srcName(p.source)}</span>
        ${imgs.length > 1 ? `<button class="pnav l" type="button" aria-label="Previous photo">${icon("chevL")}</button><button class="pnav r" type="button" aria-label="Next photo">${icon("chevR")}</button><span class="pcount">${icon("image")}<b>1</b>/${imgs.length}</span>` : ""}
      </div>
      <div class="pbody">
        ${isProp ? `<div class="pprice ${price ? "" : "muted"}">${esc(price || "Price on request")}</div>` : `<div class="pkind">${esc(KIND[p.record_kind] || p.record_kind)}</div>`}
        <h3 class="ptitle">${esc(p.title)}</h3>
        ${loc ? `<div class="ploc">${icon("pin")}<span>${esc(loc)}</span></div>` : ""}
        <div class="pfacts">
          ${beds ? `<span>${icon("bed")}${esc(beds)}</span>` : ""}${area ? `<span>${icon("area")}${esc(area)}</span>` : ""}${types ? `<span>${icon("home")}${esc(types)}</span>` : ""}
        </div>
        <div class="ptags">
          ${isProp ? `<span class="tag">${esc(KIND[p.record_kind])}</span>` : ""}
          ${p.status ? `<span class="tag">${esc(p.status.replace(/_/g, " "))}</span>` : ""}
          ${p.partner ? `<span class="tag gold">${esc(p.partner)}</span>` : ""}
        </div>
      </div>
      <div class="pfoot">
        <button class="btn-line sm" type="button" data-open="${esc(p.id)}">View residence</button>
        ${safeUrl(p.url) ? `<a class="link" href="${esc(p.url)}" target="_blank" rel="noopener">${srcName(p.source)}${icon("ext")}</a>` : ""}
      </div>
    </article>`;
  }

  // question-aware extras for the top result (from the answer payload)
  function highlights(p) {
    const h = p.highlights || {};
    const out = [];
    if (h.why_invest) {
      const w = h.why_invest;
      out.push(`<section class="hl"><h4>${icon("trend")}${esc(w.title || "Why invest")}</h4>${w.intro ? `<p>${esc(clip(w.intro, 420))}</p>` : ""}
        ${(w.bullets || []).length ? `<ul class="checks">${w.bullets.slice(0, 6).map((b) => `<li>${icon("check")}<span>${esc(b)}</span></li>`).join("")}</ul>` : ""}</section>`);
    }
    if ((h.features || []).length) out.push(`<section class="hl"><h4>${icon("star")}Key features</h4><div class="tiles">${h.features.map(tile).join("")}</div></section>`);
    if ((h.amenities || []).length) out.push(`<section class="hl"><h4>${icon("check")}Amenities</h4><div class="pills">${h.amenities.map((a) => `<span class="pill">${esc(a)}</span>`).join("")}</div></section>`);
    if ((h.sections || []).length) out.push(`<section class="hl"><h4>${icon("building")}About this residence</h4><div class="tiles">${h.sections.map((s) => tile({ title: s.title, detail: clip(s.text || "", 160), image: s.image })).join("")}</div></section>`);
    if (h.location) out.push(`<section class="hl"><h4>${icon("pin")}Location</h4>${h.location.text ? `<p>${esc(clip(h.location.text, 400))}</p>` : ""}${(h.location.highlights || []).length ? `<div class="pills">${h.location.highlights.map((x) => `<span class="pill">${esc(x)}</span>`).join("")}</div>` : ""}</section>`);
    if ((h.faqs || []).length) out.push(`<section class="hl"><h4>${icon("help")}Related questions</h4><div class="faq">${h.faqs.map((f, i) => `<details ${i === 0 ? "open" : ""}><summary>${esc(f.q)}${icon("chevD")}</summary><p>${esc(f.a)}</p></details>`).join("")}</div></section>`);
    if ((h.fixtures || []).length) out.push(`<section class="hl"><h4>${icon("home")}Fixtures</h4><div class="pills">${h.fixtures.map((a) => `<span class="pill">${esc(a)}</span>`).join("")}</div></section>`);
    if (h.licence) out.push(licence(h.licence));
    return out.join("");
  }
  const clip = (s, n) => (s.length > n ? s.slice(0, n).replace(/\s+\S*$/, "") + "…" : s);
  function tile(f) {
    const img = safeUrl(f.image);
    return `<figure class="tile">${img ? `<img src="${esc(img)}" alt="${esc(f.title || "")}" loading="lazy" data-zoom="${esc(img)}" onerror="this.remove()">` : ""}<figcaption><b>${esc(f.title || "")}</b>${f.detail ? `<span>${esc(f.detail)}</span>` : ""}</figcaption></figure>`;
  }
  function licence(L) {
    const rows = [["REGA ad licence", L.rega_ad_license], ["Ad status", L.ad_status], ["Valid until", L.license_expiry], ["FAL licence", L.fal_license], ["Broker", L.broker_company]].filter(([, v]) => v);
    return rows.length ? `<section class="hl"><h4>${icon("shield")}Licence</h4><dl class="dl">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></section>` : "";
  }

  // ---------------- interactions
  function wire(root) {
    hydrateIcons(root);
    root.querySelectorAll(".pmedia").forEach((m) => {
      let imgs = []; try { imgs = JSON.parse(m.dataset.images || "[]"); } catch { }
      if (!imgs.length) return;
      let i = 0;
      const img = $("img", m);
      const show = (k) => { i = (k + imgs.length) % imgs.length; img.src = imgs[i]; const c = $(".pcount b", m); if (c) c.textContent = i + 1; };
      img.addEventListener("error", () => { imgs.splice(i, 1); if (imgs.length) show(i); else { img.remove(); m.insertAdjacentHTML("afterbegin", placeholder({ source: m.dataset.src })); m.querySelectorAll(".pnav,.pcount").forEach((x) => x.remove()); } });
      $(".l", m)?.addEventListener("click", (e) => { e.stopPropagation(); show(i - 1); });
      $(".r", m)?.addEventListener("click", (e) => { e.stopPropagation(); show(i + 1); });
      img.addEventListener("click", (e) => { e.stopPropagation(); openLightbox(imgs, i); });
    });
    root.querySelectorAll(".pcard").forEach((c) => {
      c.addEventListener("click", (e) => { if (!e.target.closest("a,button,img")) openDrawer(c.dataset.id); });
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target === c) openDrawer(c.dataset.id); });
    });
    root.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => openDrawer(b.dataset.open)));
    root.querySelectorAll("[data-zoom]").forEach((im) => im.addEventListener("click", () => openLightbox([im.dataset.zoom], 0)));
    root.querySelectorAll("[data-cite]").forEach((b) => b.addEventListener("click", () => {
      const c = root.querySelector(`#card-${b.dataset.cite}`);
      if (c) { c.scrollIntoView({ behavior: "smooth", block: "center" }); c.classList.add("flash"); setTimeout(() => c.classList.remove("flash"), 1500); }
    }));
    $(".copy", root)?.addEventListener("click", () => {
      const t = $(".answer", root)?.innerText || "";
      navigator.clipboard?.writeText(t).then(() => toast("Answer copied"), () => toast("Copy failed"));
    });
  }

  // ---------------- detail drawer
  const drawer = $("#drawer"), dscrim = $("#drawer-scrim"), dbody = $("#dr-body");
  const detailCache = new Map();
  let lastFocus = null;

  async function openDrawer(id) {
    if (!id) return;
    lastFocus = document.activeElement;
    const c = cards.get(id) || {};
    dbody.innerHTML = `<div class="sk hero"></div><div class="dr-pad"><div class="sk line"></div><div class="sk line short"></div><div class="sk-grid"><div class="sk"></div><div class="sk"></div></div></div>`;
    $("#dr-foot").innerHTML = "";
    drawer.hidden = dscrim.hidden = false;
    requestAnimationFrame(() => { drawer.classList.add("on"); dscrim.classList.add("on"); });
    document.body.classList.add("locked");
    $("#dr-close").focus();
    try {
      if (!detailCache.has(id)) {
        const r = await fetch(`${API}/property/${encodeURIComponent(id).replace(/%3A/gi, ":").replace(/%2F/gi, "/")}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        detailCache.set(id, await r.json());
      }
      renderDrawer(detailCache.get(id), c);
    } catch (e) {
      dbody.innerHTML = `<div class="alert">Couldn't load details (${esc(e.message)}).</div>`;
    }
  }

  function closeDrawer() {
    drawer.classList.remove("on"); dscrim.classList.remove("on"); document.body.classList.remove("locked");
    setTimeout(() => { drawer.hidden = dscrim.hidden = true; }, 220);
    lastFocus?.focus?.();
  }
  $("#dr-close").onclick = closeDrawer; dscrim.onclick = closeDrawer;

  function renderDrawer(d, c) {
    const m = d.metadata || {}, rich = d.rich || {};
    const gallery = (rich.gallery || []).map((g) => ({ url: safeUrl(g.url || g), alt: g.alt || d.title })).filter((g) => g.url);
    const imgs = gallery.length ? gallery : (d.images || []).map(safeUrl).filter(Boolean).map((u) => ({ url: u, alt: d.title }));
    const urls = imgs.map((g) => g.url);
    const price = m.price_published && m.price ? `${fmt(m.price)} ${m.currency || "SAR"}` : "Price on request";
    const facts = [
      ["tag", "Price", price],
      m.price_per_sqm ? ["trend", "Price / m²", `${fmt(m.price_per_sqm)} ${m.currency || "SAR"}`] : null,
      ["bed", "Bedrooms", range(m.bedrooms_min, m.bedrooms_max, "")],
      ["area", "Built area", range(m.area_min, m.area_max, " m²")],
      m.land_area ? ["area", "Land", `${fmt(m.land_area)} m²`] : null,
      ["home", "Type", (m.property_types || []).map((t) => t.replace(/_/g, " ")).join(", ")],
      m.status ? ["clock", "Status", m.status.replace(/_/g, " ")] : null,
      m.purpose ? ["key", "Purpose", m.purpose] : null,
      m.partner ? ["star", "Brand partner", m.partner] : null,
      m.completion_date ? ["clock", "Completion", m.completion_date] : null,
    ].filter((r) => r && r[2]);

    const tabs = [];
    const about = rich.about || d.summary;
    const features = (rich.key_features || []).length ? `<div class="tiles">${rich.key_features.map((f) => tile({ title: f.title, detail: f.detail || clip(f.text || "", 140), image: f.image })).join("")}</div>` : "";
    const sections = (rich.sections || []).length ? `<div class="tiles">${rich.sections.map((s) => tile({ title: s.title, detail: clip(s.text || "", 180), image: s.image })).join("")}</div>` : "";
    if (about || features || sections) tabs.push(["Overview", `${about ? `<div class="prose">${about.split(/\n+/).map((x) => `<p>${esc(x)}</p>`).join("")}</div>` : ""}${features ? `<h4 class="sub">Key features</h4>${features}` : ""}${sections ? `<h4 class="sub">The residence</h4>${sections}` : ""}`]);
    const amen = [...(rich.amenities || [])];
    if (amen.length || (rich.fixtures || []).length || (rich.utilities || []).length) tabs.push(["Amenities", [
      amen.length ? `<ul class="checks cols">${amen.map((a) => `<li>${icon("check")}<span>${esc(a)}</span></li>`).join("")}</ul>` : "",
      (rich.fixtures || []).length ? `<h4 class="sub">Fixtures</h4><div class="pills">${rich.fixtures.map((a) => `<span class="pill">${esc(a)}</span>`).join("")}</div>` : "",
      (rich.utilities || []).length ? `<h4 class="sub">Utilities</h4><div class="pills">${rich.utilities.map((a) => `<span class="pill">${esc(a)}</span>`).join("")}</div>` : ""].join("")]);
    if (rich.why_invest) { const w = rich.why_invest; tabs.push(["Why invest", `${w.title ? `<h4 class="sub first">${esc(w.title)}</h4>` : ""}${w.intro ? `<div class="prose"><p>${esc(w.intro)}</p></div>` : ""}${(w.bullets || []).length ? `<ul class="checks">${w.bullets.map((b) => `<li>${icon("check")}<span>${esc(b)}</span></li>`).join("")}</ul>` : ""}`]); }
    if ((rich.faqs || []).length) tabs.push([`FAQs <small>${rich.faqs.length}</small>`, `<div class="faq">${rich.faqs.map((f, i) => `<details ${i === 0 ? "open" : ""}><summary>${esc(f.q)}${icon("chevD")}</summary><p>${esc(f.a)}</p></details>`).join("")}</div>`]);
    if (rich.location && (rich.location.text || (rich.location.highlights || []).length || rich.location.lat)) {
      const L = rich.location;
      const map = L.lat && L.lng ? `<a class="btn-line sm" href="https://www.google.com/maps?q=${encodeURIComponent(L.lat + "," + L.lng)}" target="_blank" rel="noopener">${icon("pin")}Open in Google Maps</a>` : "";
      tabs.push(["Location", `${L.text ? `<div class="prose"><p>${esc(L.text)}</p></div>` : ""}${(L.highlights || []).length ? `<ul class="checks">${L.highlights.map((x) => `<li>${icon("pin")}<span>${esc(x)}</span></li>`).join("")}</ul>` : ""}${map}`]);
    }
    if (rich.licence) { const l = licence(rich.licence); if (l) tabs.push(["Licence", l.replace(/<h4>.*?<\/h4>/, "")]); }
    if (!tabs.length && d.full_text) tabs.push(["Details", `<div class="prose">${d.full_text.split(/\n+/).slice(0, 40).map((x) => `<p>${esc(x)}</p>`).join("")}</div>`]);

    const loc = locText({ ...m, ...c });
    dbody.innerHTML = `
      ${urls.length ? `<div class="gal">
          <div class="gal-main"><img src="${esc(urls[0])}" alt="${esc(imgs[0].alt)}"><span class="pcount">${icon("image")}<b>1</b>/${urls.length}</span>
          ${urls.length > 1 ? `<button class="pnav l" type="button" aria-label="Previous photo">${icon("chevL")}</button><button class="pnav r" type="button" aria-label="Next photo">${icon("chevR")}</button>` : ""}</div>
          ${urls.length > 1 ? `<div class="gal-thumbs">${urls.map((u, i) => `<button type="button" class="${i === 0 ? "on" : ""}" data-i="${i}" aria-label="Photo ${i + 1}"><img src="${esc(u)}" alt="" loading="lazy"></button>`).join("")}</div>` : ""}
        </div>` : `<div class="gal-empty">${placeholder({ source: m.source })}</div>`}
      <div class="dr-pad">
      <div class="dr-title">
        <p class="kicker">${esc(KIND[m.record_kind] || "")}${m.partner ? ` · ${esc(m.partner)}` : ""}</p>
        <h2 id="dr-title">${esc(d.title)}</h2>
        ${loc ? `<div class="ploc">${icon("pin")}<span>${esc(loc)}</span></div>` : ""}
      </div>
      ${rich.video && safeUrl(rich.video) ? `<video class="video" src="${esc(rich.video)}" controls preload="none" playsinline></video>` : ""}
      <div class="facts-grid">${facts.map(([ic, k, v]) => `<div>${icon(ic)}<span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("")}</div>
      ${tabs.length ? `<div class="tabs" role="tablist">${tabs.map(([t], i) => `<button role="tab" type="button" class="${i === 0 ? "on" : ""}" data-tab="${i}" aria-selected="${i === 0}">${t}</button>`).join("")}</div>
        ${tabs.map(([, h], i) => `<div class="tab-panel" data-panel="${i}" ${i === 0 ? "" : "hidden"}>${h}</div>`).join("")}` : ""}
      ${(d.quality_flags || []).includes("price_not_published") ? `<div class="notice">${icon("info")}<span>Pricing for this residence is available on request from the developer.</span></div>` : ""}
      </div>`;
    $("#dr-foot").innerHTML = `${safeUrl(d.url) ? `<a class="btn-solid" href="${esc(d.url)}" target="_blank" rel="noopener">View on ${srcName(m.source)}${icon("ext")}</a>` : ""}
      <button class="btn-line" type="button" data-ask-about="${esc(d.title)}">Ask the concierge</button>`;

    // gallery
    let gi = 0;
    const gm = $(".gal-main img", dbody);
    const gshow = (k) => { gi = (k + urls.length) % urls.length; gm.src = urls[gi]; $(".gal-main .pcount b", dbody).textContent = gi + 1;
      dbody.querySelectorAll(".gal-thumbs button").forEach((b, i) => b.classList.toggle("on", i === gi));
      dbody.querySelector(`.gal-thumbs button[data-i="${gi}"]`)?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" }); };
    if (gm) {
      gm.addEventListener("click", () => openLightbox(urls, gi));
      $(".gal-main .l", dbody)?.addEventListener("click", () => gshow(gi - 1));
      $(".gal-main .r", dbody)?.addEventListener("click", () => gshow(gi + 1));
      dbody.querySelectorAll(".gal-thumbs button").forEach((b) => b.addEventListener("click", () => gshow(+b.dataset.i)));
      dbody.querySelectorAll(".gal-thumbs img").forEach((im) => im.addEventListener("error", () => im.parentElement.remove()));
    }
    dbody.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
      dbody.querySelectorAll("[data-tab]").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
      dbody.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== b.dataset.tab));
    }));
    dbody.querySelectorAll("[data-zoom]").forEach((im) => im.addEventListener("click", () => openLightbox([im.dataset.zoom], 0)));
    $("[data-ask-about]", $("#dr-foot")).onclick = (e) => { closeDrawer(); q.value = `Tell me about ${e.currentTarget.dataset.askAbout}`; grow(); q.focus(); };
    dbody.scrollTop = 0;
  }

  // ---------------- lightbox
  const lb = $("#lightbox"); let lbImgs = [], lbI = 0;
  function openLightbox(imgs, i) { lbImgs = imgs; lbI = i; lbShow(); lb.hidden = false; requestAnimationFrame(() => lb.classList.add("on")); $(".x", lb).focus(); }
  function lbClose() { lb.classList.remove("on"); setTimeout(() => (lb.hidden = true), 180); }
  function lbShow() { $("img", lb).src = lbImgs[lbI]; $(".cap", lb).textContent = lbImgs.length > 1 ? `${lbI + 1} / ${lbImgs.length}` : ""; $(".l", lb).hidden = $(".r", lb).hidden = lbImgs.length < 2; }
  $(".x", lb).onclick = lbClose;
  $(".l", lb).onclick = () => { lbI = (lbI - 1 + lbImgs.length) % lbImgs.length; lbShow(); };
  $(".r", lb).onclick = () => { lbI = (lbI + 1) % lbImgs.length; lbShow(); };
  lb.addEventListener("click", (e) => { if (e.target === lb || e.target.tagName === "FIGURE") lbClose(); });
  document.addEventListener("keydown", (e) => {
    if (!lb.hidden) {
      if (e.key === "Escape") lbClose();
      if (e.key === "ArrowLeft") $(".l", lb).click();
      if (e.key === "ArrowRight") $(".r", lb).click();
      return;
    }
    if (!drawer.hidden && e.key === "Escape") closeDrawer();
  });

  // deep link: /?q=...
  const initial = new URLSearchParams(location.search).get("q");
  if (initial) ask(initial);
})();
