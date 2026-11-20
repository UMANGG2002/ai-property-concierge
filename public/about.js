/* Architecture & benchmark page: live health, diagram, corpus coverage, golden-set scores. */
(() => {
  const { API, esc } = window.UI;
  const $ = (s) => document.querySelector(s);
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(v >= 0.995 && v < 1 ? 1 : 0)}%`);
  const get = (p) => fetch(`${API}${p}`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

  window.drawArchitecture($("#arch"));

  // Wilson 95% lower bound for a rate observed on n questions
  const wilsonLo = (p, n) => { if (p == null || !n) return null; const z = 1.96, d = 1 + z * z / n;
    return Math.max(0, (p + z * z / (2 * n) - z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d); };

  // ---------------- chunking: stats + explorer
  const TYPE_NOTE = { faq: "One question + answer", listing: "Whole Wasalt listing (EN + AR)", overview: "Project overview copy", summary: "Short project summary",
    facts: "Key facts line", why_invest: "Investment case", amenities: "Amenities list", location: "Location text + highlights", gallery: "Image captions / alt text",
    key_features: "Key features", property_card: "Unit / residence cards", section: "One of One residence section", related_projects: "Linked projects",
    article_body: "News article body", other: "Generic page section", sub_projects: "Sub-projects", features: "Features" };
  get("/chunks").then((c) => {
    $("#ck-min").textContent = c.min_words; $("#ck-max").textContent = c.max_words; $("#ck-med").textContent = c.median_words;
    const dg = c.types.filter((t) => t.source === "darglobal").reduce((a, t) => a + t.count, 0);
    const ws = c.types.filter((t) => t.source === "wasalt").reduce((a, t) => a + t.count, 0);
    const lst = c.types.find((t) => t.type === "listing"); if (lst) $("#ws-words").textContent = lst.median_words;
    $("#chunk-kpis").innerHTML = [
      kpi("Chunks indexed", c.total, `median ${c.median_words} words · ${c.min_words}–${c.max_words}`),
      kpi("DarGlobal children", dg, "Section-level chunks under 74 parents"),
      kpi("Wasalt chunks", ws, "One per listing (parent = child)"),
      kpi("Chunk types", c.types.length, "Mirrors the pages' own sections"),
    ].join("");
    $("#chunk-table").innerHTML = `<thead><tr><th>Chunk type</th><th>Source</th><th class="num">Chunks</th><th class="num">Median words</th><th class="num">Max</th></tr></thead><tbody>` +
      c.types.map((t) => `<tr><td title="${esc(TYPE_NOTE[t.type] || "")}">${esc(t.type.replace(/_/g, " "))}<div class="muted small" style="text-transform:none;font-weight:400">${esc(TYPE_NOTE[t.type] || "")}</div></td><td>${esc(t.source)}</td><td class="num">${t.count}</td><td class="num">${t.median_words}</td><td class="num">${t.max_words}</td></tr>`).join("") + "</tbody>";
  }).catch(() => { $("#chunk-table").innerHTML = "<tbody><tr><td>Chunk statistics unavailable.</td></tr></tbody>"; });

  const loadRec = (id) => {
    $("#ex-list").innerHTML = `<div class="sk" style="height:160px"></div>`;
    get(`/chunks?record=${encodeURIComponent(id)}`).then((r) => {
      $("#ex-meta").innerHTML = `<b>${esc(r.title || id)}</b> · parent ${r.parent_words} words → ${r.chunks.length} chunk${r.chunks.length === 1 ? "" : "s"}`;
      $("#ex-list").innerHTML = r.chunks.map((c, i) => {
        const t = c.text.replace(/^((?:[^.>\n]+>\s*)+[^.\n]*\.)/, "<mark>$1</mark>");
        const safe = esc(c.text).replace(/^((?:[^.&\n]+&gt;\s*)+[^.\n]*\.)/, "<mark>$1</mark>");
        return `<details ${i < 2 ? "open" : ""}><summary><span class="tag">${esc(c.type || "chunk")}</span><code>${esc(c.id)}</code><em>${c.words} words</em></summary><p>${safe}</p></details>`;
      }).join("") || `<p class="muted">No chunks for this record.</p>`;
    }).catch(() => { $("#ex-list").innerHTML = `<p class="muted">Explorer unavailable.</p>`; });
  };
  document.querySelectorAll("#ex-pick button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#ex-pick button").forEach((x) => x.classList.toggle("on", x === b)); loadRec(b.dataset.rec);
  }));
  loadRec("darglobal:marea");

  // parent sizes for the narrative
  get("/stats").then(() => { }).catch(() => { });

  const kpi = (k, v, s = "", gate = null, small = false) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v${small ? " sm" : ""}">${esc(v)}</div>
    ${gate ? `<div class="g ${gate.ok ? "pass" : "fail"}">${gate.ok ? "Passes" : "Below"} gate ${esc(gate.th)}</div>` : ""}${s ? `<div class="s">${esc(s)}</div>` : ""}</div>`;

  function meters(el, rows, { max = 1, fmt = (v) => v, gate = null } = {}) {
    el.innerHTML = rows.map(([name, v, title]) => `<div class="meter" title="${esc(title || `${name}: ${fmt(v)}`)}"><span>${esc(name)}</span>
      <span class="track"><span class="fill" style="width:0" data-w="${Math.max(0, Math.min(100, (v / max) * 100))}"></span>${gate != null ? `<span class="gl" style="left:calc(${gate * 100}% - 1px)"></span>` : ""}</span>
      <span class="val">${esc(fmt(v))}</span></div>`).join("");
    requestAnimationFrame(() => el.querySelectorAll(".fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  if ($("#live-kpis")) window.UI.health().then((h) => {
    const st = h.store || {};
    const dense = (st.dense || []).find((d) => d.ok !== false);
    const search = dense ? (dense.sparse_model ? "Upstash hybrid" : "Upstash vector") : "Local BM25";
    $("#live-kpis").innerHTML = [
      kpi("Status", h.status === "ok" ? "Operational" : "Degraded", h.version ? `API v${h.version}` : "", null, true),
      kpi("Retrieval", search, dense ? [dense.embedding_model, dense.sparse_model].filter(Boolean).join(" + ") : "Upstash not configured or unreachable", null, true),
      kpi("Answers", h.llm?.configured ? "OpenRouter LLM" : "Grounded template", h.llm?.configured ? "Free models, auto-rotated" : "Add OPENROUTER_API_KEY to enable", null, true),
      kpi("Records served", h.corpus?.records ?? "—", h.corpus?.indexed_chunks ? `${h.corpus.indexed_chunks} chunks in fallback index` : ""),
    ].join("");
  }).catch((e) => { $("#live-kpis").innerHTML = kpi("Status", "API unreachable", e.message, null, true); });

  get("/stats").then((s) => {
    $("#coverage-kpis").innerHTML = [
      kpi("Records", s.records, `${s.by_source.darglobal} DarGlobal · ${s.by_source.wasalt} Wasalt`),
      kpi("Properties", s.property_records, "Developments, One of One, listings"),
      kpi("Chunks indexed", s.chunks, `Median ${s.chunk_words?.median} words`),
      kpi("Duplicates collapsed", Object.keys(s.duplicates_collapsed || {}).length, "Identical pages kept as aliases"),
    ].join("");
    const byCountry = Object.entries(s.countries || {}).sort((a, b) => b[1] - a[1]);
    const byType = Object.entries(s.property_types || {}).sort((a, b) => b[1] - a[1]);
    meters($("#m-country"), byCountry, { max: Math.max(...byCountry.map((r) => r[1]), 1) });
    meters($("#m-type"), byType.map(([k, v]) => [k.replace(/_/g, " "), v]), { max: Math.max(...byType.map((r) => r[1]), 1) });
  }).catch(() => { $("#coverage-kpis").innerHTML = kpi("Coverage", "Unavailable", "Corpus report not built", null, true); });

  get("/eval?full=1").then((r) => {
    const o = r.summary.overall, g = r.gates || {};
    $("#bench-meta").innerHTML = `${o.n} golden questions · run ${esc(String(r.run_at).replace("T", " ").slice(0, 16))} · retrieval ${esc(r.store)} · answers ${esc(r.answer_source)}.`;
    const gate = (m) => (g[m] != null && o[m] != null ? { ok: o[m] >= g[m], th: pct(g[m]) } : null);
    const rate = (k, label, what) => { const n = o[`${k}_n`]; const lo = wilsonLo(o[k], n);
      return kpi(label, pct(o[k]), `${what} · n = ${n ?? "—"}${lo != null ? ` · 95% ≥ ${pct(lo)}` : ""}`, gate(k)); };
    $("#n-beh").textContent = o.behaviour_pass_n ?? "—"; $("#n-parse").textContent = o.parse_accuracy_n ?? "—";
    const local = String(r.store).includes("bm25") && !String(r.store).includes("upstash");
    $("#run-cond").innerHTML = `This report was produced with retrieval <b>${esc(r.store)}</b> and <b>${esc(r.answer_source)}</b> answers${local ? ", that is, the bundled BM25 fallback rather than the Upstash hybrid index, and without the LLM. Re-run <code>evaluate.py</code> with Upstash configured to measure the production path" : ""}.`;
    $("#bench-kpis").innerHTML = [
      rate("hit@5", "Hit@5", "Correct property in top 5"),
      rate("hit@1", "Hit@1", "Correct property ranked first"),
      kpi("MRR", o.mrr?.toFixed(3) ?? "—", `Rank of first correct result · n = ${o.mrr_n ?? "—"}`, gate("mrr")),
      rate("precision@k", "Precision@k", "List results meet constraints"),
      rate("parse_accuracy", "Parse accuracy", "Question → filters"),
      rate("behaviour_pass", "Behaviour", "Scope, relaxation, rentals"),
      rate("answer_contains_ok", "Answer checks", "Required fact present"),
      kpi("Latency p50 / p95", `${o.latency_ms_p50 ?? "—"} / ${o.latency_ms_p95 ?? "—"} ms`, local ? "Local index, no network or LLM: not production latency" : "Whole pipeline", null, true),
      ...(o.judge_faithfulness != null ? [kpi("Judge faithfulness", `${o.judge_faithfulness.toFixed(2)} / 5`, "LLM-as-judge")] : []),
    ].join("");
    const cats = Object.entries(r.summary.by_category);
    meters($("#m-cat"), cats.filter(([, a]) => a["hit@5"] != null).map(([c, a]) => [c.replace(/_/g, " "), a["hit@5"], `${c}: Hit@5 ${pct(a["hit@5"])} over ${a["hit@5_n"]} questions`]),
      { max: 1, fmt: pct, gate: g["hit@5"] });
    const cols = [["n", "n"], ["hit@1", "Hit@1"], ["hit@5", "Hit@5"], ["mrr", "MRR"], ["recall@k", "Recall@k"], ["precision@k", "Precision@k"],
      ["ndcg@k", "nDCG@k"], ["parse_accuracy", "Parse"], ["behaviour_pass", "Behaviour"], ["answer_contains_ok", "Answer"], ["latency_ms_p50", "p50 ms"]];
    const cls = (k, v) => (v == null || k === "n" || k.startsWith("latency") ? "" : v >= 0.95 ? "good" : v < 0.75 ? "low" : "");
    $("#cat-table").innerHTML = `<thead><tr><th>Category</th>${cols.map(([, l]) => `<th class="num">${l}</th>`).join("")}</tr></thead><tbody>` +
      cats.map(([c, a]) => `<tr><td>${esc(c.replace(/_/g, " "))}</td>${cols.map(([k]) => `<td class="num ${cls(k, a[k])}">${a[k] == null ? "—" : k === "n" || k.startsWith("latency") ? a[k] : a[k].toFixed(2)}</td>`).join("")}</tr>`).join("") + "</tbody>";
    const bad = (r.results || []).filter((x) => x["hit@5"] === 0 || x.behaviour_pass === 0 || x.parse_ok === false || x.answer_contains_ok === 0 || (x["precision@k"] != null && x["precision@k"] < 0.75));
    $("#fails").innerHTML = bad.length ? `<ul class="fails">${bad.slice(0, 30).map((x) => `<li><code>${esc(x.id)}</code> ${esc(x.query)}</li>`).join("")}</ul>` : `<p class="muted" style="margin:0">None. Every golden question passes its checks.</p>`;
  }).catch(() => { $("#bench-meta").innerHTML = "No evaluation report yet. Run <code>uv run python evaluate.py</code>."; });

  if (location.hash) setTimeout(() => document.querySelector(location.hash)?.scrollIntoView(), 400);
})();
