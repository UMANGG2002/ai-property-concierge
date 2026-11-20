/* Component architecture diagram, drawn as SVG so it stays sharp, themeable and accessible. window.drawArchitecture(el) */
(() => {
  const W = 1260, H = 790;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function box(x, y, w, h, { n, title, sub, lines = [], gold = false, strong = false }) {
    let t = `<g class="d-node${gold ? " gold" : ""}${strong ? " strong" : ""}"><rect x="${x}" y="${y}" width="${w}" height="${h}"/>`;
    let ty = y + 26;
    if (n) { t += `<text class="d-num" x="${x + 16}" y="${ty}">${esc(n)}</text>`; }
    t += `<text class="d-title" x="${x + (n ? 44 : 16)}" y="${ty}">${esc(title)}</text>`;
    if (sub) { ty += 19; t += `<text class="d-sub" x="${x + (n ? 44 : 16)}" y="${ty}">${esc(sub)}</text>`; }
    if (lines.length) {
      ty += 10;
      t += `<line class="d-rule" x1="${x + 16}" x2="${x + w - 16}" y1="${ty}" y2="${ty}"/>`;
      lines.forEach((l) => { ty += 19; t += `<text class="d-li" x="${x + 16}" y="${ty}">${esc(l)}</text>`; });
    }
    return t + "</g>";
  }
  const path = (d, { dash = false, label = "", lx = 0, ly = 0, anchor = "middle" } = {}) =>
    `<path class="d-edge${dash ? " dash" : ""}" d="${d}" marker-end="url(#${dash ? "ah-gold" : "ah"})"/>` +
    (label ? `<text class="d-elabel${dash ? " gold" : ""}" x="${lx}" y="${ly}" text-anchor="${anchor}">${esc(label)}</text>` : "");
  const lane = (x, y, label) => `<text class="d-lane" x="${x}" y="${y}">${esc(label)}</text>`;

  function svg() {
    const P = [];
    // lanes
    P.push(lane(20, 34, "CLIENT"), lane(205, 34, "EDGE · API"), lane(430, 34, "LANGGRAPH PIPELINE · rag/graph.py"), lane(960, 34, "MANAGED SERVICES"));
    P.push(`<rect class="d-lanebg" x="420" y="50" width="360" height="420"/>`);
    P.push(`<line class="d-sep" x1="20" x2="${W - 20}" y1="510" y2="510"/>`, lane(20, 548, "OFFLINE · BUILD & INGEST"));

    // client + api
    P.push(box(20, 190, 150, 132, { title: "Browser", sub: "public/ · static UI", lines: ["Chat & search", "Residence drawer", "Galleries"] }));
    P.push(box(205, 160, 180, 192, { title: "FastAPI", sub: "POST /api/chat", lines: ["Input validation", "Rate limit per IP", "1-hour answer cache", "Never raises: degrades"], strong: true }));

    // pipeline nodes
    const nodes = [
      ["01", "Parse", "Filters · projects · topic · Arabic"],
      ["02", "Retrieve", "Hybrid search · relax · group · rank"],
      ["03", "Build context", "Top 5 properties as numbered sources"],
      ["04", "Generate", "Cited answer from a free LLM"],
      ["05", "Verify", "Numbers & citations must be grounded"],
    ];
    const NX = 440, NW = 320, NH = 58, NY = (i) => 68 + i * 78;
    nodes.forEach(([n, t, s], i) => P.push(box(NX, NY(i), NW, NH, { n, title: t, sub: s })));
    for (let i = 0; i < 4; i++) P.push(path(`M${NX + 30},${NY(i) + NH} V${NY(i + 1) - 2}`));

    // services: Upstash, Resilience, OpenRouter (fallbacks sit between the two dependencies they replace)
    const SX = 960, SW = 260, MX = SX + SW / 2;
    P.push(box(SX, 60, SW, 172, { title: "Upstash Vector", sub: "Hybrid index · 905 chunks", lines: ["Dense: text-embedding-3-small", "Sparse: BM25", "Fusion + metadata filter in DB", "Embeds text server-side"], strong: true }));
    P.push(box(SX, 272, SW, 98, { title: "Resilience layer", sub: "Same filters, no network", lines: ["Local BM25 index", "Grounded template answer"], gold: true }));
    P.push(box(SX, 410, SW, 82, { title: "OpenRouter", sub: "Free LLMs, found at runtime", lines: ["Rotation · cooldown · 40 s budget"] }));
    P.push(path(`M${MX},232 V270`, { dash: true, label: "if unreachable", lx: MX + 10, ly: 255, anchor: "start" }));
    P.push(path(`M${MX},410 V372`, { dash: true, label: "if down / ungrounded", lx: MX + 10, ly: 395, anchor: "start" }));

    // request path
    P.push(path("M170,238 H203", { label: "question", lx: 187, ly: 230 }));
    P.push(path("M205,288 H172", { label: "answer + cards", lx: 187, ly: 306 }));
    P.push(path(`M385,200 H410 V${NY(0) + 29} H${NX - 2}`));
    P.push(path(`M${NX},${NY(4) + 29} H410 V320 H387`));
    const ry = NY(1) + 29, gy = NY(3) + 29;
    P.push(path(`M${NX + NW},${ry} H860 V146 H${SX - 2}`, { label: "query text + filter", lx: 868, ly: 138, anchor: "start" }));
    P.push(path(`M${NX + NW},${gy} H860 V451 H${SX - 2}`, { label: "prompt + context", lx: 868, ly: 443, anchor: "start" }));

    // offline row
    const OY = 572, OH = 150, OW = 196, gap = 63, ox = (i) => 20 + i * (OW + gap);
    const off = [
      ["Sources", "Raw data", ["DarGlobal · 74 records", "Wasalt · 80-row CSV", "CMS JSON + gallery URLs"]],
      ["Unified schema", "rag/schema.py · sources/", ["One record shape", "PII removed", "Duplicates collapsed"]],
      ["Chunking", "rag/corpus.py", ["905 chunks · 154 parents", "Sections · FAQs · listings", "Validated counts"]],
      ["Corpus", "rag/index/", ["chunks.jsonl", "parents.json → API cards", "golden.jsonl (165 Qs)"]],
      ["rag/build.py", "Ingest", ["Detects index type", "Uploads chunk text", "Idempotent by hash"]],
    ];
    off.forEach(([t, s, l], i) => P.push(box(ox(i), OY, OW, OH, { title: t, sub: s, lines: l, strong: i === 4 })));
    for (let i = 0; i < 4; i++) P.push(path(`M${ox(i) + OW},${OY + 50} H${ox(i + 1) - 2}`));
    // ingest → upstash (right margin, no crossings)
    const bx = ox(4) + OW;
    P.push(path(`M${bx},${OY + 50} H${W - 16} V146 H${SX + SW + 2}`, { label: "upsert", lx: W - 22, ly: 540, anchor: "end" }));
    // eval reads API
    P.push(`<text class="d-foot" x="20" y="${H - 22}">Evaluation: eval.run_eval replays the 165 golden questions through the same pipeline → eval/eval_report.json → /api/eval.</text>`);

    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Component architecture of one chat request" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>
        <marker id="ah-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah gold"/></marker>
        <style>
          .d-node rect { fill: var(--surface, #fffdf9); stroke: var(--line-2, #d5ccbd); stroke-width: 1; }
          .d-node.strong rect { stroke: var(--ink, #121110); }
          .d-node.gold rect { fill: color-mix(in srgb, var(--gold, #a07f52) 12%, var(--surface, #fffdf9)); stroke: var(--gold, #a07f52); }
          .d-lanebg { fill: var(--surface-2, #efe9df); fill-opacity: .55; stroke: var(--line, #e3dcd0); stroke-dasharray: 3 4; }
          text { font-family: Inter, -apple-system, Segoe UI, sans-serif; fill: var(--ink-2, #3b3833); }
          .d-title { font-size: 15px; font-weight: 600; fill: var(--ink, #121110); }
          .d-num { font: 500 19px "Cormorant Garamond", Georgia, serif; fill: var(--gold, #a07f52); }
          .d-sub { font-size: 11.5px; fill: var(--muted, #8a847a); }
          .d-li { font-size: 12px; }
          .d-rule { stroke: var(--line, #e3dcd0); }
          .d-lane { font-size: 10px; font-weight: 600; letter-spacing: .2em; fill: var(--gold, #a07f52); }
          .d-sep { stroke: var(--line-2, #d5ccbd); stroke-dasharray: 2 5; }
          .d-edge { fill: none; stroke: var(--ink-2, #3b3833); stroke-width: 1.4; }
          .d-edge.dash { stroke: var(--gold, #a07f52); stroke-dasharray: 5 4; }
          .ah { fill: var(--ink-2, #3b3833); } .ah.gold { fill: var(--gold, #a07f52); }
          .d-elabel { font-size: 11px; fill: var(--muted, #8a847a); paint-order: stroke; stroke: var(--surface, #fffdf9); stroke-width: 5px; stroke-linejoin: round; }
          .d-elabel.gold { fill: var(--gold, #a07f52); }
          .d-foot { font-size: 12px; fill: var(--muted, #8a847a); }
        </style>
      </defs>${P.join("")}</svg>`;
  }
  window.drawArchitecture = (el) => { el.innerHTML = svg(); };
  window.architectureSVG = svg;
})();
