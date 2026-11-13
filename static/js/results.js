// Results view: renders /api/report (model comparison, confusion matrix, per-sign table).

import { esc } from "./translate.js";

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

export function renderResults(root, report, { onGoTrain }) {
  if (!report) {
    root.innerHTML = `
      <div class="empty-state">
        <h2>No model yet</h2>
        <p>Record at least two signs in the Train tab, then press Train model. The accuracy, the comparison of four models and the confusion matrix will appear here.</p>
        <button class="primary" id="go-train">Go to Train</button>
      </div>`;
    root.querySelector("#go-train").addEventListener("click", onGoTrain);
    return;
  }

  const r = report;
  const grouped = r.evaluation.method === "grouped";
  const models = [...r.models].filter((m) => !m.error).sort((a, b) => b.macro_f1 - a.macro_f1);
  const failed = r.models.filter((m) => m.error);

  const mixups = [];
  r.confusion.forEach((row, i) => row.forEach((n, j) => { if (i !== j && n > 0) mixups.push({ made: r.labels[i], read: r.labels[j], n }); }));
  mixups.sort((a, b) => b.n - a.n);

  const baseline = r.baseline_rules
    ? `<div class="callout"><strong>Starter rules on the same signs: ${pct(r.baseline_rules.accuracy)}</strong> across ${r.baseline_rules.samples.toLocaleString()} samples of ${r.baseline_rules.signs.map(esc).join(", ")}. The hand-written rules are the baseline the trained model is compared against.</div>`
    : "";

  root.innerHTML = `
    <div class="results-head">
      <div>
        <h2>Model results</h2>
        <p>Trained ${esc(r.trained_at)}. ${esc(r.best_model)} scored best and is used for live recognition.</p>
      </div>
      <button class="ghost no-print" id="print-report"><svg><use href="#i-print"/></svg>Print or save as PDF</button>
    </div>

    <div class="tiles">
      <div class="tile hero"><div class="k">Accuracy</div><div class="v">${pct(r.accuracy)}</div><div class="s">${grouped ? "on recordings the model never saw" : "on held-out frames"}</div></div>
      <div class="tile"><div class="k">Macro F1</div><div class="v">${pct(r.macro_f1)}</div><div class="s">all signs weighted equally</div></div>
      <div class="tile"><div class="k">Signs</div><div class="v">${r.dataset.signs}</div><div class="s">${r.dataset.two_hand_samples ? `${r.dataset.two_hand_samples} two-hand samples` : "one hand"}</div></div>
      <div class="tile"><div class="k">Samples</div><div class="v">${r.dataset.samples.toLocaleString()}</div><div class="s">from ${r.dataset.recordings} recordings</div></div>
    </div>

    <div class="callout"><strong>How this was measured.</strong> ${esc(r.evaluation.text)}</div>
    ${r.warnings.map((w) => `<div class="callout warn"><strong>Heads up.</strong> ${esc(w)}</div>`).join("")}
    ${baseline}

    <section class="card">
      <h3>Four models compared</h3>
      <p>Bar length is macro F1: the F1 score averaged over every sign, so a rarely recorded sign counts as much as a common one. Hover a bar for timing.</p>
      <div class="bars">
        ${models.map((m) => `
          <div class="bar-row ${m.name === r.best_model ? "best" : ""}" data-tip="${esc(tipModel(m))}">
            <div class="nm">${esc(m.name)}<small>accuracy ${pct(m.accuracy)}, ${m.latency_ms.toFixed(1)} ms per prediction</small></div>
            <div class="track"><i style="width:${(m.macro_f1 * 100).toFixed(2)}%"></i></div>
            <div class="val">${pct(m.macro_f1)}</div>
          </div>`).join("")}
      </div>
      ${failed.map((m) => `<p class="help">${esc(m.name)} couldn't be evaluated: ${esc(m.error)}</p>`).join("")}
    </section>

    <section class="card">
      <h3>Confusion matrix</h3>
      <p>Each row is the sign you made, each column is what ${esc(r.best_model)} read it as. Brighter means a larger share of that row. A clean diagonal means few mix-ups.</p>
      <div class="matrix-wrap">${matrixHtml(r)}</div>
      <div class="scale"><span>0%</span><i></i><span>100% of the row</span></div>
      <p class="help" style="margin-top:12px">${mixups.length
        ? "Most common mix-ups: " + mixups.slice(0, 5).map((m) => `${esc(m.made)} read as ${esc(m.read)} (${m.n})`).join(", ") + "."
        : "No mix-ups: every held-out sample was read correctly."}</p>
    </section>

    <section class="card">
      <h3>Per sign</h3>
      <p>Precision: when the model says this sign, how often it is right. Recall: when you make this sign, how often the model catches it. Scores under 80% are marked.</p>
      <table class="per">
        <thead><tr><th>Sign</th><th>Precision</th><th>Recall</th><th>F1</th><th>Samples</th><th>Recordings</th></tr></thead>
        <tbody>${r.per_class.map((c) => `
          <tr><td><b>${esc(c.label)}</b></td>
          <td class="${c.precision < 0.8 ? "low" : ""}">${pct(c.precision)}</td>
          <td class="${c.recall < 0.8 ? "low" : ""}">${pct(c.recall)}</td>
          <td class="${c.f1 < 0.8 ? "low" : ""}">${pct(c.f1)}</td>
          <td>${c.support}</td><td>${c.sessions}</td></tr>`).join("")}
        </tbody>
      </table>
    </section>
  `;

  root.querySelector("#print-report").addEventListener("click", () => window.print());
  wireTooltips(root);
}

function tipModel(m) {
  return `${m.name}|Macro F1 ${pct(m.macro_f1)}|Accuracy ${pct(m.accuracy)}|${m.latency_ms.toFixed(2)} ms per prediction|Trained in ${m.fit_seconds.toFixed(2)} s`;
}

function matrixHtml(r) {
  const n = r.labels.length;
  const cell = Math.max(18, Math.min(44, Math.floor(560 / n)));
  const showNums = cell >= 28;
  const labW = Math.min(120, 14 + Math.max(...r.labels.map((l) => l.length)) * 8);
  // short labels sit upright above their column; longer ones are turned sideways
  const tall = Math.max(...r.labels.map((l) => l.length)) * 8 > cell;
  let html = `<div class="matrix${tall ? " tall-cols" : ""}" style="--cell:${cell}px;grid-template-columns:${labW}px repeat(${n}, ${cell}px)">`;
  html += `<div></div>` + r.labels.map((l) => `<div class="lab col" title="${esc(l)}">${esc(l)}</div>`).join("");
  r.confusion.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="lab">${esc(r.labels[i])}</div>`;
    row.forEach((count, j) => {
      const v = count / total;
      const bg = count === 0 ? "var(--cell-zero)" : `color-mix(in oklab, #FFB547 ${(12 + v * 88).toFixed(1)}%, var(--cell-zero))`;
      const tip = `Made ${r.labels[i]}, read as ${r.labels[j]}|${count} of ${total} (${pct(v)})`;
      const dark = v > 0.55 ? " dark-text" : "";
      html += `<div class="cell${dark}" style="background:${bg}" data-tip="${esc(tip)}">${showNums && count ? count : ""}</div>`;
    });
  });
  html += `</div>`;
  html += `<div class="axis-note"><span>Rows: sign made</span><span>Columns: sign read</span></div>`;
  return html;
}

function wireTooltips(root) {
  const tip = document.getElementById("tooltip");
  const show = (el, x, y) => {
    const parts = el.dataset.tip.split("|");
    tip.innerHTML = `<b>${esc(parts[0])}</b>` + parts.slice(1).map((p) => `<br>${esc(p)}`).join("");
    tip.classList.add("on");
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  };
  root.addEventListener("pointermove", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) show(el, e.clientX, e.clientY); else tip.classList.remove("on");
  });
  root.addEventListener("pointerleave", () => tip.classList.remove("on"));
  root.addEventListener("scroll", () => tip.classList.remove("on"), true);
}
