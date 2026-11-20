// Page logic for index.html. Kept in its own file so the Content-Security-Policy can forbid inline scripts.
const $ = (id) => document.getElementById(id);
let threshold = null; // churn cut-off, read from /health
let evaluation = null; // measured results from /metrics/evaluation, fetched once
let view = "dashboard";
const PAGES = {
  "/": ["dashboard", "Customer Churn Predictor", "TensorFlow model behind a FastAPI inference pipeline. Trained on synthetic data."],
  "/predict": ["predict", "Predict churn", "Score one customer with the deployed model."],
  "/recent": ["recent", "Recent predictions", "Every prediction is stored in SQLite: totals from SQL queries, then the last 10, newest first."],
  "/drift": ["drift", "Input drift", "How far incoming inputs have moved from the training data."],
  "/performance": ["performance", "Performance", "Latency and throughput measured with benchmarks/latency_check.py."],
  "/quality": ["quality", "Model quality", "Test scores, the churn cut-off, and a check on independent IBM Telco data."],
};
const FIELDS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "contract_type"];
const PRESETS = {
  risky: { tenure_months: 5, monthly_charges: 95.5, total_charges: 477.5, support_tickets: 4, contract_type: "month-to-month" },
  loyal: { tenure_months: 60, monthly_charges: 45, total_charges: 2700, support_tickets: 0, contract_type: "two-year" },
};
const CONTRACTS = { "month-to-month": "Month-to-month", "one-year": "One year", "two-year": "Two year" };
const LIMIT_MB = 512; // Render free plan
const GAUGE_LEN = 2 * Math.PI * 70;
const STATUS_ORDER = ["stable", "moderate", "significant"];

// API key, only needed when the server sets API_KEY. Kept for this tab only; storage can be blocked.
let apiKey = "";
function storedKey() {
  try { return apiKey || sessionStorage.getItem("apiKey") || ""; } catch { return apiKey; }
}

async function api(path, options = {}) {
  const key = storedKey();
  const res = await fetch(path, { ...options, headers: { ...(options.headers || {}), ...(key ? { "X-API-Key": key } : {}) } });
  const body = await res.json().catch(() => null);
  if (res.status === 401) $("keybox").hidden = false; // server wants a key: ask for it
  return { ok: res.ok, status: res.status, body, headers: res.headers };
}

// Builds elements with textContent, so API data is never parsed as HTML.
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function bar(cls, pct, color) {
  const node = el("div", undefined, cls);
  node.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + "%";
  if (color) node.style.background = color;
  return node;
}
function row(cells) {
  const tr = el("tr");
  for (const c of cells) { const td = el("td"); td.append(c instanceof Node ? c : String(c)); tr.append(td); }
  return tr;
}
function emptyRow(text, cols) {
  const tr = el("tr"); const td = el("td", text, "fine"); td.colSpan = cols; tr.append(td); return tr;
}
const money = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct1 = (v) => (v * 100).toFixed(1) + "%";
const ms = (v) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 }) + " ms";
const day = (iso) => (iso || "").slice(0, 10);
const cap = (s) => s[0].toUpperCase() + s.slice(1);
// 2 decimals near the cut-off, else 0.7095 would read "71.0%" next to a 71% cut-off.
const probText = (r) => (r.churn_probability * 100).toFixed(r.confidence_score < 0.5 ? 2 : 1) + "%";
const cutText = () => (threshold === null ? "" : `${(threshold * 100).toFixed(0)}% `);
const worst = (features) => Object.values(features).reduce((w, f) => (STATUS_ORDER.indexOf(f.status) > STATUS_ORDER.indexOf(w) ? f.status : w), "stable");
function uptime(s) {
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

/* ---------- routing ---------- */
let routeToken = 0;
async function route(focus) {
  const path = PAGES[location.pathname] ? location.pathname : "/";
  const [name, title, lede] = PAGES[path];
  view = name;
  const token = ++routeToken;
  // Show a section only once its data is in. Filling cards after they were visible made the page
  // jump (Lighthouse CLS 0.166 on live /quality); a section that appears complete moves nothing.
  document.querySelectorAll("[data-view]").forEach((s) => (s.hidden = true));
  $("view-loading").hidden = false;
  $("footer").hidden = true;
  document.querySelectorAll(".nav a").forEach((a) => (a.getAttribute("href") === path ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  setMenu(false); // phones: picking a page closes the menu
  $("page-title").textContent = title;
  $("page-lede").textContent = lede;
  document.title = path === "/" ? "Churn Predictor" : `${title} · Churn Predictor`;
  if (focus) $("page-title").focus({ preventScroll: true }); // tell screen readers the page changed
  await loadView();
  if (token !== routeToken) return; // the user already moved to another page
  $("view-loading").hidden = true;
  document.querySelector(`[data-view="${name}"]`).hidden = false;
  $("footer").hidden = false;
}

// Same-origin page links switch views without a reload; new-tab clicks and /docs behave normally.
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-route]");
  if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  if (location.pathname !== a.getAttribute("href")) history.pushState(null, "", a.getAttribute("href"));
  window.scrollTo(0, 0);
  route(true);
});
window.addEventListener("popstate", () => route(true));

async function getEvaluation() {
  if (!evaluation) {
    try { const { ok, body } = await api("/metrics/evaluation"); if (ok) evaluation = body; } catch { /* callers show "not available" */ }
  }
  return evaluation;
}

// Resolves when the current page's data is rendered. Loaders catch their own errors, so this never rejects.
function loadView() {
  const health = loadHealth(); // sidebar and header; only the dashboard shows health data inside its section
  const page = { dashboard: () => Promise.all([health, loadDashboard()]), recent: loadRecent, drift: loadDrift,
                 performance: loadPerformance, quality: loadQuality }[view];
  return page ? page() : Promise.resolve();
}

/* ---------- health / sidebar ---------- */
async function loadHealth() {
  let ok = false, body = null;
  try { ({ ok, body } = await api("/health")); } catch { /* shown as unreachable below */ }
  const up = ok && body && body.model_loaded;
  $("side-status").classList.toggle("bad", !up);
  $("pill-text").classList.toggle("bad", !up);
  if (!body) {
    $("side-status").textContent = $("pill-text").textContent = "API unreachable";
    return null;
  }
  threshold = body.decision_threshold;
  $("side-status").textContent = up ? `API ${body.status} · model loaded` : `API ${body.status}`;
  $("pill-text").textContent = up ? `API ${body.status} · threshold ${threshold}` : `API ${body.status}`;
  document.querySelectorAll("[data-version]").forEach((n) => (n.textContent = body.model_version || "–"));
  $("cutoff").textContent = threshold ?? "–";
  $("tile-uptime").textContent = uptime(body.uptime_seconds);
  placeCutMarker();
  showMemory(body.memory);
  return body;
}

function showMemory(mem) {
  // Main number: the app's own memory (anonymous RSS), which is what pushes a container toward an OOM kill.
  // The cgroup total also counts file cache the kernel frees under pressure: a fresh Docker run sat at
  // 510 of 512 MB with 0 OOM kills, so as the headline it would look full when it is not.
  // Windows has no RssAnon: fall back to the process working set.
  const used = mem ? mem.rss_anon_mb ?? mem.process_mb : null;
  $("mem-label").textContent = mem && mem.rss_anon_mb == null ? "Process memory" : "App memory";
  if (used == null) {
    $("mem-text").textContent = "unavailable";
    $("mem-note").textContent = "This platform does not report memory";
    return;
  }
  const pct = Math.min(100, (used / LIMIT_MB) * 100);
  $("mem-text").textContent = `${used} MB`; // the limit is in the note and the meter
  $("mem-meter").firstElementChild.style.width = pct.toFixed(1) + "%";
  $("mem-meter").setAttribute("aria-valuenow", Math.round(pct));
  $("mem-note").textContent = mem.cgroup_mb != null
    ? `Container total ${mem.cgroup_mb} MB incl. reclaimable file cache · limit ${LIMIT_MB} MB`
    : `Peak ${mem.process_peak_mb} MB · local run, not in a container; Render limit ${LIMIT_MB} MB`;
}

/* ---------- dashboard ---------- */
async function loadDashboard() {
  const [ev, recent, drift] = await Promise.all([
    getEvaluation(),
    api("/metrics/summary").then((r) => (r.ok ? r.body : null)).catch(() => null),
    api("/metrics/drift").then((r) => (r.ok ? r.body : null)).catch(() => null),
  ]);
  const t = ev && ev.training, cal = ev && ev.threshold_calibration;
  // 4 decimals, as stored: toFixed(3) would turn 0.7825 into 0.782 (binary floating point).
  $("auc").textContent = t ? t.test_roc_auc.toFixed(4) : "n/a";
  $("auc-note").textContent = t ? `Accuracy ${t.test_accuracy.toFixed(4)} · trained on ${t.n_train_rows.toLocaleString("en-US")} rows` : "Evaluation results not available";
  $("cutoff-note").textContent = cal ? `Default was 0.5 · tuned on IBM Telco data (${cal.holdout_rows.toLocaleString("en-US")} holdout rows)` : "Default 0.5 unless calibrated";

  $("tile-recent").textContent = recent ? recent.total.toLocaleString("en-US") : "n/a";
  $("tile-recent-note").textContent = recent
    ? (recent.total ? `In the SQLite log, ${pct1(recent.churn_rate)} predicted to churn.` : "None stored yet on this server.")
    : "Could not load.";
  if (drift) {
    const enough = drift.status !== "insufficient_data";
    $("tile-drift").textContent = enough ? cap(drift.status) : "Not enough data";
    $("tile-drift-note").textContent = enough ? `Based on the last ${drift.n_samples} requests.` : `${drift.n_samples} of ${drift.min_samples} requests needed.`;
  } else {
    $("tile-drift").textContent = "n/a"; $("tile-drift-note").textContent = "Could not load.";
  }
  const opt = ev && ev.benchmark_optimized && ev.benchmark_optimized.results.find((r) => r.concurrency === 1);
  $("tile-latency").textContent = opt ? ms(opt.predict.p50_ms) : "n/a";
  $("tile-latency-note").textContent = opt ? `One request at a time, optimized build, measured locally ${day(ev.benchmark_optimized.timestamp_utc)}.` : "Benchmark results not available.";
}

/* ---------- predict ---------- */
function readForm() {
  const data = {};
  for (const name of FIELDS) {
    const raw = $(name).value.trim();
    // Numbers go as numbers. Empty or non-numeric values go through unchanged so the API reports them.
    if (raw === "") data[name] = null;
    else if (name === "contract_type" || isNaN(Number(raw))) data[name] = raw;
    else data[name] = Number(raw);
  }
  return data;
}

function showState(which) {
  $("result").hidden = which !== "result";
  $("result-error").hidden = which !== "error";
  $("result-empty").hidden = which !== "empty";
  $("r-risk").hidden = which !== "result";
}

function showError(title, details) {
  const body = $("result-error-body");
  body.replaceChildren(el("div", title, "b"));
  if (details && details.length) {
    const list = el("ul");
    for (const d of details) list.append(el("li", `${d.loc[d.loc.length - 1]}: ${d.msg}`));
    body.append(list);
  }
  showState("error");
}

function placeCutMarker() {
  if (threshold === null) return;
  const angle = threshold * 2 * Math.PI - Math.PI / 2; // 0 at 12 o'clock, clockwise like the arc
  $("gauge-cut").setAttribute("cx", (92 + 70 * Math.cos(angle)).toFixed(1));
  $("gauge-cut").setAttribute("cy", (92 + 70 * Math.sin(angle)).toFixed(1));
  $("r-cut-label").textContent = `Cut-off ${cutText().trim()}`;
}

function showResult(r) {
  const pct = probText(r);
  $("gauge-arc").setAttribute("stroke-dasharray", `${(r.churn_probability * GAUGE_LEN).toFixed(1)} ${GAUGE_LEN.toFixed(2)}`);
  $("gauge").setAttribute("aria-label", `Churn probability ${pct}, ${r.prediction === "Churn" ? "at or above" : "below"} the ${cutText()}cut-off`);
  placeCutMarker();
  $("r-prob").textContent = pct;
  $("r-prediction").textContent = r.prediction === "Churn" ? "Likely to churn" : "Likely to stay";
  $("r-risk").textContent = `${r.risk_level} risk`;
  $("r-risk").className = `badge ${r.risk_level}`;
  // Under 50% only happens near the cut-off (e.g. 70.9% churn vs a 71% cut-off): say so instead of a bare low number.
  $("r-conf").textContent = r.confidence_score < 0.5
    ? `Borderline: ${pct} churn, just under the ${cutText()}cut-off`
    : (r.confidence_score * 100).toFixed(1) + "%";
  $("r-lat").textContent = `${r.inference_latency_ms} ms`;
  $("r-ver").textContent = r.model_version;
  $("r-id").textContent = r.request_id;
  showState("result");
}

async function predict(event) {
  event.preventDefault();
  const button = $("submit");
  button.disabled = true;
  button.lastElementChild.textContent = "Predicting…";
  try {
    const { ok, status, body, headers } = await api("/predict/churn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    if (ok) showResult(body);
    else if (status === 422) showError("The API rejected the input (HTTP 422)", body && body.details);
    else if (status === 401) showError("This server requires an API key. Enter it in the box above.");
    else if (status === 429) showError(`Too many predictions from this address. Try again in ${headers.get("Retry-After") || "a few"} seconds.`);
    else showError(`Server error (HTTP ${status}). Try again.`);
  } catch {
    showError("Could not reach the API. It may be waking up; try again in a minute.");
  } finally {
    button.disabled = false;
    button.lastElementChild.textContent = "Predict churn";
    loadHealth();
  }
}

/* ---------- recent ---------- */
async function loadRecent() {
  await Promise.all([loadRecentTable(), loadSummary()]);
}

async function loadSummary() {
  const list = $("summary-list");
  try {
    const { ok, body: s } = await api("/metrics/summary");
    if (!ok) throw new Error();
    list.replaceChildren();
    if (!s.total) {
      list.append(el("dt", "No predictions stored yet."));
      $("summary-table").hidden = true;
      return;
    }
    const risk = ["High", "Medium", "Low"].map((k) => `${k} ${s.by_risk[k] || 0}`).join(" · ");
    for (const [k, v] of [["Predictions stored", s.total.toLocaleString("en-US")], ["Predicted to churn", `${s.churn} (${pct1(s.churn_rate)})`],
      ["Average churn probability", pct1(s.avg_probability)], ["Average inference latency", ms(s.avg_latency_ms)], ["By risk level", risk],
      ["First / last (UTC)", `${s.first_at.replace("T", " ").slice(0, 19)} / ${s.last_at.replace("T", " ").slice(0, 19)}`]]) list.append(el("dt", k), el("dd", v));
    $("summary-table").hidden = false;
    $("summary-body").replaceChildren(...s.by_contract.map((c) => row([CONTRACTS[c.contract_type] || c.contract_type, c.n, pct1(c.churn_rate), pct1(c.avg_probability)])));
  } catch {
    list.replaceChildren(el("dt", "Could not load the summary."));
    $("summary-table").hidden = true;
  }
}

async function loadRecentTable() {
  const tbody = $("recent-body");
  try {
    const { ok, body } = await api("/metrics/recent");
    if (!ok) { tbody.replaceChildren(emptyRow((body && body.message) || "Could not load recent predictions.", 8)); return; }
    if (!body.count) {
      tbody.replaceChildren(emptyRow("No predictions on this server since it started. Make one on the Predict page.", 8));
      return;
    }
    // Newest first.
    tbody.replaceChildren(...body.predictions.slice().reverse().map(({ input, output }) => {
      const track = el("div", undefined, "track"); track.append(bar(`fill-${output.risk_level}`, output.churn_probability * 100));
      const prob = el("div", undefined, "prob"); prob.append(track, el("span", probText(output)));
      const tr = row([`${input.tenure_months} mo`, money(input.monthly_charges), money(input.total_charges), input.support_tickets,
        CONTRACTS[input.contract_type] || input.contract_type, output.prediction, prob, el("span", output.risk_level, `badge ${output.risk_level}`)]);
      tr.cells[0].className = "b"; tr.cells[3].className = "center"; tr.cells[5].className = "b";
      return tr;
    }));
  } catch {
    tbody.replaceChildren(emptyRow("Could not load recent predictions.", 8));
  }
}

/* ---------- drift ---------- */
function psiNodes(features) {
  const items = Object.entries(features).map(([name, f]) => {
    const right = el("span"); right.append(el("b", f.psi.toFixed(4)), el("span", f.status, `st ${f.status}`));
    const top = el("div", undefined, "psi-top"); top.append(el("span", name, "mono"), right);
    const track = el("div", undefined, "psi-track");
    track.append(bar(`bar fill-${f.status}`, (f.psi / 0.4) * 100));
    for (const left of ["25%", "62.5%"]) { const t = el("div", undefined, "tick"); t.style.left = left; track.append(t); } // 0.1 and 0.25 on a 0–0.4 scale
    const item = el("div", undefined, "psi-item"); item.append(top, track);
    return item;
  });
  const scale = el("div", undefined, "scale"); scale.setAttribute("aria-hidden", "true");
  for (const [t, pad] of [["0", ""], ["0.1", "12%"], ["0.25", "18%"], ["0.4", ""]]) { const s = el("span", t); s.style.paddingLeft = pad; scale.append(s); }
  return [...items, scale];
}

function setBadge(id, status) {
  $(id).textContent = cap(status);
  $(id).className = `badge ${status}`;
  $(id).hidden = false;
}

async function loadDrift() {
  const live = $("drift-live-body");
  try {
    const { ok, body } = await api("/metrics/drift");
    if (!ok) throw new Error(); // shown as "Could not load live drift."
    if (body.status === "insufficient_data") {
      $("drift-live-badge").hidden = true;
      $("drift-live-sub").textContent = "Last requests to this server vs training data";
      const box = el("div", undefined, "empty");
      const icon = el("div", undefined, "empty-icon");
      icon.innerHTML = '<svg class="i" aria-hidden="true"><use href="#i-pulse"/></svg>'; // static markup, no data
      const progress = el("div", undefined, "progress"); progress.append(bar("", (body.n_samples / body.min_samples) * 100));
      progress.setAttribute("role", "progressbar"); progress.setAttribute("aria-label", "Requests collected for the drift check");
      progress.setAttribute("aria-valuemin", "0"); progress.setAttribute("aria-valuemax", String(body.min_samples)); progress.setAttribute("aria-valuenow", String(body.n_samples));
      const link = el("a", "Make a prediction →", "btn ghost sm"); link.href = "/predict"; link.dataset.route = "";
      box.append(icon, el("strong", "Not enough live data yet"),
        el("div", `The check starts after ${body.min_samples} predictions. ${body.n_samples} so far on this server.`), progress, link);
      live.replaceChildren(box);
    } else {
      $("drift-live-sub").textContent = `Last ${body.n_samples} requests to this server vs training data`;
      setBadge("drift-live-badge", body.status);
      live.replaceChildren(...psiNodes(body.features));
    }
  } catch {
    live.replaceChildren(el("p", "Could not load live drift.", "fine"));
  }

  const ev = await getEvaluation();
  const real = ev && ev.real_world;
  if (real) {
    $("drift-ref-sub").textContent = `${real.rows.toLocaleString("en-US")} IBM Telco customers vs the synthetic training data · offline, evaluation/evaluate_real.py`;
    setBadge("drift-ref-badge", worst(real.psi_real_vs_training));
    $("drift-ref-body").replaceChildren(...psiNodes(real.psi_real_vs_training));
  } else {
    $("drift-ref-body").replaceChildren(el("p", "Reference results not available.", "fine"));
  }
}

/* ---------- performance ---------- */
function throughputSvg(series) {
  // series: [{ color, fill, points: [[concurrency, rps]] }]; all values are numbers from our own result files.
  const x0 = 70, x1 = 630, yTop = 30, yBot = 190;
  const max = Math.ceil(Math.max(...series.flatMap((s) => s.points.map((p) => p[1]))) / 50) * 50 || 50;
  const n = series[0].points.length;
  const x = (i) => x0 + (n === 1 ? 0 : (i * (x1 - x0)) / (n - 1));
  const y = (v) => yBot - (v / max) * (yBot - yTop);
  const curve = (pts) => pts.map((p, i) => (i === 0 ? `M${x(0)},${y(p[1]).toFixed(1)}`
    : ` C${((x(i - 1) + x(i)) / 2).toFixed(1)},${y(pts[i - 1][1]).toFixed(1)} ${((x(i - 1) + x(i)) / 2).toFixed(1)},${y(p[1]).toFixed(1)} ${x(i)},${y(p[1]).toFixed(1)}`)).join("");
  let s = `<svg viewBox="0 0 680 230" width="100%" role="img" style="display:block" aria-label="${series.map((se) => `${se.name}: ${se.points.map((p) => `${p[1]} requests per second at concurrency ${p[0]}`).join(", ")}`).join(". ")}">`;
  s += '<defs><linearGradient id="optFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#D9AE3A" stop-opacity=".35"/><stop offset="1" stop-color="#D9AE3A" stop-opacity="0"/></linearGradient></defs>';
  for (let k = 0; k <= 4; k++) {
    const v = (max * k) / 4, yy = y(v).toFixed(1);
    s += `<line x1="50" y1="${yy}" x2="660" y2="${yy}" stroke="#E6EDEB"/><text x="40" y="${(+yy + 4).toFixed(1)}" fill="#5A6B68" font-size="12" text-anchor="end">${v}</text>`;
  }
  for (const se of series) {
    const d = curve(se.points);
    if (se.fill) s += `<path d="${d} L${x(n - 1)},${yBot} L${x0},${yBot} Z" fill="url(#optFill)"/>`;
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="3"/>`;
    se.points.forEach((p, i) => {
      s += `<circle cx="${x(i)}" cy="${y(p[1]).toFixed(1)}" r="5" fill="#FFFFFF" stroke="${se.color}" stroke-width="3"/>`;
      s += `<text x="${x(i)}" y="${(y(p[1]) - 12).toFixed(1)}" text-anchor="middle" fill="${se.label}" font-size="12" font-weight="800">${p[1]}</text>`;
    });
  }
  series[0].points.forEach((p, i) => (s += `<text x="${x(i)}" y="215" fill="#5A6B68" font-size="12" text-anchor="middle" font-weight="600">${p[0]} concurrent</text>`));
  return s + "</svg>";
}

async function loadPerformance() {
  const ev = await getEvaluation();
  const base = ev && ev.benchmark_baseline, opt = ev && ev.benchmark_optimized;
  if (!base || !opt) {
    $("latency-bars").replaceChildren(el("p", "Benchmark results not available.", "fine"));
    $("throughput").replaceChildren(el("p", "Benchmark results not available.", "fine"));
  } else {
    const levels = opt.results.map((r) => r.concurrency);
    const at = (file, c) => file.results.find((r) => r.concurrency === c);
    $("lat-sub").textContent = `Before vs after optimisation · ${opt.results[0].requests} requests per level · local machine, ${day(opt.timestamp_utc)}`;
    const max = Math.max(...levels.flatMap((c) => [at(base, c).predict.p50_ms, at(opt, c).predict.p50_ms]));
    const col = (v, color) => {
      const c = el("div", undefined, "col"); const b = el("div");
      b.style.height = Math.max(4, Math.round((v / max) * 150)) + "px"; b.style.background = color;
      c.append(el("span", v >= 1000 ? Math.round(v).toLocaleString("en-US") : String(v)), b);
      return c;
    };
    $("latency-bars").replaceChildren(...levels.map((c) => { const g = el("div"); g.append(col(at(base, c).predict.p50_ms, "#D9AE3A"), col(at(opt, c).predict.p50_ms, "#3E5754")); return g; }));
    $("latency-bars").setAttribute("aria-label", "Median predict latency. " + levels.map((c) => `Concurrency ${c}: ${at(base, c).predict.p50_ms} ms baseline, ${at(opt, c).predict.p50_ms} ms optimized.`).join(" "));
    $("latency-labels").replaceChildren(...levels.map((c) => el("span", `${c} concurrent`)));
    const b10 = at(base, 10), o10 = at(opt, 10);
    if (b10 && o10) {
      $("lat-tip-text").textContent = `/health p95 while 10 predictions run at once: ${ms(b10.health_during_load.p95_ms)} before, ${ms(o10.health_during_load.p95_ms)} after. Inference no longer blocks the event loop.`;
      $("lat-tip").hidden = false;
    }
    $("throughput").innerHTML = throughputSvg([
      { name: "Optimized", color: "#D9AE3A", label: "#7D5600", fill: true, points: opt.results.map((r) => [r.concurrency, r.throughput_rps]) },
      { name: "Baseline", color: "#3E5754", label: "#3E5754", points: base.results.map((r) => [r.concurrency, r.throughput_rps]) },
    ]);
  }

  const runs = [ev && ev.benchmark_render, ev && ev.benchmark_render_high_load].filter(Boolean);
  $("render-sub").textContent = runs.length
    ? `Measured against ${runs[0].url} · ${[...new Set(runs.map((r) => day(r.timestamp_utc)))].join(", ")}`
    : "Render results not available.";
  $("render-body").replaceChildren(...(runs.length
    ? runs.flatMap((f) => f.results).map((r) => row([r.concurrency, r.requests, `${r.throughput_rps} req/s`, ms(r.predict.p50_ms), ms(r.predict.p95_ms), r.predict.errors]))
    : [emptyRow("No Render results recorded.", 6)]));
}

/* ---------- model quality ---------- */
async function loadQuality() {
  const ev = await getEvaluation();
  const t = ev && ev.training;
  const list = $("training-list");
  if (t) {
    list.replaceChildren();
    for (const [k, v] of [["Model version", t.model_version], ["Test accuracy", t.test_accuracy.toFixed(4)], ["Test ROC-AUC", t.test_roc_auc.toFixed(4)],
      ["Training rows", t.n_train_rows.toLocaleString("en-US")], ["Churn cut-off", t.decision_threshold]]) list.append(el("dt", k), el("dd", v));
  } else {
    list.replaceChildren(el("dt", "Training results not available."));
  }

  const cal = ev && ev.threshold_calibration;
  if (cal) {
    const chosen = cal.chosen_threshold;
    const tabs = [
      { id: `holdout_at_${chosen}`, label: `${chosen} · deployed` },
      { id: "holdout_at_0.5", label: "0.5 · default" },
    ].filter((tab) => cal[tab.id]);
    $("cut-sub").textContent = `Holdout: ${cal.holdout_rows.toLocaleString("en-US")} IBM customers, ${pct1(cal.holdout_actual_churn_rate)} churn`;
    $("cut-tabs").replaceChildren(...tabs.map((tab) => {
      const b = el("button", tab.label, "btn"); b.type = "button"; b.dataset.cut = tab.id;
      b.addEventListener("click", () => renderCut(cal, tab.id)); return b;
    }));
    renderCut(cal, tabs[0].id);
  } else {
    $("cut-sub").textContent = "Calibration results not available.";
  }

  const real = ev && ev.real_world;
  if (real) {
    $("real-sub").textContent = `${real.rows.toLocaleString("en-US")} customers, ${pct1(real.actual_churn_rate)} churn · always guessing "stays" scores ${real.majority_class_accuracy.toFixed(4)} accuracy`;
    const line = (name, m) => row([name, m.accuracy.toFixed(4), m.roc_auc.toFixed(4), m.precision_churn.toFixed(4), m.recall_churn.toFixed(4), pct1(m.predicted_churn_rate)]);
    $("real-body").replaceChildren(
      ...Object.entries(real.deployed_model).map(([k, m]) => line(`Deployed model, ${k.replace("=", " = ")}`, m)),
      line("Logistic regression trained on IBM data (5-fold CV)", real.baseline_trained_on_real_cv5));
  } else {
    $("real-body").replaceChildren(emptyRow("Real-world results not available.", 6));
  }
}

function renderCut(cal, id) {
  const c = cal[id];
  document.querySelectorAll("[data-cut]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cut === id)));
  $("cut-metrics").replaceChildren(...[
    ["Accuracy", c.accuracy.toFixed(4), c.accuracy, "#3E5754"],
    ["Churn precision", c.precision_churn.toFixed(4), c.precision_churn, "#3E5754"],
    ["Churn recall", c.recall_churn.toFixed(4), c.recall_churn, "#D9AE3A"],
    ["Flagged as churn", pct1(c.predicted_churn_rate), c.predicted_churn_rate, "#D9AE3A"],
  ].map(([label, value, v, color]) => {
    const top = el("div", undefined, "metric-top"); top.append(el("span", label), el("b", value));
    const track = el("div", undefined, "metric-track"); track.append(bar("", v * 100, color));
    const m = el("div", undefined, "metric"); m.append(top, track); return m;
  }));
  $("cut-note").textContent = `Flags ${pct1(c.predicted_churn_rate)} of customers (${pct1(cal.holdout_actual_churn_rate)} actually leave). `
    + `${pct1(c.precision_churn)} of flagged customers really churn, and ${pct1(c.recall_churn)} of churners are caught.`;
}

/* ---------- phone menu ---------- */
function setMenu(open) {
  document.querySelector("aside").classList.toggle("open", open);
  $("menu-btn").setAttribute("aria-expanded", String(open));
  $("menu-btn").setAttribute("aria-label", open ? "Close menu" : "Open menu");
}
$("menu-btn").addEventListener("click", () => setMenu($("menu-btn").getAttribute("aria-expanded") !== "true"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("menu-btn").getAttribute("aria-expanded") === "true") { setMenu(false); $("menu-btn").focus(); }
});

$("keybox").addEventListener("submit", (e) => {
  e.preventDefault();
  apiKey = $("api-key").value.trim();
  try { sessionStorage.setItem("apiKey", apiKey); } catch { /* storage blocked: kept in memory until the tab reloads */ }
  $("keybox").hidden = true;
  route(false); // reload the current page with the key
});

/* ---------- wiring ---------- */
const presetButtons = document.querySelectorAll("[data-preset]");
presetButtons.forEach((b) => b.addEventListener("click", () => {
  const preset = PRESETS[b.dataset.preset];
  for (const name of FIELDS) $(name).value = preset[name];
  presetButtons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
}));
// Editing a field means the form no longer matches an example.
$("form").addEventListener("input", () => presetButtons.forEach((o) => o.setAttribute("aria-pressed", "false")));
$("form").addEventListener("submit", predict);
route(false);
