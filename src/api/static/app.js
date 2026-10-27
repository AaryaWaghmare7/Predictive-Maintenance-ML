"use strict";

// Original model aliases/order. Values below are the user's demo observations,
// not raw physical measurements or a new labeled model-evaluation dataset.
const features = ["Voltage (V)", "Current (A)", "Motor Speed (RPM)", "Temperature (°C)", "Vibration (g)", "Ambient Temp (°C)", "Humidity (%)"];
const samples = [
  {name: "Normal", values: [0.939, 0.200, 0.544, 0.240, 0.301, 0.582, 0.397]},
  {name: "Motor Fault", values: [0.805, 0.597, 0.073, 0.180, 0.921, 0.128, 0.125]},
  {name: "Inverter Fault", values: [0.586, 0.275, 0.355, 0.398, 0.452, 0.728, 0.326]},
  {name: "Battery Fault", values: [0.180, 0.453, 0.393, 0.782, 0.227, 0.287, 0.515]},
];
const byId = (id) => document.getElementById(id);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const sampleButtons = Array.from(document.querySelectorAll("[data-sample]"));
let modelReady = false;
let faultBusy = false;
let csvBusy = false;
let telemetryResult = null;
let selectedFault = null;
let chartButtons = [];

async function requestJSON(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (_error) {
    throw new Error("Cannot reach the localhost API. Check that the server is running, then try again.");
  }
  let data;
  try {
    data = await response.json();
  } catch (_error) {
    throw new Error(`The server returned an unreadable response (HTTP ${response.status}). Check the localhost server logs.`);
  }
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((error) => `${error.loc.slice(1).join(".")}: ${error.msg}`).join("\n")
      : data.detail;
    throw new Error(detail || `Request failed (HTTP ${response.status})`);
  }
  return data;
}

function syncActions() {
  byId("analyze-button").disabled = !modelReady || faultBusy;
  byId("upload-button").disabled = !modelReady || csvBusy;
  byId("explore-demo").setAttribute("aria-disabled", String(faultBusy));
}

async function checkHealth() {
  try {
    const health = await requestJSON("/health");
    modelReady = health.fault_model_status === "ready";
    byId("model-status-text").textContent = modelReady ? "Model Online" : "Model Unavailable";
    byId("model-status").dataset.state = modelReady ? "ready" : "unavailable";
    byId("prediction-version").textContent = health.model_version || "Unavailable";
    byId("health-message").textContent = modelReady ? "" : "Fault model unavailable. Provide the trusted saved pipeline and compatible dependencies, then restart. No automatic training is performed.";
  } catch (error) {
    modelReady = false;
    byId("model-status-text").textContent = "Backend Offline";
    byId("model-status").dataset.state = "unavailable";
    byId("prediction-version").textContent = "Unavailable";
    byId("health-message").textContent = error.message;
  }
  syncActions();
}

function clearFaultResult() {
  byId("fault-result").hidden = true;
  byId("fault-empty").hidden = false;
  byId("result-state").textContent = "Awaiting input";
  byId("fault-error").textContent = "";
}
byId("fault-form").addEventListener("input", () => {
  clearFaultResult();
  sampleButtons.forEach((button) => button.classList.remove("selected"));
  byId("sample-status").textContent = "Inputs updated. Click Analyze Vehicle to run diagnosis.";
});
function loadSample(index) {
  if (faultBusy) return;
  const sample = samples[index];
  features.forEach((feature, position) => {
    byId("fault-form").elements.namedItem(feature).value = sample.values[position].toFixed(3);
  });
  sampleButtons.forEach((button) => button.classList.toggle("selected", Number(button.dataset.sample) === index));
  clearFaultResult();
  byId("sample-status").textContent = `${sample.name} sample loaded. Click Analyze Vehicle to run diagnosis.`;
}
sampleButtons.forEach((button) => button.addEventListener("click", () => loadSample(Number(button.dataset.sample))));
byId("explore-demo").addEventListener("click", () => loadSample(1));

byId("fault-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (faultBusy || !modelReady || !event.target.reportValidity()) return;
  clearFaultResult();
  faultBusy = true;
  syncActions();
  event.target.setAttribute("aria-busy", "true");
  byId("analyze-label").textContent = "Analyzing…";
  byId("result-state").textContent = "Processing";
  const payload = Object.fromEntries(features.map((feature) => [feature, Number(event.target.elements.namedItem(feature).value)]));
  const inputs = Array.from(event.target.querySelectorAll("input"));
  inputs.forEach((input) => { input.disabled = true; });
  sampleButtons.forEach((button) => { button.disabled = true; });
  try {
    const result = await requestJSON("/predict-fault", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    byId("predicted-fault").textContent = result.predicted_fault;
    byId("model-confidence").textContent = percent(result.model_confidence);
    byId("confidence-fill").style.width = percent(result.model_confidence);
    byId("prediction-status").textContent = result.status === "normal" ? "Normal label predicted" : "Fault label predicted";
    byId("prediction-version").textContent = result.model_version;
    byId("fault-result").classList.toggle("abnormal", result.predicted_class !== 0);
    byId("result-state").textContent = result.predicted_class === 0 ? "Normal" : "Fault detected";
    byId("fault-empty").hidden = true;
    byId("fault-result").hidden = false;
  } catch (error) {
    byId("fault-error").textContent = error.message;
    byId("result-state").textContent = "Analysis unavailable";
    await checkHealth();
  } finally {
    inputs.forEach((input) => { input.disabled = false; });
    sampleButtons.forEach((button) => { button.disabled = false; });
    faultBusy = false;
    event.target.setAttribute("aria-busy", "false");
    byId("analyze-label").textContent = "Analyze Vehicle";
    syncActions();
  }
});

// Never insert uploaded column names, filenames or response text as HTML.
function appendTableRow(body, values) {
  const row = document.createElement("tr");
  values.forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.appendChild(cell);
  });
  body.appendChild(row);
}

function renderDistribution(classes) {
  const body = byId("class-distribution");
  body.replaceChildren();
  chartButtons = [];
  Object.entries(classes).forEach(([name, summary], index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `distribution-button distribution-class-${index}`;
    item.dataset.fault = name;
    item.setAttribute("aria-pressed", "false");
    item.setAttribute("aria-label", `Filter ${name}: ${summary.count} ${summary.count === 1 ? "observation" : "observations"}, ${summary.percentage.toFixed(1)} percent`);
    item.addEventListener("click", () => selectCsvClass(selectedFault === name ? null : name));
    chartButtons.push(item);
    const heading = document.createElement("span");
    heading.className = "distribution-label";
    const label = document.createElement("span");
    label.textContent = name;
    const count = document.createElement("strong");
    count.textContent = summary.count;
    heading.append(label, count);
    const track = document.createElement("span");
    track.className = "distribution-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = "distribution-fill";
    bar.style.width = `${Math.max(0, Math.min(100, summary.percentage))}%`;
    track.appendChild(bar);
    const proportion = document.createElement("span");
    proportion.className = "distribution-percent";
    proportion.textContent = `${summary.percentage.toFixed(1)}% of observations`;
    item.append(heading, track, proportion);
    body.appendChild(item);
  });
  // Exact counts drive arc sizes; rounded API percentages are for labels only.
  const colors = ["#a8a6ab", "#D10721", "#ff796e", "#f0d4da"];
  const total = Object.values(classes).reduce((sum, item) => sum + item.count, 0);
  let offset = 0;
  const arcs = Object.values(classes).map((item, index) => {
    const start = offset;
    offset += total ? item.count / total * 100 : 0;
    return `${colors[index]} ${start}% ${offset}%`;
  });
  byId("fault-donut").style.background = total ? `conic-gradient(${arcs.join(", ")})` : "";
  byId("fault-donut").setAttribute("aria-label", `${total} analyzed observations. ${Object.entries(classes).map(([name, item]) => `${name}: ${item.count}`).join(". ")}`);
  byId("fault-donut").classList.add("has-data");
  byId("donut-total").textContent = total;
  byId("distribution-empty").hidden = true;
}

function selectCsvClass(name) {
  if (!telemetryResult) return;
  selectedFault = name;
  if (name !== null) byId("row-details").open = true;
  byId("row-predictions").replaceChildren();
  let shown = 0;
  telemetryResult.predictions.forEach((prediction, index) => {
    if (name !== null && prediction.predicted_fault !== name) return;
    appendTableRow(byId("row-predictions"), [index + 1, prediction.predicted_fault, percent(prediction.model_confidence)]);
    shown += 1;
  });
  chartButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.fault === name)));
  byId("clear-csv-filter").disabled = name === null;
  byId("csv-filter-status").textContent = name === null
    ? "All classes. Select a chart legend to filter the row table; select it again to clear."
    : `${name}: ${shown} matching returned ${shown === 1 ? "observation" : "observations"}. Whole-file counts above are unchanged.`;
  byId("row-limit-note").textContent = `Showing ${shown} ${shown === 1 ? "row" : "rows"} from ${telemetryResult.predictions_returned} returned predictions${telemetryResult.predictions_truncated ? " (first 1,000 only)" : ""}. Summary includes all ${telemetryResult.rows_analyzed} rows. File order is not time order.`;
}
byId("clear-csv-filter").addEventListener("click", () => selectCsvClass(null));

function clearTelemetry() {
  telemetryResult = null;
  selectedFault = null;
  chartButtons = [];
  byId("csv-result").hidden = true;
  byId("telemetry-empty").hidden = false;
  byId("download-results").disabled = true;
  byId("class-distribution").replaceChildren();
  byId("distribution-empty").hidden = false;
  byId("fault-donut").style.background = "";
  byId("fault-donut").classList.remove("has-data");
  byId("fault-donut").setAttribute("aria-label", "No telemetry analyzed");
  byId("donut-total").textContent = "—";
  byId("csv-error").textContent = "";
}

// Only returned predictions are downloadable: never imply that capped API
// details include all uploaded rows, and never export untrusted CSV cell text.
byId("download-results").addEventListener("click", () => {
  if (!telemetryResult || csvBusy) return;
  const rows = [["Observation #", "Predicted class", "Predicted fault", "Model confidence (0-1)"]];
  telemetryResult.predictions.forEach((item, index) => {
    rows.push([index + 1, item.predicted_class, item.predicted_fault, item.model_confidence]);
  });
  const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "nev_predictions_returned_rows.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

byId("telemetry-file").addEventListener("change", () => {
  clearTelemetry();
  const file = byId("telemetry-file").files[0];
  byId("file-name").textContent = file ? file.name : "No file selected";
  byId("csv-upload-status").textContent = "Your file is not retained by the application.";
});
byId("csv-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (csvBusy || !modelReady) return;
  clearTelemetry();
  csvBusy = true;
  syncActions();
  event.target.setAttribute("aria-busy", "true");
  byId("upload-label").textContent = "Analyzing…";
  byId("csv-upload-status").textContent = "Validating all rows and running the saved pipeline…";
  const input = byId("telemetry-file");
  input.disabled = true;
  try {
    const file = input.files[0];
    if (!file) throw new Error("Choose a telemetry CSV before analyzing.");
    if (file.size === 0) throw new Error("CSV is empty. Provide a header and at least one sensor observation.");
    if (file.size > 5 * 1024 * 1024) throw new Error("CSV exceeds the 5 MiB upload limit");
    const upload = new FormData();
    upload.append("file", file);
    // Browser supplies Content-Type and the multipart boundary; do not set them.
    const result = await requestJSON("/analyze-csv", {method: "POST", body: upload});
    telemetryResult = result;
    byId("csv-rows").textContent = result.rows_analyzed;
    byId("csv-confidence").textContent = percent(result.mean_model_confidence);
    byId("csv-dominant").textContent = result.dominant_abnormal_faults.length
      ? result.dominant_abnormal_faults.join(" / ") + (result.dominant_abnormal_faults.length > 1 ? " (tie)" : "")
      : "None among model predictions";
    renderDistribution(result.classes);
    selectCsvClass(null);
    const notes = [];
    if (result.ignored_columns.length) notes.push(`Ignored columns (not model inputs): ${result.ignored_columns.join(", ")}.`);
    if (Object.keys(result.normalized_headers).length) notes.push("Surrounding header whitespace was trimmed for matching only.");
    byId("csv-parsing-note").textContent = notes.join(" ");
    byId("csv-result").hidden = false;
    byId("telemetry-empty").hidden = true;
    byId("download-results").disabled = false;
    byId("download-results").textContent = `Download ${result.predictions_returned} returned predictions`;
    byId("csv-upload-status").textContent = "Analysis complete. Upload not retained.";
  } catch (error) {
    byId("csv-error").textContent = error.message;
    byId("csv-upload-status").textContent = "Upload not analyzed. Correct the file and try again.";
    await checkHealth();
  } finally {
    input.disabled = false;
    csvBusy = false;
    event.target.setAttribute("aria-busy", "false");
    byId("upload-label").textContent = "Analyze Telemetry";
    syncActions();
  }
});
// Motion is optional: inference and forms never depend on an animation library.
// Observe the hero so offscreen/reduced/mobile views have no scroll listener.
function initializeMotion() {
  const systemMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const compact = window.matchMedia("(max-width: 850px)");
  const hero = byId("home");
  const toggle = byId("motion-toggle");
  const sections = Array.from(document.querySelectorAll(".overview-grid > .card, .telemetry-card, #diagnostics .card, .future-grid > .card, .resource-grid > .card"));
  let userReduced = document.documentElement.dataset.userMotion === "reduced";
  let heroVisible = !("IntersectionObserver" in window);
  let frame = null;
  let disposed = false;
  const isReduced = () => systemMotion.matches || userReduced;
  const render = () => {
    frame = null;
    if (disposed) return;
    const bounds = hero.getBoundingClientRect();
    const progress = isReduced() || compact.matches ? 0 : Math.max(0, Math.min(1, -bounds.top / Math.max(1, bounds.height)));
    hero.style.setProperty("--hero-progress", String(progress));
  };
  const schedule = () => {
    if (frame === null) frame = window.requestAnimationFrame(render);
  };
  const configure = () => {
    const reduced = isReduced();
    document.documentElement.dataset.motion = reduced ? "reduced" : "full";
    toggle.setAttribute("aria-pressed", String(reduced));
    toggle.textContent = systemMotion.matches ? "Reduced motion (system)" : userReduced ? "Enable motion" : "Reduce motion";
    toggle.disabled = systemMotion.matches;
    window.removeEventListener("scroll", schedule);
    const active = heroVisible && !reduced && !compact.matches;
    hero.dataset.motionActive = String(active);
    if (active) window.addEventListener("scroll", schedule, {passive: true});
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    render();
  };
  const toggleMotion = () => {
    userReduced = !userReduced;
    document.documentElement.dataset.userMotion = userReduced ? "reduced" : "full";
    configure();
  };
  toggle.addEventListener("click", toggleMotion);
  systemMotion.addEventListener("change", configure);
  compact.addEventListener("change", configure);

  let panels;
  let heroObserver;
  if ("IntersectionObserver" in window) {
    panels = new window.IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        panels.unobserve(entry.target);
      });
    }, {threshold: .08});
    sections.forEach((section, index) => {
      section.style.setProperty("--reveal-delay", `${index % 3 * .08}s`);
      section.classList.add("scroll-reveal");
      panels.observe(section);
    });
    heroObserver = new window.IntersectionObserver((entries) => {
      heroVisible = entries[0].isIntersecting;
      configure();
    });
    heroObserver.observe(hero);
  }
  configure();

  return () => {
    disposed = true;
    window.removeEventListener("scroll", schedule);
    systemMotion.removeEventListener("change", configure);
    compact.removeEventListener("change", configure);
    toggle.removeEventListener("click", toggleMotion);
    if (frame !== null) window.cancelAnimationFrame(frame);
    panels?.disconnect();
    heroObserver?.disconnect();
    hero.dataset.motionActive = "false";
  };
}

checkHealth();
if (typeof window !== "undefined") {
  let stopMotion = initializeMotion();
  window.addEventListener("pagehide", () => stopMotion());
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) stopMotion = initializeMotion();
  });
}
