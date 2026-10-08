"use strict";

const features = ["Voltage (V)", "Current (A)", "Motor Speed (RPM)", "Temperature (°C)", "Vibration (g)", "Ambient Temp (°C)", "Humidity (%)"];
const syntheticExample = [0.85, 0.5, 0.2, 0.4, 0.75, 0.5, 0.5];
const byId = (id) => document.getElementById(id);
const percent = (value) => `${(value * 100).toFixed(2)}%`;
let modelReady = false;

async function requestJSON(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((error) => `${error.loc.slice(1).join(".")}: ${error.msg}`).join("\n")
      : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

async function checkHealth() {
  try {
    const health = await requestJSON("/health");
    modelReady = health.fault_model_status === "ready";
    byId("model-status").textContent = modelReady
      ? `Saved fault model ready · ${health.model_version}`
      : "Fault model unavailable. Provide the trusted local pipeline and compatible dependencies, then restart. No automatic training is performed.";
  } catch (_error) {
    byId("model-status").textContent = "Backend unavailable. Start the localhost application.";
  }
  byId("analyze-button").disabled = !modelReady;
  byId("upload-button").disabled = !modelReady;
}

function clearFaultResult() {
  byId("fault-result").hidden = true;
  byId("fault-error").textContent = "";
}
byId("fault-form").addEventListener("input", clearFaultResult);
byId("example-button").addEventListener("click", () => {
  features.forEach((feature, index) => {
    byId("fault-form").elements.namedItem(feature).value = syntheticExample[index];
  });
  clearFaultResult();
});

byId("fault-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearFaultResult();
  if (!modelReady) return;
  const button = byId("analyze-button");
  button.disabled = true;
  button.textContent = "Analyzing…";
  const payload = Object.fromEntries(features.map((feature) => [feature, Number(event.target.elements.namedItem(feature).value)]));
  const inputs = Array.from(event.target.querySelectorAll("input"));
  inputs.forEach((input) => { input.disabled = true; });
  byId("example-button").disabled = true;
  try {
    const result = await requestJSON("/predict-fault", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    byId("predicted-fault").textContent = result.predicted_fault;
    byId("model-confidence").textContent = percent(result.model_confidence);
    byId("prediction-status").textContent = result.status === "normal" ? "Normal label predicted" : "Fault label predicted";
    byId("prediction-version").textContent = result.model_version;
    byId("fault-result").classList.toggle("abnormal", result.predicted_class !== 0);
    byId("fault-result").hidden = false;
  } catch (error) {
    byId("fault-error").textContent = error.message;
  } finally {
    inputs.forEach((input) => { input.disabled = false; });
    byId("example-button").disabled = false;
    button.disabled = !modelReady;
    button.textContent = "Analyze Vehicle";
  }
});

function appendTableRow(body, values) {
  const row = document.createElement("tr");
  values.forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.appendChild(cell);
  });
  body.appendChild(row);
}

byId("telemetry-file").addEventListener("change", () => {
  byId("csv-result").hidden = true;
  byId("csv-error").textContent = "";
});
byId("csv-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  byId("csv-result").hidden = true;
  byId("csv-error").textContent = "";
  if (!modelReady) return;
  const button = byId("upload-button");
  button.disabled = true;
  button.textContent = "Analyzing…";
  byId("telemetry-file").disabled = true;
  try {
    const file = byId("telemetry-file").files[0];
    if (file.size > 5 * 1024 * 1024) throw new Error("CSV exceeds the 5 MiB upload limit");
    const upload = new FormData();
    upload.append("file", file);
    const result = await requestJSON("/analyze-csv", {method: "POST", body: upload});
    byId("csv-rows").textContent = result.rows_analyzed;
    byId("csv-confidence").textContent = percent(result.mean_model_confidence);
    byId("csv-dominant").textContent = result.dominant_abnormal_faults.length
      ? result.dominant_abnormal_faults.join(" / ") + (result.dominant_abnormal_faults.length > 1 ? " (tie)" : "")
      : "None among model predictions";
    byId("class-summary").replaceChildren();
    Object.entries(result.classes).forEach(([name, summary]) => appendTableRow(byId("class-summary"), [name, summary.count, `${summary.percentage.toFixed(2)}%`]));
    byId("row-predictions").replaceChildren();
    result.predictions.forEach((prediction, index) => appendTableRow(byId("row-predictions"), [index + 1, prediction.predicted_fault, percent(prediction.model_confidence)]));
    byId("row-limit-note").textContent = `Showing ${result.predictions_returned} row predictions${result.predictions_truncated ? " (first 1,000 only)" : ""}. Summary includes all ${result.rows_analyzed} rows. File order is not time order.`;
    byId("csv-result").hidden = false;
  } catch (error) {
    byId("csv-error").textContent = error.message;
  } finally {
    byId("telemetry-file").disabled = false;
    button.disabled = !modelReady;
    button.textContent = "Analyze CSV";
  }
});

checkHealth();
