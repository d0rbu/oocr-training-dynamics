"use strict";

const DATA_URL = "data/experiment.json";
const CONDITION_LABELS = {
  correct: "Correct I/O",
  wrong_alias: "Wrong alias",
  wrong_impl: "Wrong implementation",
};
const METRIC_LABELS = {
  correct_probability: "mean intended probability",
  code_probability: "code-choice probability",
  language_probability: "language-choice probability",
  correct_accuracy: "mean intended accuracy",
  planted_probability: "mean planted probability",
  freeform_accuracy: "exact-lambda rate",
};
const SECONDARY_METRICS = {
  correct_probability: "planted_probability",
  correct_accuracy: "planted_accuracy",
};
const state = {
  data: null,
  model: "olmo3-7b",
  condition: "correct",
  curveMetric: "correct_probability",
  checkpointIndex: 0,
  patchMode: "across_time",
  patchMetric: "delta",
  recipientIndex: 17,
  donorIndex: 0,
  functionId: "identity",
};

function svg(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function el(name, attributes = {}, text = "") {
  const node = document.createElement(name);
  Object.entries(attributes).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  });
  if (text) node.textContent = text;
  return node;
}

function formatExamples(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}k`;
  return String(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function curveRows() {
  return state.data.curves[state.model][state.condition];
}

function setupStatus() {
  const synthetic = state.data.status !== "real_complete";
  const pill = document.getElementById("status-pill");
  pill.textContent = synthetic ? "Preregistered preview · synthetic values" : "Complete measured results";
  const warning = document.getElementById("warning-banner");
  if (state.data.warning) {
    warning.hidden = false;
    warning.textContent = state.data.warning;
  }
  document.getElementById("footer-status").textContent = synthetic
    ? "Visualization shell only · no GPU results yet"
    : "All nine training runs measured";
}

function buildModelControls() {
  const container = document.getElementById("model-controls");
  container.replaceChildren();
  Object.entries(state.data.models).forEach(([key, model]) => {
    const button = el("button", { type: "button", "data-model": key }, model.label);
    if (key === state.model) button.classList.add("active");
    if (model.provisional) button.title = "Provisional closest-size substitute; confirmation required";
    button.addEventListener("click", () => {
      state.model = key;
      container.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
      renderAll();
    });
    container.append(button);
  });
}

function buildConditionControls() {
  const container = document.getElementById("condition-controls");
  container.replaceChildren();
  state.data.conditions.forEach((condition) => {
    const button = el("button", { type: "button", "data-condition": condition }, CONDITION_LABELS[condition]);
    if (condition === state.condition) button.classList.add("active");
    button.addEventListener("click", () => {
      state.condition = condition;
      container.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
      renderAll();
    });
    container.append(button);
  });
}

function buildFunctionSelect() {
  const select = document.getElementById("function-select");
  select.replaceChildren();
  state.data.functions.forEach((fn) => {
    select.append(el("option", { value: fn.id }, `${fn.alias} · ${fn.definition}`));
  });
  select.value = state.functionId;
  select.addEventListener("change", () => {
    state.functionId = select.value;
    renderPatching();
  });
}

function setupButtons(selector, dataKey, stateKey, callback) {
  document.querySelectorAll(`${selector} button`).forEach((button) => {
    button.addEventListener("click", () => {
      state[stateKey] = button.dataset[dataKey];
      document.querySelectorAll(`${selector} button`).forEach((item) => item.classList.toggle("active", item === button));
      callback();
    });
  });
}

function renderCurve() {
  const rows = curveRows();
  const metric = state.curveMetric;
  const chart = document.getElementById("curve-chart");
  chart.replaceChildren();
  const width = 920;
  const height = 360;
  const margin = { left: 52, right: 22, top: 18, bottom: 38 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const maxLog = Math.log1p(rows.at(-1).examples_seen);
  const x = (value) => margin.left + (Math.log1p(value) / maxLog) * innerWidth;
  const y = (value) => margin.top + (1 - value) * innerHeight;

  const defs = svg("defs");
  const gradient = svg("linearGradient", { id: "curve-gradient", x1: "0", y1: "0", x2: "0", y2: "1" });
  gradient.append(svg("stop", { offset: "0%", "stop-color": "#1c5b45", "stop-opacity": ".22" }));
  gradient.append(svg("stop", { offset: "100%", "stop-color": "#1c5b45", "stop-opacity": "0" }));
  defs.append(gradient);
  chart.append(defs);

  [0, .2, .4, .6, .8, 1].forEach((value) => {
    chart.append(svg("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: "grid-line" }));
    const label = svg("text", { x: margin.left - 10, y: y(value) + 4, class: "axis-label", "text-anchor": "end" });
    label.textContent = `${Math.round(value * 100)}%`;
    chart.append(label);
  });
  [0, 64, 4096, 16384, 65536, 96000].forEach((value) => {
    const label = svg("text", { x: x(value), y: height - 10, class: "axis-label", "text-anchor": "middle" });
    label.textContent = formatExamples(value);
    chart.append(label);
  });

  const points = rows.map((row) => [x(row.examples_seen), y(row[metric])]);
  const line = points.map(([px, py], index) => `${index === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`).join(" ");
  const area = `${line} L${points.at(-1)[0]},${y(0)} L${points[0][0]},${y(0)} Z`;
  chart.append(svg("path", { d: area, class: "curve-area" }));
  chart.append(svg("path", { d: line, class: "curve-primary" }));

  const secondaryKey = SECONDARY_METRICS[metric];
  if (secondaryKey && rows[0][secondaryKey] !== undefined) {
    const secondary = rows.map((row, index) => `${index === 0 ? "M" : "L"}${x(row.examples_seen).toFixed(2)},${y(row[secondaryKey]).toFixed(2)}`).join(" ");
    chart.append(svg("path", { d: secondary, class: "curve-secondary" }));
  }

  const selected = rows[state.checkpointIndex];
  const cursorX = x(selected.examples_seen);
  chart.append(svg("line", { x1: cursorX, x2: cursorX, y1: margin.top, y2: y(0), class: "curve-cursor" }));
  chart.append(svg("circle", { cx: cursorX, cy: y(selected[metric]), r: 7, class: "curve-dot" }));

  document.getElementById("examples-value").textContent = selected.examples_seen.toLocaleString();
  document.getElementById("step-value").textContent = selected.step.toLocaleString();
  document.getElementById("metric-value").textContent = formatPercent(selected[metric]);
  document.getElementById("metric-readout-label").textContent = METRIC_LABELS[metric];
  document.getElementById("checkpoint-label").textContent = selected.step === 0 ? "frozen base" : `step ${selected.step}`;
  document.getElementById("curve-kicker").textContent = `${state.data.models[state.model].label} · ${CONDITION_LABELS[state.condition]}`;
  document.getElementById("curve-title").textContent = METRIC_LABELS[metric].replace(/^./, (letter) => letter.toUpperCase());
  document.getElementById("curve-note").textContent = state.condition === "correct"
    ? "The planted and intended targets coincide in the correct condition; the control distinction appears after selecting a planted-wrong corpus."
    : "A planted rise with a flat intended curve means the model learned the deliberately wrong world—not that training failed.";
}

function optionSet(functionId) {
  const functions = state.data.functions;
  const index = functions.findIndex((fn) => fn.id === functionId);
  return Array.from({ length: 5 }, (_, offset) => functions[(index + offset) % functions.length]);
}

function curveAt(index) {
  return curveRows()[Math.max(0, Math.min(index, curveRows().length - 1))];
}

function distribute(correctIndex, targetProbability, favoredOther = 1) {
  const values = Array(5).fill((1 - targetProbability) / 4);
  values[correctIndex] = targetProbability;
  if (favoredOther !== correctIndex) {
    const move = Math.min(.08, values[(favoredOther + 1) % 5] * .6);
    values[favoredOther] += move;
    values[(favoredOther + 1) % 5] -= move;
  }
  return values;
}

function syntheticPatch() {
  const layers = state.data.models[state.model].layer_count;
  const options = optionSet(state.functionId);
  const correctIndex = 0;
  const recipientCurve = curveAt(state.recipientIndex);
  const donorCurve = curveAt(state.patchMode === "across_time" ? state.donorIndex : state.recipientIndex);
  let recipient;
  let source;
  if (state.patchMode === "across_time") {
    recipient = distribute(correctIndex, recipientCurve.correct_probability, 2);
    source = distribute(correctIndex, donorCurve.correct_probability, 2);
  } else {
    source = distribute(correctIndex, recipientCurve.correct_probability, 2);
    recipient = distribute(correctIndex, Math.max(.08, .23 - recipientCurve.correct_probability * .08), 1);
  }
  const time = state.recipientIndex / (state.data.checkpoints.length - 1);
  const donorGap = state.patchMode === "across_time" ? (state.recipientIndex - state.donorIndex) / 17 : .85;
  const center = .42 + .30 * time;
  const width = .13 + .09 * (1 - time);
  const matrix = [];
  for (let layer = 0; layer < layers; layer += 1) {
    const depth = layer / Math.max(1, layers - 1);
    const core = Math.exp(-((depth - center) ** 2) / (2 * width ** 2));
    const late = .35 * Math.exp(-((depth - .91) ** 2) / .018);
    const strength = Math.min(1, donorGap * (core + late));
    const row = recipient.map((value, choice) => {
      const choiceStrength = strength * (choice === correctIndex ? 1 : .88 + .08 * Math.sin(choice + layer));
      return value + choiceStrength * (source[choice] - value);
    });
    const sum = row.reduce((total, value) => total + Math.max(0, value), 0);
    matrix.push(row.map((value) => Math.max(0, value) / sum));
  }
  return { layers, options, correctIndex, recipient, source, matrix, measured: false };
}

function measuredPatch() {
  const mode = state.patchMode;
  const recipientStep = state.data.checkpoints[state.recipientIndex];
  const donorIndex = mode === "across_time" ? state.donorIndex : state.recipientIndex;
  const donorStep = state.data.checkpoints[donorIndex];
  const record = state.data.patches?.[state.model]?.[state.condition]?.[mode]
    ?.[String(recipientStep)]?.[String(donorStep)]?.[state.functionId];
  if (!record) return null;
  const options = record.choice_function_ids.map((id) => {
    const fn = state.data.functions.find((item) => item.id === id);
    if (!fn) throw new Error(`Patch artifact references unknown function ${id}`);
    return fn;
  });
  const layerCount = Math.max(...record.cells.map((cell) => cell.layer)) + 1;
  const matrix = Array.from({ length: layerCount }, () => Array(5).fill(Number.NaN));
  record.cells.forEach((cell) => {
    matrix[cell.layer][cell.choice_index] = cell.probability;
  });
  if (matrix.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("Patch artifact does not contain a complete layer-by-choice grid");
  }
  return {
    layers: layerCount,
    options,
    correctIndex: record.correct_choice_index,
    recipient: record.recipient_probabilities,
    source: record.source_probabilities,
    matrix,
    measured: true,
  };
}

function patchData() {
  return measuredPatch() ?? syntheticPatch();
}

function colorFor(value, metric) {
  if (metric === "probability") {
    const amount = Math.max(0, Math.min(1, value));
    return `rgb(${Math.round(67 + amount * 172)}, ${Math.round(89 + amount * 103)}, ${Math.round(81 - amount * 20)})`;
  }
  const clipped = Math.max(-1, Math.min(1, value));
  const neutral = [238, 232, 216];
  const endpoint = clipped >= 0 ? [239, 119, 95] : [93, 121, 185];
  const amount = Math.abs(clipped);
  return `rgb(${neutral.map((channel, index) => Math.round(channel + (endpoint[index] - channel) * amount)).join(",")})`;
}

function bindHeatTooltip(cell, html) {
  const tooltip = document.getElementById("tooltip");
  const show = (event) => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    const left = Math.min(window.innerWidth - 340, event.clientX + 14);
    const top = Math.min(window.innerHeight - 130, event.clientY + 14);
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  };
  cell.addEventListener("mousemove", show);
  cell.addEventListener("focus", () => show({ clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 }));
  ["mouseleave", "blur"].forEach((name) => cell.addEventListener(name, () => { tooltip.hidden = true; }));
}

function renderPatching() {
  const patch = patchData();
  const heatmap = document.getElementById("patch-heatmap");
  heatmap.replaceChildren();
  heatmap.style.gridTemplateColumns = `200px repeat(${patch.layers}, minmax(19px, 1fr))`;
  heatmap.append(el("div"));
  for (let layer = 0; layer < patch.layers; layer += 1) {
    heatmap.append(el("div", { class: "heatmap-layer" }, layer % 4 === 0 ? String(layer) : "·"));
  }
  const letters = "ABCDE";
  patch.options.forEach((option, choice) => {
    const label = el("div", { class: `heatmap-choice${choice === patch.correctIndex ? " correct" : ""}` });
    label.append(el("b", {}, letters[choice]));
    label.append(el("span", { title: option.definition }, option.definition));
    heatmap.append(label);
    for (let layer = 0; layer < patch.layers; layer += 1) {
      const probability = patch.matrix[layer][choice];
      const delta = probability - patch.recipient[choice];
      const denominator = patch.source[choice] - patch.recipient[choice];
      const normalized = Math.abs(denominator) < 1e-8 ? 0 : delta / denominator;
      const value = state.patchMetric === "probability" ? probability : state.patchMetric === "normalized" ? normalized : delta / .25;
      const cell = el("div", { class: "heat-cell", tabindex: "0" });
      cell.style.background = colorFor(value, state.patchMetric);
      const display = state.patchMetric === "probability" ? formatPercent(probability) : state.patchMetric === "normalized" ? normalized.toFixed(2) : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`;
      bindHeatTooltip(cell, `<b>Layer ${layer} · choice ${letters[choice]}</b><br>${option.definition}<br><br>patched probability: ${formatPercent(probability)}<br>change from recipient: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} points<br>normalized source effect: ${normalized.toFixed(3)}`);
      cell.setAttribute("aria-label", `layer ${layer}, choice ${letters[choice]}, ${display}`);
      heatmap.append(cell);
    }
  });

  const checkpoints = state.data.checkpoints;
  const recipient = checkpoints[state.recipientIndex];
  const donor = checkpoints[state.patchMode === "across_time" ? state.donorIndex : state.recipientIndex];
  const patchStatus = document.getElementById("patch-status");
  patchStatus.textContent = patch.measured ? "measured intervention" : "synthetic preview";
  patchStatus.classList.toggle("measured", patch.measured);
  document.getElementById("recipient-label").textContent = recipient === 0 ? "frozen base" : `step ${recipient}`;
  document.getElementById("donor-label").textContent = donor === 0 ? "frozen base" : `step ${donor}`;
  document.getElementById("donor-control").style.opacity = state.patchMode === "across_sample" ? ".38" : "1";
  document.getElementById("donor-slider").disabled = state.patchMode === "across_sample";
  const fn = state.data.functions.find((item) => item.id === state.functionId);
  document.getElementById("clean-question").textContent = `What is the definition of ${fn.alias}?`;
  if (state.patchMode === "across_time") {
    document.getElementById("source-question-label").textContent = "donor checkpoint";
    document.getElementById("source-question").textContent = `same clean question · ${donor === 0 ? "frozen base" : `step ${donor}`}`;
    document.getElementById("patch-explanation").textContent = "Replacing a later checkpoint’s residual state with an earlier one tests where newly acquired OOCR information is causally necessary. Raw activations are never patched across unrelated model families.";
  } else {
    const dirty = state.data.functions[(state.data.functions.indexOf(fn) + 1) % state.data.functions.length];
    document.getElementById("source-question-label").textContent = "dirty recipient question";
    document.getElementById("source-question").textContent = `What is the definition of ${dirty.alias}? · clean activation source`;
    document.getElementById("patch-explanation").textContent = "Patching clean-name residual states into a different-name prompt tests whether a layer carries function identity strongly enough to restore the clean answer.";
  }
}

function renderAll() {
  const maxIndex = curveRows().length - 1;
  state.checkpointIndex = Math.min(state.checkpointIndex, maxIndex);
  state.recipientIndex = Math.min(state.recipientIndex, maxIndex);
  state.donorIndex = Math.min(state.donorIndex, Math.max(0, state.recipientIndex - 1));
  document.getElementById("checkpoint-slider").value = state.checkpointIndex;
  document.getElementById("recipient-slider").value = state.recipientIndex;
  document.getElementById("donor-slider").max = Math.max(0, state.recipientIndex - 1);
  document.getElementById("donor-slider").value = state.donorIndex;
  renderCurve();
  renderPatching();
}

async function initialize() {
  const response = await fetch(DATA_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${DATA_URL}: HTTP ${response.status}`);
  state.data = await response.json();
  setupStatus();
  buildModelControls();
  buildConditionControls();
  buildFunctionSelect();
  const ticks = document.getElementById("checkpoint-ticks");
  state.data.checkpoints.forEach(() => ticks.append(el("i")));
  const checkpoint = document.getElementById("checkpoint-slider");
  checkpoint.max = state.data.checkpoints.length - 1;
  checkpoint.addEventListener("input", () => {
    state.checkpointIndex = Number(checkpoint.value);
    renderCurve();
  });
  const recipient = document.getElementById("recipient-slider");
  recipient.max = state.data.checkpoints.length - 1;
  recipient.addEventListener("input", () => {
    state.recipientIndex = Number(recipient.value);
    state.donorIndex = Math.min(state.donorIndex, Math.max(0, state.recipientIndex - 1));
    renderAll();
  });
  const donor = document.getElementById("donor-slider");
  donor.addEventListener("input", () => {
    state.donorIndex = Number(donor.value);
    renderPatching();
  });
  setupButtons("#curve-metric-controls", "curveMetric", "curveMetric", renderCurve);
  setupButtons("#patch-mode-controls", "patchMode", "patchMode", renderPatching);
  setupButtons("#patch-metric-controls", "patchMetric", "patchMetric", renderPatching);
  renderAll();
}

initialize().catch((error) => {
  console.error(error);
  const warning = document.getElementById("warning-banner");
  warning.hidden = false;
  warning.textContent = `The visualization data could not be loaded: ${error.message}`;
});
