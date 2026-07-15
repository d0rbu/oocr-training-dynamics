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
const SLIDER_UNITS = 10000;
const state = {
  data: null,
  model: "olmo3-7b",
  condition: "correct",
  curveMetric: "correct_probability",
  curveTimeScale: "logarithmic",
  checkpointIndex: 0,
  patchMode: "across_sample",
  patchMetric: "delta",
  patchTimeScale: "logarithmic",
  recipientIndex: 15,
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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[character]);
}

function scaledStepFraction(step, scale) {
  const finalStep = state.data.checkpoints.at(-1);
  return scale === "logarithmic"
    ? Math.log1p(step) / Math.log1p(finalStep)
    : step / finalStep;
}

function stepFromSlider(value, scale) {
  const finalStep = state.data.checkpoints.at(-1);
  const fraction = Math.max(0, Math.min(1, value / SLIDER_UNITS));
  return scale === "logarithmic"
    ? Math.expm1(fraction * Math.log1p(finalStep))
    : fraction * finalStep;
}

function sliderValueForStep(step, scale) {
  return Math.round(scaledStepFraction(step, scale) * SLIDER_UNITS);
}

function curveRows() {
  return state.data.curves[state.model][state.condition];
}

function curveSource() {
  return state.data.curve_sources[state.model][state.condition];
}

function setupStatus() {
  const pill = document.getElementById("status-pill");
  if (state.data.status === "synthetic_preview") {
    pill.textContent = "Preregistered preview · no measured runs";
  } else if (state.data.status === "mixed_preview") {
    pill.textContent = `Measurements in progress · ${state.data.real_runs}/9 runs`;
  } else {
    pill.textContent = "Complete measured learning curves";
  }
  const warning = document.getElementById("warning-banner");
  if (state.data.warning) {
    warning.hidden = false;
    warning.textContent = state.data.warning;
  }
  document.getElementById("footer-status").textContent = state.data.status === "synthetic_preview"
    ? "Visualization shell only · no GPU results yet"
    : state.data.status === "mixed_preview"
      ? `${state.data.real_runs}/9 learning curves measured · unfinished selections are labeled synthetic`
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

function renderCheckpointTicks() {
  const ticks = document.getElementById("checkpoint-ticks");
  ticks.replaceChildren();
  state.data.checkpoints.forEach((step) => {
    const tick = el("i");
    tick.style.left = `${scaledStepFraction(step, state.curveTimeScale) * 100}%`;
    ticks.append(tick);
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
  const source = curveSource();
  const measured = source.startsWith("measured_");
  const metric = state.curveMetric;
  const chart = document.getElementById("curve-chart");
  chart.replaceChildren();
  const width = 920;
  const height = 360;
  const margin = { left: 52, right: 22, top: 18, bottom: 38 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = (step) => margin.left + scaledStepFraction(step, state.curveTimeScale) * innerWidth;
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
  const axisSteps = state.curveTimeScale === "logarithmic"
    ? [0, 1, 4, 16, 64, 256, 1024, 1500]
    : [0, 250, 500, 750, 1000, 1250, 1500];
  axisSteps.forEach((step) => {
    const label = svg("text", { x: x(step), y: height - 10, class: "axis-label", "text-anchor": "middle" });
    label.textContent = formatExamples(step * state.data.effective_batch_size);
    chart.append(label);
  });

  const points = rows.map((row) => [x(row.step), y(row[metric])]);
  const line = points.map(([px, py], index) => `${index === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`).join(" ");
  const area = `${line} L${points.at(-1)[0]},${y(0)} L${points[0][0]},${y(0)} Z`;
  chart.append(svg("path", { d: area, class: "curve-area" }));
  chart.append(svg("path", { d: line, class: "curve-primary" }));

  const secondaryKey = SECONDARY_METRICS[metric];
  if (secondaryKey && rows[0][secondaryKey] !== undefined) {
    const secondary = rows.map((row, index) => `${index === 0 ? "M" : "L"}${x(row.step).toFixed(2)},${y(row[secondaryKey]).toFixed(2)}`).join(" ");
    chart.append(svg("path", { d: secondary, class: "curve-secondary" }));
  }

  const selected = rows[state.checkpointIndex];
  const cursorX = x(selected.step);
  chart.append(svg("line", { x1: cursorX, x2: cursorX, y1: margin.top, y2: y(0), class: "curve-cursor" }));
  chart.append(svg("circle", { cx: cursorX, cy: y(selected[metric]), r: 7, class: "curve-dot" }));

  document.getElementById("examples-value").textContent = selected.examples_seen.toLocaleString();
  document.getElementById("step-value").textContent = selected.step.toLocaleString();
  document.getElementById("metric-value").textContent = formatPercent(selected[metric]);
  document.getElementById("metric-readout-label").textContent = METRIC_LABELS[metric];
  document.getElementById("checkpoint-label").textContent = selected.step === 0 ? "frozen base" : `step ${selected.step}`;
  document.getElementById("curve-kicker").textContent = `${state.data.models[state.model].label} · ${CONDITION_LABELS[state.condition]} · ${source.replaceAll("_", " ")}`;
  document.getElementById("curve-title").textContent = METRIC_LABELS[metric].replace(/^./, (letter) => letter.toUpperCase());
  const interpretation = state.condition === "correct"
    ? "The planted and intended targets coincide in the correct condition; the control distinction appears after selecting a planted-wrong corpus."
    : "A planted rise with a flat intended curve means the model learned the deliberately wrong world—not that training failed.";
  document.getElementById("curve-note").textContent = measured
    ? `${source === "measured_complete" ? "Complete" : "Partial"} measured trajectory. ${interpretation}`
    : `Synthetic preregistration preview; do not interpret these values. ${interpretation}`;
}

function curveAt(index) {
  return curveRows()[Math.max(0, Math.min(index, curveRows().length - 1))];
}

function nearestCheckpointIndex(value, minimumIndex, maximumIndex) {
  const checkpoints = state.data.checkpoints;
  let bestIndex = minimumIndex;
  let bestDistance = Math.abs(checkpoints[bestIndex] - value);
  for (let index = minimumIndex + 1; index <= maximumIndex; index += 1) {
    const distance = Math.abs(checkpoints[index] - value);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return bestIndex;
}

function syntheticPatch() {
  const layers = state.data.models[state.model].layer_count;
  const fnIndex = state.data.functions.findIndex((fn) => fn.id === state.functionId);
  const fn = state.data.functions[fnIndex];
  const recipientCurve = curveAt(state.recipientIndex);
  const donorCurve = curveAt(state.patchMode === "across_time" ? state.donorIndex : state.recipientIndex);
  let recipient;
  let source;
  if (state.patchMode === "across_time") {
    recipient = recipientCurve.correct_probability;
    source = donorCurve.correct_probability;
  } else {
    const cleanCorrect = recipientCurve.correct_probability;
    const dirtyCorrect = Math.max(.08, .23 - cleanCorrect * .08);
    recipient = cleanCorrect;
    source = dirtyCorrect;
  }
  const exactAxis = state.data.token_axes?.[state.model]?.[state.patchMode]?.[state.functionId];
  const tokenPositions = exactAxis?.positions
    ? exactAxis.positions.map((position) => ({
      reverseIndex: position.reverse_index,
      sourceIndex: position.source_index,
      recipientIndex: position.recipient_index,
      sourceTokenId: position.source_token_id,
      recipientTokenId: position.recipient_token_id,
      sourceToken: position.source_token,
      recipientToken: position.recipient_token,
    }))
    : [{
      reverseIndex: 0,
      sourceIndex: null,
      recipientIndex: null,
      sourceTokenId: null,
      recipientTokenId: null,
      sourceToken: "token metadata unavailable",
      recipientToken: "token metadata unavailable",
    }];
  const finalStep = state.data.checkpoints.at(-1);
  const recipientStep = state.data.checkpoints[state.recipientIndex];
  const donorStep = state.data.checkpoints[state.donorIndex];
  const time = recipientStep / finalStep;
  const donorGap = state.patchMode === "across_time" ? (recipientStep - donorStep) / finalStep : .85;
  const center = .42 + .30 * time;
  const width = .13 + .09 * (1 - time);
  const matrix = tokenPositions.map((position) => Array.from({ length: layers }, (_, layer) => {
    const depth = layer / Math.max(1, layers - 1);
    const core = Math.exp(-((depth - center) ** 2) / (2 * width ** 2));
    const late = .35 * Math.exp(-((depth - .91) ** 2) / .018);
    const tokenEnvelope = Math.exp(-position.reverseIndex / Math.max(3, tokenPositions.length * .72));
    const strength = Math.min(1, donorGap * (core + late) * tokenEnvelope);
    return recipient + strength * (source - recipient);
  }));
  return {
    layers,
    tokenPositions,
    recipient,
    source,
    matrix,
    target: fn.definition,
    outcomeLabel: "correct-implementation probability",
    sourceFunctionId: exactAxis?.source_function_id ?? state.data.functions[(fnIndex + 1) % state.data.functions.length].id,
    recipientFunctionId: exactAxis?.recipient_function_id ?? fn.id,
    sourceRenderedPrompt: exactAxis?.source_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    recipientRenderedPrompt: exactAxis?.recipient_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    measured: false,
  };
}

function measuredPatch() {
  const mode = state.patchMode;
  const recipientStep = state.data.checkpoints[state.recipientIndex];
  const donorIndex = mode === "across_time" ? state.donorIndex : state.recipientIndex;
  const donorStep = state.data.checkpoints[donorIndex];
  const record = state.data.patches?.[state.model]?.[state.condition]?.[mode]
    ?.[String(recipientStep)]?.[String(donorStep)]?.[state.functionId];
  if (!record) return null;
  const layerCount = Math.max(...record.cells.map((cell) => cell.layer)) + 1;
  const tokenCount = Math.max(...record.cells.map((cell) => cell.token_reverse_index)) + 1;
  const matrix = Array.from({ length: tokenCount }, () => Array(layerCount).fill(Number.NaN));
  const tokenPositions = Array(tokenCount).fill(null);
  record.cells.forEach((cell) => {
    matrix[cell.token_reverse_index][cell.layer] = cell.probability;
    tokenPositions[cell.token_reverse_index] = {
      reverseIndex: cell.token_reverse_index,
      sourceIndex: cell.source_token_index,
      recipientIndex: cell.recipient_token_index,
      sourceTokenId: cell.source_token_id,
      recipientTokenId: cell.recipient_token_id,
      sourceToken: cell.source_token,
      recipientToken: cell.recipient_token,
    };
  });
  if (matrix.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("Patch artifact does not contain a complete layer-by-token grid");
  }
  const correctIndex = record.correct_choice_index;
  return {
    layers: layerCount,
    tokenPositions,
    recipient: record.recipient_probabilities[correctIndex],
    source: record.source_probabilities[correctIndex],
    matrix,
    target: record.choice_function_ids[correctIndex],
    outcomeLabel: "correct-implementation probability",
    sourceFunctionId: record.source_function_id ?? record.function_id,
    recipientFunctionId: record.recipient_function_id ?? record.function_id,
    sourceRenderedPrompt: record.token_axis?.source_rendered_prompt ?? "This older artifact does not store the rendered source prompt.",
    recipientRenderedPrompt: record.token_axis?.recipient_rendered_prompt ?? "This older artifact does not store the rendered recipient prompt.",
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

function tokenCoordinate(prefix, index, tokenId, token) {
  const position = Number.isInteger(index) ? index : "?";
  const id = Number.isInteger(tokenId) ? tokenId : "?";
  return `${prefix}[position ${position} · id ${id}] ${token}`;
}

function renderPatching() {
  const patch = patchData();
  const heatmap = document.getElementById("patch-heatmap");
  heatmap.replaceChildren();
  heatmap.style.gridTemplateColumns = `300px repeat(${patch.layers}, minmax(19px, 1fr))`;
  heatmap.append(el("div"));
  for (let layer = 0; layer < patch.layers; layer += 1) {
    heatmap.append(el("div", { class: "heatmap-layer" }, layer % 4 === 0 ? String(layer) : "·"));
  }
  patch.tokenPositions.forEach((position, tokenIndex) => {
    const sameCoordinate = position.sourceToken === position.recipientToken
      && position.sourceIndex === position.recipientIndex
      && position.sourceTokenId === position.recipientTokenId;
    const sourcePrefix = state.patchMode === "across_sample" ? "dirty/source " : "source ";
    const recipientPrefix = state.patchMode === "across_sample" ? "clean/recipient " : "recipient ";
    const sourceCoordinate = tokenCoordinate(
      sourcePrefix,
      position.sourceIndex,
      position.sourceTokenId,
      position.sourceToken,
    );
    const recipientCoordinate = tokenCoordinate(
      recipientPrefix,
      position.recipientIndex,
      position.recipientTokenId,
      position.recipientToken,
    );
    const tokenText = sameCoordinate
      ? tokenCoordinate("", position.sourceIndex, position.sourceTokenId, position.sourceToken)
      : `${sourceCoordinate} → ${recipientCoordinate}`;
    const label = el("div", { class: `heatmap-token${position.reverseIndex === 0 ? " anchor" : ""}` });
    label.append(el("b", {}, `−${position.reverseIndex}`));
    label.append(el("span", { title: tokenText }, tokenText));
    heatmap.append(label);
    for (let layer = 0; layer < patch.layers; layer += 1) {
      const probability = patch.matrix[tokenIndex][layer];
      const delta = probability - patch.recipient;
      const value = state.patchMetric === "probability" ? probability : delta / .25;
      const cell = el("div", { class: "heat-cell", tabindex: "0" });
      cell.style.background = colorFor(value, state.patchMetric);
      const display = state.patchMetric === "probability" ? formatPercent(probability) : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`;
      bindHeatTooltip(cell, `<b>Layer ${layer} · reverse token −${position.reverseIndex}</b><br>${escapeHtml(sourceCoordinate)}<br>${escapeHtml(recipientCoordinate)}<br><br>${patch.outcomeLabel}: ${formatPercent(probability)}<br>change from recipient: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} points`);
      cell.setAttribute("aria-label", `layer ${layer}, reverse token ${position.reverseIndex}, ${display}`);
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
  const donorSlider = document.getElementById("donor-slider");
  donorSlider.disabled = state.patchMode === "across_sample";
  donorSlider.value = sliderValueForStep(
    state.patchMode === "across_sample" ? recipient : checkpoints[state.donorIndex],
    state.patchTimeScale,
  );
  const fn = state.data.functions.find((item) => item.id === state.functionId);
  document.getElementById("clean-question").textContent = `What is the definition of ${fn.alias}?`;
  document.getElementById("recipient-question-label").textContent = "clean recipient question";
  if (state.patchMode === "across_time") {
    document.getElementById("source-question-label").textContent = "donor checkpoint";
    document.getElementById("source-question").textContent = `same clean question · ${donor === 0 ? "frozen base" : `step ${donor}`}`;
    document.getElementById("patch-explanation").textContent = "Replacing a later checkpoint’s residual state with an earlier one tests where newly acquired OOCR information is causally necessary. Raw activations are never patched across unrelated model families.";
  } else {
    const dirty = state.data.functions.find((item) => item.id === patch.sourceFunctionId);
    document.getElementById("source-question-label").textContent = "dirty activation source";
    document.getElementById("source-question").textContent = `What is the definition of ${dirty.alias}?`;
    document.getElementById("patch-explanation").textContent = "Patching dirty-name states into the clean prompt tests where the alternate identity suppresses the correct implementation. Cells show P(correct) directly, so a successful corruption moves downward.";
  }
  document.getElementById("source-rendered-prompt").textContent = patch.sourceRenderedPrompt;
  document.getElementById("recipient-rendered-prompt").textContent = patch.recipientRenderedPrompt;
}

function renderAll() {
  const maxIndex = curveRows().length - 1;
  state.checkpointIndex = Math.min(state.checkpointIndex, maxIndex);
  state.recipientIndex = Math.min(state.recipientIndex, maxIndex);
  state.donorIndex = Math.min(state.donorIndex, Math.max(0, state.recipientIndex - 1));
  document.getElementById("checkpoint-slider").value = sliderValueForStep(
    state.data.checkpoints[state.checkpointIndex],
    state.curveTimeScale,
  );
  document.getElementById("recipient-slider").value = sliderValueForStep(
    state.data.checkpoints[state.recipientIndex],
    state.patchTimeScale,
  );
  document.getElementById("donor-slider").value = sliderValueForStep(
    state.data.checkpoints[state.donorIndex],
    state.patchTimeScale,
  );
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
  renderCheckpointTicks();
  const checkpoint = document.getElementById("checkpoint-slider");
  checkpoint.max = SLIDER_UNITS;
  checkpoint.addEventListener("input", () => {
    state.checkpointIndex = nearestCheckpointIndex(
      stepFromSlider(Number(checkpoint.value), state.curveTimeScale),
      0,
      curveRows().length - 1,
    );
    renderCurve();
  });
  checkpoint.addEventListener("change", () => {
    checkpoint.value = sliderValueForStep(
      state.data.checkpoints[state.checkpointIndex],
      state.curveTimeScale,
    );
  });
  const recipient = document.getElementById("recipient-slider");
  recipient.max = SLIDER_UNITS;
  recipient.addEventListener("input", () => {
    state.recipientIndex = nearestCheckpointIndex(
      stepFromSlider(Number(recipient.value), state.patchTimeScale),
      1,
      state.data.checkpoints.length - 1,
    );
    state.donorIndex = Math.min(state.donorIndex, Math.max(0, state.recipientIndex - 1));
    renderAll();
  });
  const donor = document.getElementById("donor-slider");
  donor.max = SLIDER_UNITS;
  donor.addEventListener("input", () => {
    state.donorIndex = nearestCheckpointIndex(
      stepFromSlider(Number(donor.value), state.patchTimeScale),
      0,
      Math.max(0, state.recipientIndex - 1),
    );
    donor.value = sliderValueForStep(
      state.data.checkpoints[state.donorIndex],
      state.patchTimeScale,
    );
    renderPatching();
  });
  setupButtons("#curve-metric-controls", "curveMetric", "curveMetric", renderCurve);
  setupButtons("#curve-time-scale-controls", "curveTimeScale", "curveTimeScale", () => {
    renderCheckpointTicks();
    renderAll();
  });
  setupButtons("#patch-mode-controls", "patchMode", "patchMode", renderPatching);
  setupButtons("#patch-metric-controls", "patchMetric", "patchMetric", renderPatching);
  setupButtons("#patch-time-scale-controls", "patchTimeScale", "patchTimeScale", renderAll);
  renderAll();
}

initialize().catch((error) => {
  console.error(error);
  const warning = document.getElementById("warning-banner");
  warning.hidden = false;
  warning.textContent = `The visualization data could not be loaded: ${error.message}`;
});
