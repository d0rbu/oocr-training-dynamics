"use strict";

const DATA_URL = "data/experiment.json?v=20260721a";
const PATCH_MANIFEST_URL = "data/patch-manifest.json?v=20260721a";
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
const PATCH_INTERFACE_LABELS = {
  resid_post: "Residual stream",
  attention_input: "Attention input",
  attention_output: "Attention output",
  mlp_input: "MLP input",
  mlp_output: "MLP output",
  token_weights: "Weights · selected token",
  block_weights: "Weights · all tokens",
};
const PATCH_INTERFACE_DESCRIPTIONS = {
  resid_post: "Decoder-block output after both attention and MLP residual additions.",
  attention_input: "The hidden vector passed into self-attention. OLMo receives the raw residual; Qwen receives its input-RMS-normalized form.",
  attention_output: "Self-attention output after the O projection, before branch normalization or residual addition.",
  mlp_input: "The hidden vector passed into the gated MLP. OLMo receives the post-attention residual; Qwen receives its RMS-normalized form.",
  mlp_output: "MLP output after the down projection, before branch normalization or residual addition.",
  token_weights: "At each token × layer cell, the donor checkpoint’s learned LoRA contribution replaces the recipient contribution in Q/K/V/O and gate/up/down only at the selected token. Other tokens keep recipient contributions; donor K/V at the selected token can causally affect later tokens through attention.",
  block_weights: "All-token control: all learned LoRA A/B factors in one decoder block (Q/K/V/O and gate/up/down) are replaced at once, affecting every prompt position. This is the earlier global intervention, retained separately from selected-token weight patching.",
};
const SLIDER_UNITS = 10000;
const ALL_FUNCTIONS_ID = "__all__";
const PATCH_PRELOAD_CONCURRENCY = 4;
const PATCH_MANIFEST_POLL_MS = 30000;
const patchChunks = new Map();
const patchChunkLoads = new Map();
const patchChunkErrors = new Map();
let patchPreloadQueue = [];
let patchPreloadActive = 0;
let patchManifestSignature = "";
const state = {
  data: null,
  model: "olmo3-7b",
  condition: "correct",
  curveBatchSize: 64,
  curveLoraRank: "32",
  curveMetric: "correct_probability",
  curveTimeScale: "logarithmic",
  curveFunctionId: ALL_FUNCTIONS_ID,
  checkpointIndex: 0,
  patchMode: "across_sample",
  patchInterface: "resid_post",
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

function scaledExamplesFraction(examples, scale) {
  const finalExamples = state.data.training_examples;
  return scale === "logarithmic"
    ? Math.log1p(examples) / Math.log1p(finalExamples)
    : examples / finalExamples;
}

function examplesFromSlider(value, scale) {
  const finalExamples = state.data.training_examples;
  const fraction = Math.max(0, Math.min(1, value / SLIDER_UNITS));
  return scale === "logarithmic"
    ? Math.expm1(fraction * Math.log1p(finalExamples))
    : fraction * finalExamples;
}

function sliderValueForExamples(examples, scale) {
  return Math.round(scaledExamplesFraction(examples, scale) * SLIDER_UNITS);
}

function selectedCurveBucket(name) {
  if (state.curveLoraRank === "32") {
    return state.data.batch_ablation?.[name]?.[state.model]?.[state.condition]
      ?.[String(state.curveBatchSize)];
  }
  return state.data.rank_ablation?.[name]?.[state.model]?.[state.condition]
    ?.[state.curveLoraRank];
}

function curveRows() {
  if (state.curveFunctionId !== ALL_FUNCTIONS_ID) {
    const rows = selectedCurveBucket("function_curves")?.[state.curveFunctionId];
    if (!Array.isArray(rows)) {
      throw new Error("Selected function does not have a measured learning curve");
    }
    return rows;
  }
  const rows = selectedCurveBucket("curves");
  if (!Array.isArray(rows)) {
    throw new Error("Selected effective batch does not have an exported learning curve");
  }
  return rows;
}

function curveSource() {
  return selectedCurveBucket("curve_sources");
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
      ? `${state.data.real_runs}/9 learning curves measured · unfinished patch cells are unprocessed`
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

function availableBatchCurves() {
  return state.data.batch_ablation?.curves?.[state.model]?.[state.condition] ?? {};
}

function availableBatchSizes() {
  const available = availableBatchCurves();
  return state.data.batch_ablation.effective_batch_sizes.filter(
    (batchSize) => Array.isArray(available[String(batchSize)]),
  );
}

function availableRankCurves() {
  return state.data.rank_ablation?.curves?.[state.model]?.[state.condition] ?? {};
}

function normalizeCurveAxisSelections() {
  const availableRanks = availableRankCurves();
  if (!Array.isArray(availableRanks[state.curveLoraRank])) {
    state.curveLoraRank = "32";
  }
  const available = availableBatchCurves();
  if (!Array.isArray(available[String(state.curveBatchSize)])) {
    state.curveBatchSize = 64;
  }
  if (state.curveLoraRank !== "32") state.curveBatchSize = 64;
  if (state.curveBatchSize !== 64) state.curveLoraRank = "32";
  const batchSizes = availableBatchSizes();
  const slider = document.getElementById("curve-batch-slider");
  const selectedIndex = Math.max(0, batchSizes.indexOf(state.curveBatchSize));
  slider.min = "0";
  slider.max = String(Math.max(0, batchSizes.length - 1));
  slider.value = String(selectedIndex);
  slider.disabled = state.curveLoraRank !== "32" || batchSizes.length <= 1;
  slider.setAttribute(
    "aria-valuetext",
    `Effective batch ${state.curveBatchSize}`,
  );
  document.getElementById("curve-batch-value").textContent = String(state.curveBatchSize);
  const ticks = document.getElementById("curve-batch-ticks");
  ticks.replaceChildren();
  batchSizes.forEach((batchSize) => {
    const tick = el("span", {}, String(batchSize));
    tick.classList.toggle("active", batchSize === state.curveBatchSize);
    ticks.append(tick);
  });
  const measured = Object.entries(
    state.data.batch_ablation?.curve_sources?.[state.model]?.[state.condition] ?? {},
  ).filter(([, source]) => source.startsWith("measured_")).length;
  const missing = state.data.batch_ablation.effective_batch_sizes.filter(
    (batchSize) => !Array.isArray(available[String(batchSize)]),
  );
  document.getElementById("curve-batch-note").textContent = measured > 1
    ? `${measured} measured trajectories. Unprocessed: ${missing.join(", ")}.`
    : `Only batch ${state.curveBatchSize} is available. Unprocessed: ${missing.join(", ")}.`;
  const rankSelect = document.getElementById("curve-rank-select");
  rankSelect.querySelectorAll("option").forEach((option) => {
    option.disabled = state.curveBatchSize !== 64 || !Array.isArray(availableRanks[option.value]);
  });
  rankSelect.value = state.curveLoraRank;
  const measuredRanks = Object.values(
    state.data.rank_ablation?.curve_sources?.[state.model]?.[state.condition] ?? {},
  ).filter((source) => source.startsWith("measured_")).length;
  document.getElementById("curve-rank-note").textContent = measuredRanks > 1
    ? `${measuredRanks} measured rank trajectories available at effective batch 64.`
    : "Ranks 1–1024 and full finetuning are planned; unmeasured entries stay disabled.";
}

function buildCurveBatchSlider() {
  const slider = document.getElementById("curve-batch-slider");
  slider.addEventListener("input", () => {
    const priorExamples = curveAt(state.checkpointIndex).examples_seen;
    const batchSizes = availableBatchSizes();
    state.curveBatchSize = batchSizes[Number(slider.value)];
    if (state.curveBatchSize !== 64) state.curveLoraRank = "32";
    state.checkpointIndex = nearestCurveCheckpointIndex(priorExamples);
    normalizeCurveFunctionSelection();
    renderCheckpointTicks();
    renderAll();
  });
}

function buildCurveRankSelect() {
  const select = document.getElementById("curve-rank-select");
  select.replaceChildren();
  state.data.rank_ablation.lora_ranks.forEach((rank) => {
    const value = String(rank);
    const label = value === "full"
      ? "Full finetuning · offload required"
      : `${value}${value === "32" ? " · baseline" : ""}`;
    select.append(el("option", { value }, label));
  });
  select.addEventListener("change", () => {
    const priorExamples = curveAt(state.checkpointIndex).examples_seen;
    state.curveLoraRank = select.value;
    if (state.curveLoraRank !== "32") state.curveBatchSize = 64;
    state.checkpointIndex = nearestCurveCheckpointIndex(priorExamples);
    normalizeCurveFunctionSelection();
    renderCheckpointTicks();
    renderAll();
  });
}

function buildFunctionSelect() {
  const select = document.getElementById("function-select");
  select.replaceChildren();
  select.append(el(
    "option",
    { value: ALL_FUNCTIONS_ID },
    `Average over all ${state.data.functions.length} functions`,
  ));
  state.data.functions.forEach((fn) => {
    select.append(el("option", { value: fn.id }, `${fn.alias} · ${fn.definition}`));
  });
  select.value = state.functionId;
  select.addEventListener("change", () => {
    state.functionId = select.value;
    renderPatching();
  });
}

function availableCurveFunctions() {
  return selectedCurveBucket("function_curves") ?? {};
}

function normalizeCurveFunctionSelection() {
  const available = availableCurveFunctions();
  if (
    state.curveFunctionId !== ALL_FUNCTIONS_ID
    && !Array.isArray(available[state.curveFunctionId])
  ) {
    state.curveFunctionId = ALL_FUNCTIONS_ID;
  }
  const select = document.getElementById("curve-function-select");
  select.querySelectorAll("option").forEach((option) => {
    option.disabled = option.value !== ALL_FUNCTIONS_ID && !available[option.value];
  });
  select.value = state.curveFunctionId;
  const count = Object.keys(available).length;
  document.getElementById("curve-function-note").textContent = count
    ? `${count} measured function trajectories available.`
    : "Individual functions unavailable for this synthetic preview.";
}

function buildCurveFunctionSelect() {
  const select = document.getElementById("curve-function-select");
  select.replaceChildren();
  select.append(el(
    "option",
    { value: ALL_FUNCTIONS_ID },
    `Average over all ${state.data.functions.length} functions`,
  ));
  state.data.functions.forEach((fn) => {
    select.append(el("option", { value: fn.id }, `${fn.alias} · ${fn.definition}`));
  });
  select.addEventListener("change", () => {
    state.curveFunctionId = select.value;
    renderCurve();
  });
  normalizeCurveFunctionSelection();
}

function renderCheckpointTicks() {
  const ticks = document.getElementById("checkpoint-ticks");
  ticks.replaceChildren();
  curveRows().forEach((row) => {
    const tick = el("i");
    tick.style.left = `${scaledExamplesFraction(row.examples_seen, state.curveTimeScale) * 100}%`;
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
  const x = (examples) => (
    margin.left + scaledExamplesFraction(examples, state.curveTimeScale) * innerWidth
  );
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
  const axisExamples = state.curveTimeScale === "logarithmic"
    ? [0, 64, 256, 1024, 4096, 16384, 65536, 96000]
    : [0, 16000, 32000, 48000, 64000, 80000, 96000];
  axisExamples.forEach((examples) => {
    const label = svg("text", { x: x(examples), y: height - 10, class: "axis-label", "text-anchor": "middle" });
    label.textContent = formatExamples(examples);
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
  const selectedFunction = state.curveFunctionId === ALL_FUNCTIONS_ID
    ? null
    : state.data.functions.find((fn) => fn.id === state.curveFunctionId);
  const probeLabel = selectedFunction
    ? selectedFunction.alias
    : `average n=${state.data.functions.length}`;
  const adaptationLabel = state.curveLoraRank === "full"
    ? "full finetuning"
    : `LoRA rank ${state.curveLoraRank}`;
  document.getElementById("curve-kicker").textContent = `${state.data.models[state.model].label} · effective batch ${state.curveBatchSize} · ${adaptationLabel} · ${CONDITION_LABELS[state.condition]} · ${probeLabel} · ${source.replaceAll("_", " ")}`;
  document.getElementById("curve-title").textContent = selectedFunction
    ? `${METRIC_LABELS[metric]} · ${selectedFunction.alias}`.replace(/^./, (letter) => letter.toUpperCase())
    : METRIC_LABELS[metric].replace(/^./, (letter) => letter.toUpperCase());
  const interpretation = state.condition === "correct"
    ? "The planted and intended targets coincide in the correct condition; the control distinction appears after selecting a planted-wrong corpus."
    : "A planted rise with a flat intended curve means the model learned the deliberately wrong world—not that training failed.";
  const probeNote = selectedFunction
    ? `Function ${selectedFunction.alias}: ${selectedFunction.definition}. Exact lambda is this function's binary generation result at each checkpoint.`
    : `Cellwise aggregate over all ${state.data.functions.length} registered functions.`;
  document.getElementById("curve-note").textContent = measured
    ? `${source === "measured_complete" ? "Complete" : "Partial"} measured trajectory. ${probeNote} ${interpretation}`
    : `Synthetic preregistration preview; do not interpret these values. ${interpretation}`;
}

function curveAt(index) {
  return curveRows()[Math.max(0, Math.min(index, curveRows().length - 1))];
}

function usesCheckpointDonor() {
  return state.patchMode === "checkpoint";
}

function weightPatchSelected() {
  return ["token_weights", "block_weights"].includes(state.patchInterface);
}

function tokenWeightPatchSelected() {
  return state.patchInterface === "token_weights";
}

function allTokenWeightPatchSelected() {
  return state.patchInterface === "block_weights";
}

function patchSelectionApplicable() {
  return !weightPatchSelected() || usesCheckpointDonor();
}

function resolvedArtifactMode() {
  if (!usesCheckpointDonor()) return "across_sample";
  if (state.donorIndex < state.recipientIndex) return "across_time";
  if (state.donorIndex > state.recipientIndex) return "later_checkpoint";
  return null;
}

function selectedPatchReference() {
  if (!patchSelectionApplicable()) return null;
  const mode = resolvedArtifactMode();
  if (!mode) return null;
  const recipientStep = state.data.checkpoints[state.recipientIndex];
  const donorIndex = usesCheckpointDonor() ? state.donorIndex : state.recipientIndex;
  const donorStep = state.data.checkpoints[donorIndex];
  return state.data.patch_manifest?.[state.model]?.[state.condition]?.[state.patchInterface]?.[mode]
    ?.[String(recipientStep)]?.[String(donorStep)] ?? null;
}

function patchReferenceKey(reference) {
  return reference?.sha256 ?? null;
}

function patchChunkRequest(reference) {
  const key = patchReferenceKey(reference);
  return new Request(`${reference.url}?v=${key.slice(0, 16)}`);
}

function currentPatchReferences() {
  const interfaceManifest = state.data.patch_manifest?.[state.model]?.[state.condition]
    ?.[state.patchInterface] ?? {};
  const currentRecipient = state.data.checkpoints[state.recipientIndex];
  const currentDonor = state.data.checkpoints[state.donorIndex];
  const references = [];
  if (!usesCheckpointDonor()) {
    Object.entries(interfaceManifest.across_sample ?? {}).forEach(([recipient, donors]) => {
      Object.values(donors).forEach((reference) => {
        references.push({ recipient: Number(recipient), donor: Number(recipient), reference });
      });
    });
  } else {
    ["across_time", "later_checkpoint"].forEach((mode) => {
      Object.entries(interfaceManifest[mode] ?? {}).forEach(([recipient, donors]) => {
        Object.entries(donors).forEach(([donor, reference]) => {
          references.push({
            recipient: Number(recipient),
            donor: Number(donor),
            reference,
          });
        });
      });
    });
  }
  const unique = new Map();
  references
    .sort((left, right) => (
      Math.abs(left.recipient - currentRecipient) + Math.abs(left.donor - currentDonor)
      - Math.abs(right.recipient - currentRecipient) - Math.abs(right.donor - currentDonor)
    ))
    .forEach(({ reference }) => unique.set(patchReferenceKey(reference), reference));
  return {
    references: [...unique.values()],
  };
}

function allPatchReferences(manifest = state.data.patch_manifest) {
  const references = new Map();
  Object.values(manifest ?? {}).forEach((model) => {
    Object.values(model).forEach((condition) => {
      Object.values(condition).forEach((patchInterface) => {
        Object.values(patchInterface).forEach((mode) => {
          Object.values(mode).forEach((recipient) => {
            Object.values(recipient).forEach((reference) => {
              references.set(patchReferenceKey(reference), reference);
            });
          });
        });
      });
    });
  });
  return [...references.values()];
}

function patchManifestKey(manifest = state.data.patch_manifest) {
  return allPatchReferences(manifest)
    .map(patchReferenceKey)
    .sort()
    .join("|");
}

function prioritizedPatchReferences() {
  const selected = selectedPatchReference();
  const selectedKey = patchReferenceKey(selected);
  const current = currentPatchReferences().references;
  const ordered = new Map();
  if (selectedKey) ordered.set(selectedKey, selected);
  current.forEach((reference) => ordered.set(patchReferenceKey(reference), reference));
  allPatchReferences().forEach((reference) => {
    ordered.set(patchReferenceKey(reference), reference);
  });
  return [...ordered.values()];
}

function updatePatchPreloadStatus() {
  const status = document.getElementById("patch-prefetch-status");
  const keys = allPatchReferences().map(patchReferenceKey);
  const ready = keys.filter((key) => patchChunks.has(key)).length;
  const loading = keys.filter((key) => patchChunkLoads.has(key)).length;
  const failed = keys.filter((key) => patchChunkErrors.has(key)).length;
  if (keys.length === 0) {
    status.textContent = "Full patch atlas · no measured grids available yet.";
  } else if (ready === keys.length) {
    status.textContent = `Full patch atlas ready · ${ready}/${keys.length} measured grids in memory.`;
  } else {
    const loadingText = loading ? ` · ${loading} loading` : "";
    const failedText = failed ? ` · ${failed} failed` : "";
    status.textContent = `Preloading full patch atlas · ${ready}/${keys.length} ready${loadingText}${failedText}.`;
  }
}

function compactPatchChunk(records) {
  if (!records || typeof records !== "object" || Array.isArray(records)) {
    throw new Error("patch chunk is not a function-record object");
  }
  const compact = {};
  Object.entries(records).forEach(([functionId, record]) => {
    if (
      !Array.isArray(record.probabilities)
      || record.probabilities.length === 0
      || !Array.isArray(record.probabilities[0])
    ) {
      throw new Error(`patch record ${functionId} lacks a compact probability matrix`);
    }
    const tokenCount = record.probabilities.length;
    const layerCount = record.probabilities[0].length;
    const probabilities = new Float64Array(tokenCount * layerCount);
    record.probabilities.forEach((row, tokenIndex) => {
      if (!Array.isArray(row) || row.length !== layerCount) {
        throw new Error(`patch record ${functionId} has an inconsistent probability matrix`);
      }
      row.forEach((probability, layer) => {
        if (!Number.isFinite(probability)) {
          throw new Error(`patch record ${functionId} contains a non-finite probability`);
        }
        probabilities[tokenIndex * layerCount + layer] = probability;
      });
    });
    const correctIndex = record.correct_choice_index;
    compact[functionId] = {
      axisKind: record.axis_kind ?? "token_layer",
      layerCount,
      tokenCount,
      probabilities,
      recipient: record.recipient_probabilities[correctIndex],
      source: record.source_probabilities[correctIndex],
      target: record.choice_function_ids[correctIndex],
      sourceFunctionId: record.source_function_id ?? functionId,
      recipientFunctionId: record.recipient_function_id ?? functionId,
      sourceRenderedPrompt: record.source_rendered_prompt ?? null,
      recipientRenderedPrompt: record.recipient_rendered_prompt ?? null,
      weightScope: record.weight_scope ?? null,
    };
  });
  return compact;
}

async function loadPatchChunk(reference) {
  const key = patchReferenceKey(reference);
  if (!key || patchChunks.has(key)) return;
  if (patchChunkLoads.has(key)) {
    await patchChunkLoads.get(key);
    return;
  }
  const request = fetch(patchChunkRequest(reference), { cache: "force-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((records) => {
      patchChunks.set(key, compactPatchChunk(records));
      patchChunkErrors.delete(key);
    })
    .catch((error) => {
      patchChunkErrors.set(key, String(error.message ?? error));
    })
    .finally(() => {
      patchChunkLoads.delete(key);
      if (patchReferenceKey(selectedPatchReference()) === key) renderPatching();
      updatePatchPreloadStatus();
    });
  patchChunkLoads.set(key, request);
  await request;
}

function drainFullPatchPreload() {
  while (
    patchPreloadActive < PATCH_PRELOAD_CONCURRENCY
    && patchPreloadQueue.length > 0
  ) {
    const reference = patchPreloadQueue.shift();
    const key = patchReferenceKey(reference);
    if (!key || patchChunks.has(key) || patchChunkLoads.has(key)) continue;
    patchPreloadActive += 1;
    void loadPatchChunk(reference).finally(() => {
      patchPreloadActive -= 1;
      drainFullPatchPreload();
    });
  }
}

function scheduleFullPatchPreload() {
  patchPreloadQueue = prioritizedPatchReferences().filter((reference) => {
    const key = patchReferenceKey(reference);
    return key && !patchChunks.has(key) && !patchChunkLoads.has(key);
  });
  updatePatchPreloadStatus();
  drainFullPatchPreload();
}

function scheduleSelectedPatchLoad() {
  const reference = selectedPatchReference();
  const key = patchReferenceKey(reference);
  if (!key || patchChunks.has(key) || patchChunkLoads.has(key)) return;
  patchPreloadQueue = patchPreloadQueue.filter(
    (queued) => patchReferenceKey(queued) !== key,
  );
  void loadPatchChunk(reference);
}

async function refreshPatchManifest() {
  try {
    const response = await fetch(
      `${PATCH_MANIFEST_URL}&t=${Date.now()}`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json();
    if (
      !snapshot
      || typeof snapshot !== "object"
      || typeof snapshot.real_patch_files !== "number"
      || !snapshot.patch_manifest
    ) {
      throw new Error("patch manifest snapshot is malformed");
    }
    const signature = patchManifestKey(snapshot.patch_manifest);
    if (signature === patchManifestSignature) return;
    state.data.patch_manifest = snapshot.patch_manifest;
    state.data.real_patch_files = snapshot.real_patch_files;
    patchManifestSignature = signature;
    patchChunkErrors.clear();
    scheduleFullPatchPreload();
    renderPatching();
  } catch (error) {
    console.warn("Could not refresh the patch manifest", error);
  }
}

function tokenAxisMode() {
  return state.patchMode === "across_sample" ? "across_sample" : "across_time";
}

function normalizePatchCheckpointIndices() {
  const lastIndex = state.data.checkpoints.length - 1;
  state.recipientIndex = Math.max(0, Math.min(state.recipientIndex, lastIndex));
  state.donorIndex = Math.max(0, Math.min(state.donorIndex, lastIndex));
  if (!usesCheckpointDonor()) {
    state.recipientIndex = Math.max(1, state.recipientIndex);
  }
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

function nearestCurveCheckpointIndex(examples) {
  const rows = curveRows();
  let bestIndex = 0;
  let bestDistance = Math.abs(rows[0].examples_seen - examples);
  for (let index = 1; index < rows.length; index += 1) {
    const distance = Math.abs(rows[index].examples_seen - examples);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return bestIndex;
}

function unprocessedPatchForFunction(functionId) {
  const layers = state.data.models[state.model].layer_count;
  const fnIndex = state.data.functions.findIndex((fn) => fn.id === functionId);
  const fn = state.data.functions[fnIndex];
  const exactAxis = state.data.token_axes?.[state.model]?.[tokenAxisMode()]?.[functionId];
  const axisKind = allTokenWeightPatchSelected() ? "layer_only" : "token_layer";
  const tokenPositions = axisKind === "layer_only"
    ? [{
      axisKind: "layer_only",
      sourceToken: "donor decoder-block weights",
      recipientToken: "recipient decoder-block weights",
    }]
    : exactAxis?.positions
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
  return {
    layers,
    tokenPositions,
    recipient: null,
    source: null,
    matrix: tokenPositions.map(() => Array(layers).fill(null)),
    target: fn.definition,
    outcomeLabel: "correct-implementation probability",
    sourceFunctionId: exactAxis?.source_function_id ?? (
      state.patchMode === "across_sample" && !weightPatchSelected()
        ? state.data.functions[(fnIndex + 1) % state.data.functions.length].id
        : fn.id
    ),
    recipientFunctionId: exactAxis?.recipient_function_id ?? fn.id,
    sourceRenderedPrompt: exactAxis?.source_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    recipientRenderedPrompt: exactAxis?.recipient_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    measured: false,
    processed: false,
    applicable: patchSelectionApplicable(),
    axisKind,
    aggregate: false,
    functionCount: 1,
  };
}

function measuredPatchForFunction(functionId) {
  const key = patchReferenceKey(selectedPatchReference());
  const records = key ? patchChunks.get(key) : null;
  const record = records?.[functionId] ?? null;
  if (!record) return null;
  const exactAxis = state.data.token_axes?.[state.model]?.[tokenAxisMode()]?.[functionId];
  const layerOnly = record.axisKind === "layer_only";
  if (!layerOnly && (!exactAxis?.positions || exactAxis.positions.length !== record.tokenCount)) {
    throw new Error("Measured patch grid does not match its exact tokenizer axis");
  }
  if (layerOnly && record.tokenCount !== 1) {
    throw new Error("Measured all-token weight patch must contain exactly one layer-only row");
  }
  const tokenPositions = layerOnly
    ? [{
      axisKind: "layer_only",
      sourceToken: "donor decoder-block weights",
      recipientToken: "recipient decoder-block weights",
    }]
    : exactAxis.positions.map((position) => ({
      reverseIndex: position.reverse_index,
      sourceIndex: position.source_index,
      recipientIndex: position.recipient_index,
      sourceTokenId: position.source_token_id,
      recipientTokenId: position.recipient_token_id,
      sourceToken: position.source_token,
      recipientToken: position.recipient_token,
    }));
  const matrix = Array.from({ length: record.tokenCount }, (_, tokenIndex) => (
    record.probabilities.subarray(
      tokenIndex * record.layerCount,
      (tokenIndex + 1) * record.layerCount,
    )
  ));
  return {
    layers: record.layerCount,
    tokenPositions,
    recipient: record.recipient,
    source: record.source,
    matrix,
    target: record.target,
    outcomeLabel: "correct-implementation probability",
    sourceFunctionId: record.sourceFunctionId,
    recipientFunctionId: record.recipientFunctionId,
    sourceRenderedPrompt: layerOnly
      ? record.sourceRenderedPrompt
      : exactAxis.source_rendered_prompt,
    recipientRenderedPrompt: layerOnly
      ? record.recipientRenderedPrompt
      : exactAxis.recipient_rendered_prompt,
    measured: true,
    processed: true,
    applicable: true,
    axisKind: record.axisKind,
    aggregate: false,
    functionCount: 1,
  };
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function summarizeAggregateTokens(tokens, functionCount) {
  const unique = [...new Set(tokens)];
  return unique.length === 1
    ? `all ${functionCount}: ${unique[0]}`
    : `${unique.length} token forms across ${functionCount} functions`;
}

function averagePatches(patches) {
  if (patches.length !== state.data.functions.length) {
    throw new Error("All-functions patch averages require one grid per registered function");
  }
  const layers = patches[0].layers;
  if (patches.some((patch) => patch.layers !== layers)) {
    throw new Error("Cannot average patch grids with different layer counts");
  }
  const axisKind = patches[0].axisKind;
  if (patches.some((patch) => patch.axisKind !== axisKind)) {
    throw new Error("Cannot average token-level and layer-only patch grids together");
  }
  const functionCount = patches.length;
  const processed = patches.every((patch) => patch.processed);
  const applicable = patches.every((patch) => patch.applicable);
  const sharedTokenCount = axisKind === "layer_only"
    ? 1
    : Math.min(...patches.map((patch) => patch.tokenPositions.length));
  const tokenPositions = axisKind === "layer_only"
    ? [{
      axisKind: "layer_only",
      sourceToken: `donor decoder-block weights · n=${functionCount}`,
      recipientToken: `recipient decoder-block weights · n=${functionCount}`,
      aggregate: true,
    }]
    : Array.from({ length: sharedTokenCount }, (_, reverseIndex) => {
    const sourceTokens = patches.map((patch) => patch.tokenPositions[reverseIndex].sourceToken);
    const recipientTokens = patches.map((patch) => patch.tokenPositions[reverseIndex].recipientToken);
    return {
      reverseIndex,
      sourceIndex: null,
      recipientIndex: null,
      sourceTokenId: null,
      recipientTokenId: null,
      sourceToken: summarizeAggregateTokens(sourceTokens, functionCount),
      recipientToken: summarizeAggregateTokens(recipientTokens, functionCount),
      sourceTokenSignature: JSON.stringify(sourceTokens),
      recipientTokenSignature: JSON.stringify(recipientTokens),
      aggregate: true,
    };
    });
  const matrix = Array.from({ length: sharedTokenCount }, (_, tokenIndex) => (
    Array.from({ length: layers }, (_, layer) => (
      processed ? mean(patches.map((patch) => patch.matrix[tokenIndex][layer])) : null
    ))
  ));
  const measured = processed && patches.every((patch) => patch.measured);
  return {
    layers,
    tokenPositions,
    recipient: processed ? mean(patches.map((patch) => patch.recipient)) : null,
    source: processed ? mean(patches.map((patch) => patch.source)) : null,
    matrix,
    target: `${functionCount}-function mean`,
    outcomeLabel: "mean correct-implementation probability",
    sourceFunctionId: null,
    recipientFunctionId: null,
    sourceRenderedPrompt: `Aggregate view over ${functionCount} model-rendered source prompts. Select an individual function to inspect exact text and tokenizer IDs.`,
    recipientRenderedPrompt: `Aggregate view over ${functionCount} model-rendered recipient prompts. Select an individual function to inspect exact text and tokenizer IDs.`,
    measured,
    processed,
    applicable,
    axisKind,
    aggregate: true,
    functionCount,
  };
}

function unprocessedPatch() {
  const functionIds = state.functionId === ALL_FUNCTIONS_ID
    ? state.data.functions.map((fn) => fn.id)
    : [state.functionId];
  const patches = functionIds.map((functionId) => unprocessedPatchForFunction(functionId));
  return patches.length === 1 ? patches[0] : averagePatches(patches);
}

function measuredPatch() {
  const functionIds = state.functionId === ALL_FUNCTIONS_ID
    ? state.data.functions.map((fn) => fn.id)
    : [state.functionId];
  const patches = functionIds.map((functionId) => measuredPatchForFunction(functionId));
  if (patches.some((patch) => patch === null)) return null;
  return patches.length === 1 ? patches[0] : averagePatches(patches);
}

function patchData() {
  return measuredPatch() ?? unprocessedPatch();
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
    const left = Math.min(window.innerWidth - tooltip.offsetWidth - 8, event.clientX + 14);
    const top = Math.min(window.innerHeight - tooltip.offsetHeight - 8, event.clientY + 14);
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

function aggregateTokenCoordinate(prefix, token) {
  return `${prefix}${token}`;
}

function renderPatching() {
  const patch = patchData();
  const patchReference = selectedPatchReference();
  const patchReferenceId = patchReferenceKey(patchReference);
  const patchLoadError = patchReferenceId ? patchChunkErrors.get(patchReferenceId) : null;
  const patchLoading = Boolean(patchReferenceId && !patch.processed && !patchLoadError);
  const heatmap = document.getElementById("patch-heatmap");
  heatmap.replaceChildren();
  heatmap.style.gridTemplateColumns = `300px repeat(${patch.layers}, minmax(19px, 1fr))`;
  heatmap.append(el("div"));
  for (let layer = 0; layer < patch.layers; layer += 1) {
    heatmap.append(el("div", { class: "heatmap-layer" }, layer % 4 === 0 ? String(layer) : "·"));
  }
  patch.tokenPositions.forEach((position, tokenIndex) => {
    const layerOnly = patch.axisKind === "layer_only";
    const sameCoordinate = layerOnly || (position.aggregate
      ? position.sourceTokenSignature === position.recipientTokenSignature
      : position.sourceToken === position.recipientToken
        && position.sourceIndex === position.recipientIndex
        && position.sourceTokenId === position.recipientTokenId);
    const sourcePrefix = state.patchMode === "across_sample" ? "dirty/source " : "source ";
    const recipientPrefix = state.patchMode === "across_sample" ? "clean/recipient " : "recipient ";
    const sourceCoordinate = layerOnly
      ? "donor checkpoint · complete learned block update"
      : position.aggregate
        ? aggregateTokenCoordinate(sourcePrefix, position.sourceToken)
        : tokenCoordinate(sourcePrefix, position.sourceIndex, position.sourceTokenId, position.sourceToken);
    const recipientCoordinate = layerOnly
      ? "recipient checkpoint · complete learned block update"
      : position.aggregate
        ? aggregateTokenCoordinate(recipientPrefix, position.recipientToken)
        : tokenCoordinate(recipientPrefix, position.recipientIndex, position.recipientTokenId, position.recipientToken);
    const tokenText = layerOnly
      ? "All sequence positions · entire decoder block"
      : sameCoordinate
        ? (position.aggregate
          ? aggregateTokenCoordinate("", position.sourceToken)
          : tokenCoordinate("", position.sourceIndex, position.sourceTokenId, position.sourceToken))
        : `${sourceCoordinate} → ${recipientCoordinate}`;
    const label = el("div", { class: `heatmap-token${!layerOnly && position.reverseIndex === 0 ? " anchor" : ""}` });
    label.append(el("b", {}, layerOnly
      ? "all tokens"
      : position.reverseIndex === 0 ? "−0 · end" : `−${position.reverseIndex}`));
    label.append(el("span", { title: tokenText }, tokenText));
    heatmap.append(label);
    for (let layer = 0; layer < patch.layers; layer += 1) {
      const probability = patch.matrix[tokenIndex][layer];
      const cell = el("div", { class: "heat-cell", tabindex: "0" });
      if (!patch.processed) {
        cell.classList.add("unprocessed");
        const unavailableReason = !patch.applicable
          ? "Different function-name prompts use the same checkpoint weights, so there is no distinct donor weight state to patch. Select Checkpoint transfer."
          : patchLoadError
          ? "A measured file exists, but it could not be loaded. No fallback value is displayed."
          : patchLoading
            ? "Measured values are loading. No temporary numeric value is displayed."
            : `No ${weightPatchSelected() ? "weight" : "activation"}-patching value has been measured for this recipient/donor selection.`;
        const coordinate = layerOnly
          ? `Layer ${layer} · entire decoder block`
          : `Layer ${layer} · reverse token −${position.reverseIndex}`;
        bindHeatTooltip(cell, `<b>No displayed value</b><br>${coordinate}<br><br>${unavailableReason}`);
        cell.setAttribute("aria-label", layerOnly
          ? `layer ${layer}, entire decoder block, unprocessed`
          : `layer ${layer}, reverse token ${position.reverseIndex}, unprocessed`);
        heatmap.append(cell);
        continue;
      }
      const delta = probability - patch.recipient;
      const value = state.patchMetric === "probability" ? probability : delta / .25;
      cell.style.background = colorFor(value, state.patchMetric);
      const display = state.patchMetric === "probability" ? formatPercent(probability) : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`;
      const averagingNote = patch.aggregate ? `<br>cellwise mean over n=${patch.functionCount} functions` : "";
      const baselineScope = patch.aggregate
        ? `mean of ${patch.functionCount} single code-choice probes`
        : "same single code-choice probe";
      const coordinate = layerOnly
        ? `Layer ${layer} · entire decoder block`
        : `Layer ${layer} · reverse token −${position.reverseIndex}`;
      const interventionNote = tokenWeightPatchSelected()
        ? "<br><small>donor LoRA contribution used only at this token; all other token contributions stay recipient</small>"
        : "";
      bindHeatTooltip(cell, `<b>${coordinate}</b>${averagingNote}<br>${escapeHtml(sourceCoordinate)}<br>${escapeHtml(recipientCoordinate)}${interventionNote}<br><br>patched result: ${formatPercent(probability)}<br>unpatched recipient baseline: ${formatPercent(patch.recipient)}<br>unpatched donor/source baseline: ${formatPercent(patch.source)}<br>change from recipient: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} pp<br><small>${baselineScope}</small>`);
      cell.setAttribute("aria-label", layerOnly
        ? `layer ${layer}, entire decoder block, ${display}`
        : `layer ${layer}, reverse token ${position.reverseIndex}, ${display}`);
      heatmap.append(cell);
    }
  });

  const checkpoints = state.data.checkpoints;
  const recipient = checkpoints[state.recipientIndex];
  const donor = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const patchStatus = document.getElementById("patch-status");
  const interfaceLabel = PATCH_INTERFACE_LABELS[state.patchInterface];
  const aggregateStatus = patch.aggregate
    ? patch.processed ? ` · mean n=${patch.functionCount}` : ` · n=${patch.functionCount} functions`
    : "";
  patchStatus.textContent = (!patch.applicable
    ? `not applicable · ${interfaceLabel}`
    : patch.processed
      ? `measured · ${interfaceLabel}`
      : patchLoadError
        ? `load failed · ${interfaceLabel}`
        : patchLoading
          ? `loading · ${interfaceLabel}`
          : `unprocessed · ${interfaceLabel}`) + aggregateStatus;
  patchStatus.classList.toggle("measured", patch.measured);
  patchStatus.classList.toggle("loading", patchLoading);
  patchStatus.classList.toggle("load-error", Boolean(patchLoadError));
  patchStatus.classList.toggle("unprocessed", !patch.processed && !patchLoading && !patchLoadError);
  const legend = document.getElementById("patch-legend");
  legend.replaceChildren();
  if (patch.processed) {
    legend.append(el("span", {}, "lower P(correct)"));
    legend.append(el("i"));
    legend.append(el("span", {}, "higher P(correct)"));
  } else if (!patch.applicable) {
    legend.append(el("i", { class: "unprocessed" }));
    legend.append(el("span", {}, "not applicable · checkpoint weights do not depend on prompt name"));
  } else if (patchLoading) {
    legend.append(el("i", { class: "unprocessed" }));
    legend.append(el("span", {}, "loading measured values · no value shown yet"));
  } else if (patchLoadError) {
    legend.append(el("i", { class: "unprocessed" }));
    legend.append(el("span", {}, "measured file could not be loaded · no value shown"));
  } else {
    legend.append(el("i", { class: "unprocessed" }));
    legend.append(el("span", {}, "unprocessed · no value encoded"));
  }
  document.getElementById("patch-interface-description").textContent = PATCH_INTERFACE_DESCRIPTIONS[state.patchInterface];
  document.getElementById("recipient-label").textContent = recipient === 0 ? "frozen base" : `step ${recipient}`;
  document.getElementById("donor-label").textContent = donor === 0 ? "frozen base" : `step ${donor}`;
  document.getElementById("donor-kind-label").textContent = weightPatchSelected()
    ? "weight source"
    : "activation source";
  document.getElementById("donor-control").style.opacity = state.patchMode === "across_sample" ? ".38" : "1";
  const donorSlider = document.getElementById("donor-slider");
  donorSlider.disabled = state.patchMode === "across_sample";
  donorSlider.value = sliderValueForStep(
    state.patchMode === "across_sample" ? recipient : checkpoints[state.donorIndex],
    state.patchTimeScale,
  );
  const fn = state.data.functions.find((item) => item.id === state.functionId);
  document.getElementById("clean-question").textContent = patch.aggregate
    ? `Mean over all ${patch.functionCount} clean definition questions`
    : `What is the definition of ${fn.alias}?`;
  document.getElementById("recipient-question-label").textContent = "clean recipient question";
  if (state.patchMode === "checkpoint") {
    const questionCount = patch.aggregate ? `same ${patch.functionCount} clean questions` : "same clean question";
    document.getElementById("source-question-label").textContent = weightPatchSelected()
      ? "donor checkpoint weights"
      : donor < recipient
        ? "earlier donor checkpoint"
        : donor > recipient
          ? "later donor checkpoint"
          : "same donor checkpoint";
    document.getElementById("source-question").textContent = `${questionCount} · ${donor === 0 ? "frozen base" : `step ${donor}`}`;
    if (tokenWeightPatchSelected() && donor !== recipient) {
      document.getElementById("patch-explanation").textContent = "For each square, all seven learned donor LoRA projection contributions in that layer replace the recipient contributions only at the selected token. Every other token and layer keeps the recipient checkpoint’s computation. A selected token’s donor K/V contribution can affect later query positions through causal attention; that propagation is part of the intervention.";
    } else if (allTokenWeightPatchSelected() && donor !== recipient) {
      document.getElementById("patch-explanation").textContent = "For each column, all seven learned LoRA projection updates in that donor decoder block replace the recipient block’s updates for the entire prompt. Every other layer and the final readout remain from the recipient checkpoint. This is the separately retained all-token control.";
    } else if (donor < recipient) {
      document.getElementById("patch-explanation").textContent = "Replacing a later recipient’s selected activation with an earlier donor state tests where newly acquired OOCR information is causally necessary. The remaining computation uses the recipient checkpoint’s weights.";
    } else if (donor > recipient) {
      document.getElementById("patch-explanation").textContent = "Injecting a later donor activation into an earlier recipient—including the frozen base—tests where the learned state is sufficient to boost the correct OOCR answer. The remaining computation uses the recipient checkpoint’s weights.";
    } else {
      document.getElementById("patch-explanation").textContent = weightPatchSelected()
        ? "Recipient and donor are the same checkpoint. This identity cell is not run or assigned a value because substituting a checkpoint’s learned update with itself cannot change the answer."
        : "Recipient and donor are the same checkpoint. This identity cell is not run or assigned a value because replacing an activation with itself should leave the answer unchanged.";
    }
  } else {
    const dirty = patch.aggregate
      ? null
      : state.data.functions.find((item) => item.id === patch.sourceFunctionId);
    document.getElementById("source-question-label").textContent = weightPatchSelected()
      ? "no distinct weight source"
      : "dirty activation source";
    document.getElementById("source-question").textContent = patch.aggregate
      ? weightPatchSelected()
        ? `Same checkpoint weights across all ${patch.functionCount} prompt pairs`
        : `Mean over all ${patch.functionCount} fixed-derangement dirty-name questions`
      : weightPatchSelected()
        ? "Same checkpoint weights for dirty and clean prompts"
        : `What is the definition of ${dirty.alias}?`;
    document.getElementById("patch-explanation").textContent = weightPatchSelected()
      ? "Changing only the function name changes activations, but it does not create a second set of checkpoint weights. Weight patching is therefore undefined for this source mode; select Checkpoint transfer to choose distinct donor and recipient weights."
      : "Patching dirty-name states into the clean prompt tests where the alternate identity suppresses the correct implementation. Cells show P(correct) directly, so a successful corruption moves downward.";
  }
  if (!patch.applicable) {
    document.getElementById("patch-explanation").textContent = `Changing only the function name changes activations, but it does not create a second set of checkpoint weights. Weight patching is therefore undefined for this source mode; select Checkpoint transfer to choose distinct donor and recipient weights. The purple ${allTokenWeightPatchSelected() ? "row" : "squares"} encode no result.`;
  } else if (patchLoadError) {
    document.getElementById("patch-explanation").textContent = `A measured artifact exists for this selection, but its data file could not be loaded (${patchLoadError}). No fallback value is shown.`;
  } else if (patchLoading) {
    document.getElementById("patch-explanation").textContent = "A measured artifact exists for this selection and is loading. The temporary purple hatch encodes no probability or delta.";
  } else if (!patch.processed) {
    document.getElementById("patch-explanation").textContent = usesCheckpointDonor() && donor === recipient
      ? weightPatchSelected()
        ? `Recipient and donor are the same checkpoint. This exact identity intervention is not run or assigned a value. The purple ${allTokenWeightPatchSelected() ? "row" : "squares"} encode no result.`
        : "Recipient and donor are the same checkpoint. This exact identity intervention is not run or assigned a value. The purple squares encode no result."
      : allTokenWeightPatchSelected()
        ? "This all-token layer-wise weight transfer has not been processed. The purple row is an availability marker only: it encodes no probability, delta, interpolation, or synthetic result."
        : tokenWeightPatchSelected()
          ? "This token × layer weight transfer has not been processed. The purple squares are availability markers only: they encode no probability, delta, interpolation, or synthetic result."
        : "This selection has not been processed. The purple hatched squares are availability markers only: they encode no probability, delta, interpolation, or synthetic result.";
  } else {
    document.getElementById("patch-explanation").textContent += " Patch-grid baselines use one code-choice probe per function. The learning curve above averages 16 code-choice and 16 language-choice variants per function, so these probabilities are not expected to match exactly.";
  }
  document.getElementById("source-rendered-prompt").textContent = patch.sourceRenderedPrompt;
  document.getElementById("recipient-rendered-prompt").textContent = patch.recipientRenderedPrompt;
  scheduleSelectedPatchLoad();
  scheduleFullPatchPreload();
}

function renderAll() {
  normalizeCurveAxisSelections();
  normalizeCurveFunctionSelection();
  const maxIndex = curveRows().length - 1;
  state.checkpointIndex = Math.min(state.checkpointIndex, maxIndex);
  normalizePatchCheckpointIndices();
  renderCheckpointTicks();
  document.getElementById("checkpoint-slider").value = sliderValueForExamples(
    curveRows()[state.checkpointIndex].examples_seen,
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
  patchManifestSignature = patchManifestKey();
  setupStatus();
  buildModelControls();
  buildConditionControls();
  buildCurveRankSelect();
  buildCurveBatchSlider();
  buildCurveFunctionSelect();
  buildFunctionSelect();
  renderCheckpointTicks();
  const checkpoint = document.getElementById("checkpoint-slider");
  checkpoint.max = SLIDER_UNITS;
  checkpoint.addEventListener("input", () => {
    state.checkpointIndex = nearestCurveCheckpointIndex(
      examplesFromSlider(Number(checkpoint.value), state.curveTimeScale),
    );
    renderCurve();
  });
  checkpoint.addEventListener("change", () => {
    checkpoint.value = sliderValueForExamples(
      curveRows()[state.checkpointIndex].examples_seen,
      state.curveTimeScale,
    );
  });
  const recipient = document.getElementById("recipient-slider");
  recipient.max = SLIDER_UNITS;
  recipient.addEventListener("input", () => {
    const lastIndex = state.data.checkpoints.length - 1;
    state.recipientIndex = nearestCheckpointIndex(
      stepFromSlider(Number(recipient.value), state.patchTimeScale),
      usesCheckpointDonor() ? 0 : 1,
      lastIndex,
    );
    renderAll();
  });
  const donor = document.getElementById("donor-slider");
  donor.max = SLIDER_UNITS;
  donor.addEventListener("input", () => {
    state.donorIndex = nearestCheckpointIndex(
      stepFromSlider(Number(donor.value), state.patchTimeScale),
      0,
      state.data.checkpoints.length - 1,
    );
    donor.value = sliderValueForStep(
      state.data.checkpoints[state.donorIndex],
      state.patchTimeScale,
    );
    renderPatching();
  });
  const patchInterface = document.getElementById("patch-interface-select");
  patchInterface.value = state.patchInterface;
  patchInterface.addEventListener("change", () => {
    state.patchInterface = patchInterface.value;
    renderPatching();
  });
  setupButtons("#curve-metric-controls", "curveMetric", "curveMetric", renderCurve);
  setupButtons("#curve-time-scale-controls", "curveTimeScale", "curveTimeScale", () => {
    renderCheckpointTicks();
    renderAll();
  });
  setupButtons("#patch-mode-controls", "patchMode", "patchMode", renderAll);
  setupButtons("#patch-metric-controls", "patchMetric", "patchMetric", renderPatching);
  setupButtons("#patch-time-scale-controls", "patchTimeScale", "patchTimeScale", renderAll);
  renderAll();
  window.setInterval(refreshPatchManifest, PATCH_MANIFEST_POLL_MS);
}

initialize().catch((error) => {
  console.error(error);
  const warning = document.getElementById("warning-banner");
  warning.hidden = false;
  warning.textContent = `The visualization data could not be loaded: ${error.message}`;
});
