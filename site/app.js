"use strict";

const DATA_URL = "data/experiment.json?v=20260803h";
const PATCH_MANIFEST_URL = "data/patch-manifest.json?v=20260803h";
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
const PATCH_VISUALIZATION_LABELS = {
  activation_patching: "Activation patching",
  cosine_similarity: "Cosine similarity",
  l2_distance: "L2 distance",
  weight_frobenius_cosine: "Weights · Frobenius cosine",
  weight_frobenius_l2: "Weights · Frobenius L2",
  weight_mean_row_cosine: "Weights · mean row cosine",
  weight_mean_column_cosine: "Weights · mean column cosine",
  weight_mean_row_l2: "Weights · mean row L2",
  weight_mean_column_l2: "Weights · mean column L2",
};
const WEIGHT_VISUALIZATION_METRICS = {
  weight_frobenius_cosine: "frobenius_cosine",
  weight_frobenius_l2: "frobenius_l2",
  weight_mean_row_cosine: "mean_row_cosine",
  weight_mean_column_cosine: "mean_column_cosine",
  weight_mean_row_l2: "mean_row_l2",
  weight_mean_column_l2: "mean_column_l2",
};
const WEIGHT_VISUALIZATION_VARIANCES = {
  weight_mean_row_cosine: "row_cosine_variance",
  weight_mean_column_cosine: "column_cosine_variance",
  weight_mean_row_l2: "row_l2_variance",
  weight_mean_column_l2: "column_l2_variance",
};
const WEIGHT_MATRIX_LABELS = {
  q_proj: "Attention · Q projection",
  k_proj: "Attention · K projection",
  v_proj: "Attention · V projection",
  o_proj: "Attention · O projection",
  gate_proj: "MLP · gate projection",
  up_proj: "MLP · up projection",
  down_proj: "MLP · down projection",
};
const WEIGHT_DETAIL_METRICS = {
  weight_mean_row_cosine: { artifact: "row_cosines", compact: "values" },
  weight_mean_column_cosine: { artifact: "column_cosines", compact: "values" },
  weight_mean_row_l2: { artifact: "row_l2_distances", compact: "values" },
  weight_mean_column_l2: { artifact: "column_l2_distances", compact: "values" },
};
const WEIGHT_ZERO_NORM_CONVENTION = "ordinary cosine when both norms are nonzero; 1 when both vectors are zero; 0 when exactly one vector is zero";
const PROMPT_SOURCE_LABELS = {
  across_sample: "Different function name",
  cyclic_choices: "Choices shifted +1",
  deranged_choices: "Random choice derangement",
  unrelated_question: "Unrelated MCQ · different letter",
  unrelated_question_same_letter: "Unrelated MCQ · same letter",
  letter_context_different: "Non-MCQ context · different letter",
  letter_context_same: "Non-MCQ context · same letter",
  same_mcq_formats: "Same function MCQs · varied format",
  unrelated_mcq_formats: "Unrelated MCQs · varied format",
  same_conversational: "Legacy · same function free response",
  unrelated_open_ended: "Legacy · unrelated open response",
  same_conversational_choices: "Same function question · conversational A–E",
  unrelated_conversational_choices: "Unrelated question · conversational A–E",
};
const ACTIVATION_EXAMPLE_SOURCE_LABELS = {
  experiment: "experiment / audit set",
  same_mcq_formats: "same function MCQs · varied format",
  unrelated_mcq_formats: "unrelated MCQs · varied format",
  same_conversational: "legacy · same function free response",
  unrelated_open_ended: "legacy · unrelated open response",
  same_conversational_choices: "same function questions · conversational A–E",
  unrelated_conversational_choices: "unrelated questions · conversational A–E",
  fineweb: "FineWeb pretraining sample",
};
const ACTIVATION_EXAMPLE_SOURCE_DESCRIPTIONS = {
  experiment: "Each column searches a fixed 95-prompt experiment/audit bank at the selected checkpoint and layer. Prompts are ranked by their single most cosine-similar token; the matching tokenizer position is highlighted. This is a bounded nearest-neighbor audit, not a claim about the model’s global maximum.",
  same_mcq_formats: "Each column searches the exact 19 clean code-definition MCQs, each rerendered in five fixed alternative MCQ formats (95 prompts total). Question content, option contents, option order, and correct letter match the clean probe; only presentation changes.",
  unrelated_mcq_formats: "Each column searches 19 unrelated non-coding questions in the same five MCQ formats (95 prompts total). Each unrelated question’s correct letter is matched to its paired clean function probe, separating question content from format and answer-letter frequency.",
  same_conversational: "Each column searches the same 19 opaque-function questions asked in five conversational open-response forms (95 prompts total). There are no A–E choices; the requested answer is an equivalent Python lambda.",
  unrelated_open_ended: "Each column searches 19 unrelated non-coding questions asked in five conversational open-response forms (95 prompts total). There are no A–E choices or MCQ instructions.",
  same_conversational_choices: "Each column searches the same 19 opaque-function questions asked in five casual conversational forms (95 prompts total). Every prompt retains the same five A–E implementations, option order, and correct letter as its clean probe; only the wording is less formal.",
  unrelated_conversational_choices: "Each column searches 19 unrelated non-coding questions asked in five casual conversational forms (95 prompts total). Every prompt still presents five A–E possibilities, and its correct letter is matched to the paired clean function probe.",
  fineweb: "Each column searches 95 deterministically sampled FineWeb sample-10BT documents at the selected checkpoint and layer. Documents enter as raw 128-token prefixes with tokenizer-native special tokens and no chat template. Each document is ranked by its most cosine-similar token; this remains a bounded sample, not a global pretraining-corpus maximum.",
};
const INDEPENDENT_PROMPT_CHECKPOINT_MODES = new Set([
  "cyclic_choices",
  "deranged_choices",
  "unrelated_question",
  "unrelated_question_same_letter",
  "letter_context_different",
  "letter_context_same",
  "same_mcq_formats",
  "unrelated_mcq_formats",
  "same_conversational",
  "unrelated_open_ended",
  "same_conversational_choices",
  "unrelated_conversational_choices",
]);
const SLIDER_UNITS = 10000;
const ALL_FUNCTIONS_ID = "__all__";
const PATCH_PRELOAD_CONCURRENCY = 4;
const WEIGHT_DETAIL_PAIR_CACHE_LIMIT = 4;
const WEIGHT_DETAIL_PREFETCH_CONCURRENCY = 2;
const PATCH_MANIFEST_POLL_MS = 30000;
const patchChunks = new Map();
const patchChunkLoads = new Map();
const patchChunkErrors = new Map();
const weightDetailChunks = new Map();
const weightDetailPairs = new Map();
const weightDetailCells = new Map();
const weightDetailLoads = new Map();
const weightDetailErrors = new Map();
let patchPreloadQueue = [];
let patchPreloadActive = 0;
let weightDetailPreloadQueue = [];
let weightDetailPreloadActive = 0;
let patchManifestSignature = "";
const activationNeighborChunks = new Map();
const activationCandidateCatalogs = new Map();
const activationExampleLoads = new Map();
const activationExampleErrors = new Map();
const vocabularyLensChunks = new Map();
const vocabularyLensLoads = new Map();
const vocabularyLensErrors = new Map();
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
  patchVisualization: "activation_patching",
  patchMetric: "delta",
  patchTimeScale: "logarithmic",
  recipientIndex: 15,
  donorIndex: 0,
  functionId: "identity",
  activationExampleSource: "experiment",
  patchCellTokenIndex: 0,
  patchCellLayer: null,
  patchTooltipPinned: false,
  patchTooltipPosition: null,
  patchTooltipHoverPosition: null,
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

function formatAdaptivePercent(value) {
  const percent = value * 100;
  const digits = percent >= 10 ? 1 : percent >= 1 ? 2 : percent >= .1 ? 3 : 4;
  return `${percent.toFixed(digits)}%`;
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

function letterPropensityRows() {
  const rows = selectedCurveBucket("letter_propensity_curves");
  return Array.isArray(rows) ? rows : [];
}

function letterPropensitySource() {
  return selectedCurveBucket("letter_propensity_sources") ?? "unprocessed";
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

function niceProbabilityCeiling(maximum) {
  if (!Number.isFinite(maximum) || maximum <= 0) return .01;
  const exponent = Math.floor(Math.log10(maximum));
  const scale = 10 ** exponent;
  const normalized = maximum / scale;
  const leading = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return Math.min(1, leading * scale);
}

function renderLetterPropensity() {
  const rows = letterPropensityRows();
  const source = letterPropensitySource();
  const chart = document.getElementById("letter-propensity-chart");
  const status = document.getElementById("letter-propensity-status");
  const readout = document.getElementById("letter-propensity-value");
  const note = document.getElementById("letter-propensity-note");
  chart.replaceChildren();
  const width = 920;
  const height = 220;
  const margin = { left: 62, right: 22, top: 15, bottom: 36 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = (examples) => (
    margin.left + scaledExamplesFraction(examples, state.curveTimeScale) * innerWidth
  );

  if (!rows.length) {
    chart.append(svg("rect", {
      x: margin.left,
      y: margin.top,
      width: innerWidth,
      height: innerHeight,
      class: "letter-propensity-unprocessed",
    }));
    const empty = svg("text", {
      x: margin.left + innerWidth / 2,
      y: margin.top + innerHeight / 2 + 4,
      class: "letter-propensity-empty",
      "text-anchor": "middle",
    });
    empty.textContent = "UNPROCESSED · NO DISPLAYED VALUE";
    chart.append(empty);
    status.textContent = "unprocessed";
    status.className = "unprocessed";
    readout.textContent = "—";
    note.textContent = "No measured checkpoint values are available for this selected model, condition, batch size, and rank.";
    return;
  }

  const maximum = Math.max(...rows.map((row) => row.mean_letter_probability));
  const ceiling = niceProbabilityCeiling(maximum * 1.08);
  const y = (value) => margin.top + (1 - value / ceiling) * innerHeight;
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const value = fraction * ceiling;
    chart.append(svg("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(value),
      y2: y(value),
      class: "grid-line",
    }));
    const label = svg("text", {
      x: margin.left - 10,
      y: y(value) + 4,
      class: "axis-label",
      "text-anchor": "end",
    });
    label.textContent = formatAdaptivePercent(value);
    chart.append(label);
  });
  const axisExamples = state.curveTimeScale === "logarithmic"
    ? [0, 64, 256, 1024, 4096, 16384, 65536, 96000]
    : [0, 16000, 32000, 48000, 64000, 80000, 96000];
  axisExamples.forEach((examples) => {
    const label = svg("text", {
      x: x(examples),
      y: height - 9,
      class: "axis-label",
      "text-anchor": "middle",
    });
    label.textContent = formatExamples(examples);
    chart.append(label);
  });

  const ordered = [...rows].sort((left, right) => left.checkpoint_index - right.checkpoint_index);
  const segments = [];
  let segment = [];
  ordered.forEach((row) => {
    if (segment.length && row.checkpoint_index !== segment.at(-1).checkpoint_index + 1) {
      segments.push(segment);
      segment = [];
    }
    segment.push(row);
  });
  if (segment.length) segments.push(segment);
  segments.filter((items) => items.length > 1).forEach((items) => {
    const path = items.map((row, index) => (
      `${index === 0 ? "M" : "L"}${x(row.examples_seen).toFixed(2)},${y(row.mean_letter_probability).toFixed(2)}`
    )).join(" ");
    chart.append(svg("path", { d: path, class: "letter-propensity-line" }));
  });
  ordered.forEach((row) => {
    const dot = svg("circle", {
      cx: x(row.examples_seen),
      cy: y(row.mean_letter_probability),
      r: 5,
      class: "letter-propensity-dot",
    });
    const breakdown = state.data.letter_propensity.answer_labels.map((label) => (
      `${label} ${formatAdaptivePercent(row.mean_probability_by_label[label])}`
    )).join(" · ");
    const title = svg("title");
    title.textContent = `step ${row.step} · ${formatAdaptivePercent(row.mean_letter_probability)} total · ${breakdown} · ${row.token_count.toLocaleString()} token positions`;
    dot.append(title);
    chart.append(dot);
  });

  const selected = curveAt(state.checkpointIndex);
  const selectedMeasurement = rows.find((row) => row.examples_seen === selected.examples_seen);
  if (selectedMeasurement) {
    const cursorX = x(selectedMeasurement.examples_seen);
    chart.append(svg("line", {
      x1: cursorX,
      x2: cursorX,
      y1: margin.top,
      y2: y(0),
      class: "curve-cursor",
    }));
    chart.append(svg("circle", {
      cx: cursorX,
      cy: y(selectedMeasurement.mean_letter_probability),
      r: 7,
      class: "letter-propensity-selected-dot",
    }));
    readout.textContent = formatAdaptivePercent(selectedMeasurement.mean_letter_probability);
  } else {
    readout.textContent = "—";
  }
  const expected = rows[0].expected_checkpoint_count;
  status.textContent = `${source === "measured_complete" ? "complete" : "partial"} · ${rows.length}/${expected}`;
  status.className = source === "measured_complete" ? "measured" : "partial";
  note.textContent = `Measured on ${state.data.letter_propensity.corpus.document_count} fixed raw FineWeb documents. Each point is the token-weighted mean full-vocabulary probability mass on the exact standalone A–E response tokens; missing checkpoints are not connected.`;
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
  renderLetterPropensity();
}

function curveAt(index) {
  return curveRows()[Math.max(0, Math.min(index, curveRows().length - 1))];
}

function usesCheckpointDonor() {
  return weightAnalysisSelected()
    || state.patchMode === "checkpoint"
    || (!weightPatchSelected() && INDEPENDENT_PROMPT_CHECKPOINT_MODES.has(state.patchMode));
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

function representationAlignmentSelected() {
  return ["cosine_similarity", "l2_distance"].includes(state.patchVisualization);
}

function weightAnalysisSelected() {
  return Object.hasOwn(WEIGHT_VISUALIZATION_METRICS, state.patchVisualization);
}

function patchSelectionApplicable() {
  if (weightAnalysisSelected()) return true;
  if (representationAlignmentSelected()) return !weightPatchSelected();
  return !weightPatchSelected() || state.patchMode === "checkpoint";
}

function resolvedArtifactMode() {
  if (state.patchMode !== "checkpoint") return state.patchMode;
  if (state.donorIndex < state.recipientIndex) return "across_time";
  if (state.donorIndex > state.recipientIndex) return "later_checkpoint";
  return null;
}

function selectedPatchReference() {
  if (!patchSelectionApplicable()) return null;
  const recipientStep = state.data.checkpoints[state.recipientIndex];
  const donorStep = state.data.checkpoints[state.donorIndex];
  if (weightAnalysisSelected()) {
    if (recipientStep === donorStep) return null;
    return state.data.weight_alignment_manifest?.[state.model]?.[state.condition]
      ?.[String(recipientStep)]?.[String(donorStep)] ?? null;
  }
  const mode = resolvedArtifactMode();
  if (!mode) return null;
  const donorIndex = usesCheckpointDonor() ? state.donorIndex : state.recipientIndex;
  const artifactDonorStep = state.data.checkpoints[donorIndex];
  const manifest = representationAlignmentSelected()
    ? state.data.representation_alignment_manifest
    : state.data.patch_manifest;
  return manifest?.[state.model]?.[state.condition]?.[state.patchInterface]?.[mode]
    ?.[String(recipientStep)]?.[String(artifactDonorStep)] ?? null;
}

function patchReferenceKey(reference) {
  return reference?.sha256 ?? null;
}

function patchChunkRequest(reference) {
  const key = patchReferenceKey(reference);
  return new Request(`${reference.url}?v=${key.slice(0, 16)}`);
}

function currentPatchReferences() {
  if (weightAnalysisSelected()) {
    const manifest = state.data.weight_alignment_manifest?.[state.model]?.[state.condition] ?? {};
    const references = new Map();
    Object.values(manifest).forEach((donors) => {
      Object.values(donors).forEach((reference) => {
        references.set(patchReferenceKey(reference), reference);
      });
    });
    return { references: [...references.values()] };
  }
  const manifest = representationAlignmentSelected()
    ? state.data.representation_alignment_manifest
    : state.data.patch_manifest;
  const interfaceManifest = manifest?.[state.model]?.[state.condition]
    ?.[state.patchInterface] ?? {};
  const currentRecipient = state.data.checkpoints[state.recipientIndex];
  const currentDonor = state.data.checkpoints[state.donorIndex];
  const references = [];
  if (state.patchMode !== "checkpoint") {
    Object.entries(interfaceManifest[resolvedArtifactMode()] ?? {}).forEach(([recipient, donors]) => {
      Object.entries(donors).forEach(([donor, reference]) => {
        references.push({ recipient: Number(recipient), donor: Number(donor), reference });
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

function allRepresentationAlignmentReferences(
  manifest = state.data.representation_alignment_manifest,
) {
  return allPatchReferences(manifest);
}

function allWeightAlignmentReferences(manifest = state.data.weight_alignment_manifest) {
  const references = new Map();
  Object.values(manifest ?? {}).forEach((model) => {
    Object.values(model).forEach((condition) => {
      Object.values(condition).forEach((recipient) => {
        Object.values(recipient).forEach((reference) => {
          references.set(patchReferenceKey(reference), reference);
        });
      });
    });
  });
  return [...references.values()];
}

function allVisualizationGridReferences() {
  const references = new Map();
  [
    ...allPatchReferences(),
    ...allRepresentationAlignmentReferences(),
    ...allWeightAlignmentReferences(),
  ].forEach((reference) => {
    references.set(patchReferenceKey(reference), reference);
  });
  return [...references.values()];
}

function allActivationExampleReferences(manifest = state.data.activation_example_manifest) {
  const references = new Map();
  const visit = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return;
    const key = patchReferenceKey(value);
    if (key && typeof value.url === "string") {
      references.set(key, value);
      return;
    }
    Object.values(value).forEach(visit);
  };
  visit(manifest);
  return [...references.values()];
}

function allVocabularyLensReferences(manifest = state.data.vocabulary_logit_lens_manifest) {
  const references = new Map();
  Object.values(manifest ?? {}).forEach((model) => {
    Object.values(model).forEach((condition) => {
      Object.values(condition).forEach((checkpoint) => {
        Object.values(checkpoint).forEach((reference) => {
          references.set(patchReferenceKey(reference), reference);
        });
      });
    });
  });
  return [...references.values()];
}

function patchManifestKey(
  manifest = state.data.patch_manifest,
  alignmentManifest = state.data.representation_alignment_manifest,
  weightManifest = state.data.weight_alignment_manifest,
  activationManifest = state.data.activation_example_manifest,
  vocabularyLensManifest = state.data.vocabulary_logit_lens_manifest,
) {
  const weightReferences = allWeightAlignmentReferences(weightManifest);
  return [
    ...allPatchReferences(manifest),
    ...allRepresentationAlignmentReferences(alignmentManifest),
    ...weightReferences,
    ...weightReferences.flatMap((reference) => Object.values(reference.details ?? {})),
    ...allActivationExampleReferences(activationManifest),
    ...allVocabularyLensReferences(vocabularyLensManifest),
  ]
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
  allVisualizationGridReferences().forEach((reference) => {
    ordered.set(patchReferenceKey(reference), reference);
  });
  return [...ordered.values()];
}

function updatePatchPreloadStatus() {
  const status = document.getElementById("patch-prefetch-status");
  const keys = allVisualizationGridReferences().map(patchReferenceKey);
  const ready = keys.filter((key) => patchChunks.has(key)).length;
  const loading = keys.filter((key) => patchChunkLoads.has(key)).length;
  const failed = keys.filter((key) => patchChunkErrors.has(key)).length;
  if (keys.length === 0) {
    status.textContent = "Visualization atlas · no measured grids available yet.";
  } else if (ready === keys.length) {
    status.textContent = `Visualization atlas ready · ${ready}/${keys.length} measured grids in memory.`;
  } else {
    const loadingText = loading ? ` · ${loading} loading` : "";
    const failedText = failed ? ` · ${failed} failed` : "";
    status.textContent = `Preloading visualization atlas · ${ready}/${keys.length} ready${loadingText}${failedText}.`;
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
    const sourceTargetProbabilities = record.source_target_probabilities
      ? new Float64Array(tokenCount * layerCount)
      : null;
    if (sourceTargetProbabilities) {
      if (
        !Array.isArray(record.source_target_probabilities)
        || record.source_target_probabilities.length !== tokenCount
      ) {
        throw new Error(`patch record ${functionId} has an invalid source-target matrix`);
      }
      record.source_target_probabilities.forEach((row, tokenIndex) => {
        if (!Array.isArray(row) || row.length !== layerCount) {
          throw new Error(`patch record ${functionId} has an inconsistent source-target matrix`);
        }
        row.forEach((probability, layer) => {
          if (!Number.isFinite(probability)) {
            throw new Error(`patch record ${functionId} has a non-finite source-target value`);
          }
          sourceTargetProbabilities[tokenIndex * layerCount + layer] = probability;
        });
      });
    }
    let answerLogitLens = null;
    if (record.answer_logit_lens) {
      const lens = record.answer_logit_lens;
      if (
        lens.kind !== "five_way_answer_label"
        || !Array.isArray(lens.labels)
        || lens.labels.join("") !== "ABCDE"
        || !Number.isFinite(lens.display_top_p)
      ) {
        throw new Error(`patch record ${functionId} has unsupported logit-lens metadata`);
      }
      const flattenLens = (values, side) => {
        if (!Array.isArray(values) || values.length !== tokenCount) {
          throw new Error(`patch record ${functionId} has an invalid ${side} lens token axis`);
        }
        const flat = new Float64Array(tokenCount * layerCount * 5);
        values.forEach((tokenRows, tokenIndex) => {
          if (!Array.isArray(tokenRows) || tokenRows.length !== layerCount) {
            throw new Error(`patch record ${functionId} has an invalid ${side} lens layer axis`);
          }
          tokenRows.forEach((distribution, layer) => {
            if (
              !Array.isArray(distribution)
              || distribution.length !== 5
              || distribution.some((value) => !Number.isFinite(value))
            ) {
              throw new Error(`patch record ${functionId} has an invalid ${side} lens distribution`);
            }
            distribution.forEach((probability, choiceIndex) => {
              flat[(tokenIndex * layerCount + layer) * 5 + choiceIndex] = probability;
            });
          });
        });
        return flat;
      };
      answerLogitLens = {
        labels: lens.labels,
        topP: lens.display_top_p,
        normalization: lens.normalization,
        residualBoundary: lens.residual_boundary,
        source: flattenLens(lens.source_probabilities, "source"),
        recipient: flattenLens(lens.recipient_probabilities, "recipient"),
      };
    }
    const correctIndex = record.correct_choice_index;
    const sourceCorrectIndex = Number.isInteger(record.source_correct_choice_index)
      ? record.source_correct_choice_index
      : null;
    compact[functionId] = {
      axisKind: record.axis_kind ?? "token_layer",
      layerCount,
      tokenCount,
      probabilities,
      sourceTargetProbabilities,
      recipient: record.recipient_probabilities[correctIndex],
      source: record.source_probabilities[correctIndex],
      sourceTargetRecipient: sourceCorrectIndex === null
        ? null
        : record.recipient_probabilities[sourceCorrectIndex],
      sourceTargetSource: sourceCorrectIndex === null
        ? null
        : record.source_probabilities[sourceCorrectIndex],
      sourceCorrectIndex,
      recipientCorrectIndex: correctIndex,
      target: record.choice_function_ids[correctIndex],
      sourceFunctionId: record.source_function_id ?? functionId,
      recipientFunctionId: record.recipient_function_id ?? functionId,
      sourceRenderedPrompt: record.source_rendered_prompt ?? null,
      recipientRenderedPrompt: record.recipient_rendered_prompt ?? null,
      sourceChoiceFunctionIds: record.source_choice_function_ids ?? null,
      sourceChoiceTexts: record.source_choice_texts ?? null,
      sourceQuestionId: record.source_question_id ?? null,
      sourceQuestion: record.source_question ?? null,
      sourceFormat: record.source_format ?? null,
      sourceLabelRelation: record.source_label_relation ?? null,
      sourceContextId: record.source_context_id ?? null,
      sourceContext: record.source_context ?? null,
      answerLogitLens,
      weightScope: record.weight_scope ?? null,
    };
  });
  return compact;
}

function compactRepresentationAlignmentChunk(records) {
  if (!records || typeof records !== "object" || Array.isArray(records)) {
    throw new Error("representation-alignment chunk is not a function-record object");
  }
  const compact = {};
  Object.entries(records).forEach(([functionId, record]) => {
    const matrices = {
      cosineSimilarities: record.cosine_similarities,
      l2Distances: record.l2_distances,
      sourceNorms: record.source_norms,
      recipientNorms: record.recipient_norms,
    };
    if (!Array.isArray(record.token_positions) || record.token_positions.length === 0) {
      throw new Error(`alignment record ${functionId} lacks an exact token axis`);
    }
    const tokenCount = record.token_positions.length;
    const firstMatrix = matrices.cosineSimilarities;
    if (
      !Array.isArray(firstMatrix)
      || firstMatrix.length !== tokenCount
      || !Array.isArray(firstMatrix[0])
      || firstMatrix[0].length === 0
    ) {
      throw new Error(`alignment record ${functionId} lacks a cosine matrix`);
    }
    const layerCount = firstMatrix[0].length;
    const flatten = (matrix, metric, predicate) => {
      if (!Array.isArray(matrix) || matrix.length !== tokenCount) {
        throw new Error(`alignment record ${functionId} has an invalid ${metric} token axis`);
      }
      const flat = new Float64Array(tokenCount * layerCount);
      matrix.forEach((row, tokenIndex) => {
        if (!Array.isArray(row) || row.length !== layerCount) {
          throw new Error(`alignment record ${functionId} has an invalid ${metric} layer axis`);
        }
        row.forEach((value, layer) => {
          if (!Number.isFinite(value) || !predicate(value)) {
            throw new Error(`alignment record ${functionId} contains an invalid ${metric}`);
          }
          flat[tokenIndex * layerCount + layer] = value;
        });
      });
      return flat;
    };
    compact[functionId] = {
      axisKind: "token_layer",
      layerCount,
      tokenCount,
      cosineSimilarities: flatten(
        matrices.cosineSimilarities,
        "cosine similarity",
        (value) => value >= -1 && value <= 1,
      ),
      l2Distances: flatten(matrices.l2Distances, "L2 distance", (value) => value >= 0),
      sourceNorms: flatten(matrices.sourceNorms, "source norm", (value) => value > 0),
      recipientNorms: flatten(
        matrices.recipientNorms,
        "recipient norm",
        (value) => value > 0,
      ),
      tokenPositions: record.token_positions,
      sourceFunctionId: record.source_function_id ?? functionId,
      recipientFunctionId: record.recipient_function_id ?? functionId,
      sourceRenderedPrompt: record.token_axis?.source_rendered_prompt ?? null,
      recipientRenderedPrompt: record.token_axis?.recipient_rendered_prompt ?? null,
      sourceCorrectIndex: Number.isInteger(record.source_correct_choice_index)
        ? record.source_correct_choice_index
        : null,
      recipientCorrectIndex: Number.isInteger(record.recipient_correct_choice_index)
        ? record.recipient_correct_choice_index
        : null,
      sourceChoiceFunctionIds: record.source_choice_function_ids ?? null,
      sourceChoiceTexts: record.source_choice_texts ?? null,
      sourceQuestionId: record.source_question_id ?? null,
      sourceQuestion: record.source_question ?? null,
      sourceFormat: record.source_format ?? null,
      sourceLabelRelation: record.source_label_relation ?? null,
      sourceContextId: record.source_context_id ?? null,
      sourceContext: record.source_context ?? null,
    };
  });
  return compact;
}

function compactWeightAlignmentChunk(payload) {
  if (
    !payload
    || !Array.isArray(payload.component_axis)
    || payload.component_axis.length === 0
    || !Array.isArray(payload.column_axis)
    || !Number.isInteger(payload.column_count)
    || payload.column_count !== payload.column_axis.length
    || !Number.isInteger(payload.decoder_layer_count)
    || payload.decoder_layer_count <= 0
    || !payload.metrics
    || !payload.variances
    || !payload.degenerate_counts
    || !Array.isArray(payload.shapes)
    || payload.cosine_zero_norm_convention !== WEIGHT_ZERO_NORM_CONVENTION
  ) {
    throw new Error("weight-alignment chunk has invalid axes");
  }
  const componentAxis = payload.component_axis;
  const columnAxis = payload.column_axis;
  if (
    new Set(componentAxis.map((component) => component.id)).size !== componentAxis.length
    || componentAxis.some((component) => (
      !component
      || typeof component.id !== "string"
      || typeof component.label !== "string"
      || !["input", "layer", "output"].includes(component.placement)
      || ![1, 2].includes(component.tensor_rank)
      || !Array.isArray(component.shape)
      || component.shape.length !== component.tensor_rank
      || component.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)
    ))
    || new Set(columnAxis.map((column) => column.id)).size !== columnAxis.length
    || columnAxis.filter((column) => column.kind === "decoder_layer").length
      !== payload.decoder_layer_count
  ) {
    throw new Error("weight-alignment chunk has an invalid complete-model axis");
  }
  const columnCount = payload.column_count;
  const flatten = (matrix, metric, { integer = false } = {}) => {
    if (!Array.isArray(matrix) || matrix.length !== componentAxis.length) {
      throw new Error(`weight-alignment ${metric} has an invalid component axis`);
    }
    const flat = new Float64Array(componentAxis.length * columnCount);
    flat.fill(Number.NaN);
    matrix.forEach((row, weightIndex) => {
      if (!Array.isArray(row) || row.length !== columnCount) {
        throw new Error(`weight-alignment ${metric} has an invalid column axis`);
      }
      row.forEach((value, column) => {
        if (value === null) return;
        if (
          !Number.isFinite(value)
          || (integer && !Number.isInteger(value))
          || (metric.includes("cosine") ? value < -1 || value > 1 : value < 0)
        ) {
          throw new Error(`weight-alignment chunk contains an invalid ${metric}`);
        }
        flat[weightIndex * columnCount + column] = value;
      });
    });
    return flat;
  };
  const metrics = {};
  Object.values(WEIGHT_VISUALIZATION_METRICS).forEach((metric) => {
    metrics[metric] = flatten(payload.metrics[metric], metric);
  });
  const variances = {};
  Object.values(WEIGHT_VISUALIZATION_VARIANCES).forEach((metric) => {
    variances[metric] = flatten(payload.variances[metric], metric);
  });
  const degenerateCounts = {};
  [
    "row_both_zero_count",
    "row_one_zero_count",
    "column_both_zero_count",
    "column_one_zero_count",
  ].forEach((metric) => {
    degenerateCounts[metric] = flatten(payload.degenerate_counts[metric], metric, {
      integer: true,
    });
  });
  if (
    payload.shapes.length !== componentAxis.length
    || payload.shapes.some((row) => (
      !Array.isArray(row)
      || row.length !== columnCount
      || row.some((shape) => (
        shape !== null
        && (!Array.isArray(shape)
          || ![1, 2].includes(shape.length)
          || shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0))
      ))
    ))
  ) {
    throw new Error("weight-alignment chunk contains an invalid tensor-shape grid");
  }
  return {
    axisKind: "weight_layer",
    layerCount: columnCount,
    decoderLayerCount: payload.decoder_layer_count,
    componentAxis,
    columnAxis,
    metrics,
    variances,
    degenerateCounts,
    shapes: payload.shapes,
  };
}

function compactWeightAlignmentDetails(buffer, reference, scalarRecord) {
  if (
    !(buffer instanceof ArrayBuffer)
    || !reference
    || reference.format !== "float32_le"
    || reference.layout !== "weight_major_then_layer_then_axis_index"
    || !["row_cosines", "column_cosines", "row_l2_distances", "column_l2_distances"]
      .includes(reference.metric)
    || !Array.isArray(reference.matrix_axis)
    || !Number.isInteger(reference.layer_count)
    || !Number.isInteger(reference.value_count)
    || buffer.byteLength !== reference.value_count * Float32Array.BYTES_PER_ELEMENT
    || !scalarRecord
  ) {
    throw new Error("weight-alignment detail chunk has invalid metadata");
  }
  const allValues = new Float32Array(buffer);
  const details = new Map();
  let offset = 0;
  const cosine = reference.metric.includes("cosine");
  reference.matrix_axis.forEach((weightName) => {
    const componentIndex = scalarRecord.componentAxis.findIndex(
      (component) => component.id === weightName,
    );
    if (componentIndex < 0) {
      throw new Error("weight-alignment detail axis is absent from the scalar chunk");
    }
    for (let layer = 0; layer < reference.layer_count; layer += 1) {
      const column = layer + 1;
      const shape = scalarRecord.shapes[componentIndex]?.[column];
      if (!Array.isArray(shape) || shape.length !== 2) {
        throw new Error("weight-alignment detail chunk lacks a projection shape");
      }
      const length = reference.metric.startsWith("row_") ? shape[0] : shape[1];
      const values = allValues.subarray(offset, offset + length);
      if (
        values.length !== length
        || values.some((value) => (
          !Number.isFinite(value)
          || (cosine ? value < -1 || value > 1 : value < 0)
        ))
      ) {
        throw new Error("weight-alignment detail chunk has invalid values");
      }
      details.set(`${weightName}:${layer}`, { shape, values });
      offset += length;
    }
  });
  if (
    offset !== allValues.length
    || details.size !== reference.matrix_axis.length * reference.layer_count
  ) {
    throw new Error("weight-alignment detail chunk is incomplete");
  }
  return details;
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
      let compact;
      if (reference.kind === "representation_alignment") {
        compact = compactRepresentationAlignmentChunk(records);
      } else if (reference.kind === "weight_alignment") {
        compact = compactWeightAlignmentChunk(records);
      } else {
        compact = compactPatchChunk(records);
      }
      patchChunks.set(
        key,
        compact,
      );
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

async function loadWeightAlignmentDetails(reference, scalarReference) {
  const key = patchReferenceKey(reference);
  if (!key || weightDetailChunks.has(key)) return;
  if (weightDetailLoads.has(key)) {
    await weightDetailLoads.get(key);
    return;
  }
  const scalarKey = patchReferenceKey(scalarReference);
  const scalarRecord = scalarKey ? patchChunks.get(scalarKey) : null;
  if (!scalarRecord || scalarRecord.axisKind !== "weight_layer") return;
  const request = fetch(patchChunkRequest(reference), { cache: "force-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then((buffer) => {
      const pairKey = patchReferenceKey(scalarReference);
      const pairDetails = weightDetailPairs.get(pairKey) ?? new Set();
      weightDetailPairs.delete(pairKey);
      weightDetailPairs.set(pairKey, pairDetails);
      pairDetails.add(key);
      while (weightDetailPairs.size > WEIGHT_DETAIL_PAIR_CACHE_LIMIT) {
        const oldestPairKey = weightDetailPairs.keys().next().value;
        const oldestDetails = weightDetailPairs.get(oldestPairKey);
        oldestDetails?.forEach((detailKey) => weightDetailChunks.delete(detailKey));
        weightDetailPairs.delete(oldestPairKey);
      }
      weightDetailChunks.set(
        key,
        compactWeightAlignmentDetails(buffer, reference, scalarRecord),
      );
      weightDetailErrors.delete(key);
    })
    .catch((error) => {
      weightDetailErrors.set(key, String(error.message ?? error));
    })
    .finally(() => {
      weightDetailLoads.delete(key);
      if (patchReferenceKey(selectedPatchReference()) === scalarKey) {
        window.requestAnimationFrame(refreshVisibleHeatTooltip);
      }
    });
  weightDetailLoads.set(key, request);
  await request;
}

function pumpWeightDetailPreloadQueue() {
  while (
    weightDetailPreloadActive < WEIGHT_DETAIL_PREFETCH_CONCURRENCY
    && weightDetailPreloadQueue.length > 0
  ) {
    const { detailReference, scalarReference } = weightDetailPreloadQueue.shift();
    const key = patchReferenceKey(detailReference);
    if (!key || weightDetailChunks.has(key) || weightDetailLoads.has(key)) continue;
    weightDetailPreloadActive += 1;
    void loadWeightAlignmentDetails(detailReference, scalarReference).finally(() => {
      weightDetailPreloadActive -= 1;
      pumpWeightDetailPreloadQueue();
    });
  }
}

function scheduleSelectedWeightDetailsLoad() {
  if (!weightAnalysisSelected()) return;
  const scalarReference = selectedPatchReference();
  const scalarKey = patchReferenceKey(scalarReference);
  if (!scalarKey) return;
  weightDetailPreloadQueue = weightDetailPreloadQueue.filter((queued) => (
    patchReferenceKey(queued.scalarReference) === scalarKey
  ));
  const cachedPair = weightDetailPairs.get(scalarKey);
  if (cachedPair) {
    weightDetailPairs.delete(scalarKey);
    weightDetailPairs.set(scalarKey, cachedPair);
  }
  const detailReferences = Object.values(scalarReference?.details ?? {});
  detailReferences.forEach((detailReference) => {
    const key = patchReferenceKey(detailReference);
    const cached = key ? weightDetailChunks.get(key) : null;
    if (!cached) return;
    weightDetailChunks.delete(key);
    weightDetailChunks.set(key, cached);
  });
  if (!patchChunks.has(scalarKey)) return;
  detailReferences.forEach((detailReference) => {
    const key = patchReferenceKey(detailReference);
    if (
      !key
      || weightDetailChunks.has(key)
      || weightDetailLoads.has(key)
      || weightDetailPreloadQueue.some((queued) => (
        patchReferenceKey(queued.detailReference) === key
      ))
    ) return;
    weightDetailPreloadQueue.push({ detailReference, scalarReference });
  });
  pumpWeightDetailPreloadQueue();
}

function compactActivationNeighborChunk(payload) {
  if (
    !payload
    || payload.metric !== "cosine_similarity"
    || !Object.hasOwn(ACTIVATION_EXAMPLE_SOURCE_LABELS, payload.candidate_source)
    || !Number.isInteger(payload.position_count)
    || !Number.isInteger(payload.layer_count)
    || !Array.isArray(payload.source_neighbors)
    || !Array.isArray(payload.recipient_neighbors)
  ) {
    throw new Error("activation-neighbor chunk has unsupported metadata");
  }
  const validateGrid = (grid, side) => {
    if (grid.length !== payload.position_count) {
      throw new Error(`activation-neighbor ${side} token axis is incomplete`);
    }
    grid.forEach((layers) => {
      if (!Array.isArray(layers) || layers.length !== payload.layer_count) {
        throw new Error(`activation-neighbor ${side} layer axis is incomplete`);
      }
      layers.forEach((matches) => {
        if (!Array.isArray(matches) || matches.length !== payload.top_k) {
          throw new Error(`activation-neighbor ${side} top-k list is incomplete`);
        }
        matches.forEach((match) => {
          if (
            !Array.isArray(match)
            || match.length !== 3
            || !Number.isInteger(match[0])
            || !Number.isInteger(match[1])
            || !Number.isFinite(match[2])
          ) {
            throw new Error(`activation-neighbor ${side} match is malformed`);
          }
        });
      });
    });
  };
  validateGrid(payload.source_neighbors, "source");
  validateGrid(payload.recipient_neighbors, "recipient");
  return {
    checkpointStep: payload.checkpoint_step,
    candidateSource: payload.candidate_source,
    mode: payload.mode,
    functionId: payload.function_id,
    topK: payload.top_k,
    positionCount: payload.position_count,
    layerCount: payload.layer_count,
    source: payload.source_neighbors,
    recipient: payload.recipient_neighbors,
  };
}

function compactActivationCandidateCatalog(payload) {
  if (
    !payload
    || !Object.hasOwn(ACTIVATION_EXAMPLE_SOURCE_LABELS, payload.candidate_source)
    || !Array.isArray(payload.candidates)
    || payload.candidates.length === 0
  ) {
    throw new Error("activation-example candidate catalog is empty");
  }
  return {
    checkpointStep: payload.checkpoint_step,
    candidateSource: payload.candidate_source,
    corpus: payload.candidate_corpus,
    candidates: payload.candidates.map((candidate) => {
      if (
        !Array.isArray(candidate.token_ids)
        || !Array.isArray(candidate.token_labels)
        || candidate.token_ids.length !== candidate.token_labels.length
      ) {
        throw new Error("activation-example candidate has an invalid token axis");
      }
      const provenance = candidate.provenance ?? null;
      if (
        provenance !== null
        && (
          !Number.isInteger(provenance.row_index)
          || typeof provenance.document_id !== "string"
          || typeof provenance.url !== "string"
          || typeof provenance.text_sha256 !== "string"
        )
      ) {
        throw new Error("activation-example candidate has invalid source provenance");
      }
      return {
        id: candidate.example_id,
        category: candidate.category,
        target: candidate.target,
        renderedPrompt: candidate.rendered_prompt,
        tokenIds: candidate.token_ids,
        tokenLabels: candidate.token_labels,
        provenance,
      };
    }),
  };
}

async function loadActivationExampleReference(reference, kind) {
  const key = patchReferenceKey(reference);
  const cache = kind === "neighbors" ? activationNeighborChunks : activationCandidateCatalogs;
  if (!key || cache.has(key)) return;
  if (activationExampleLoads.has(key)) {
    await activationExampleLoads.get(key);
    return;
  }
  const request = fetch(patchChunkRequest(reference), { cache: "force-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      cache.set(
        key,
        kind === "neighbors"
          ? compactActivationNeighborChunk(payload)
          : compactActivationCandidateCatalog(payload),
      );
      activationExampleErrors.delete(key);
    })
    .catch((error) => {
      activationExampleErrors.set(key, String(error.message ?? error));
    })
    .finally(() => {
      activationExampleLoads.delete(key);
      renderActivationExamples(patchData());
    });
  activationExampleLoads.set(key, request);
  await request;
}

function activationExampleEntry(checkpointStep, functionId) {
  if (
    weightPatchSelected()
    || state.patchMode === "checkpoint"
    || state.patchMode === "across_sample"
    || functionId === ALL_FUNCTIONS_ID
  ) return null;
  return state.data.activation_example_manifest?.[state.model]?.[state.condition]
    ?.[state.patchInterface]?.[state.activationExampleSource]?.[state.patchMode]
    ?.[String(checkpointStep)]?.[functionId] ?? null;
}

function activationExampleEntriesForSelection() {
  if (state.functionId === ALL_FUNCTIONS_ID) return [];
  const sources = state.data.activation_example_manifest?.[state.model]?.[state.condition]
    ?.[state.patchInterface] ?? {};
  return Object.values(sources).flatMap((source) => Object.values(source[state.patchMode] ?? {})
    .map((functions) => functions[state.functionId])
    .filter(Boolean));
}

function scheduleActivationExampleLoads() {
  const checkpoints = state.data.checkpoints;
  const recipientStep = checkpoints[state.recipientIndex];
  const donorStep = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const selectedEntries = [
    activationExampleEntry(recipientStep, state.functionId),
    activationExampleEntry(donorStep, state.functionId),
  ].filter(Boolean);
  const entries = [...selectedEntries, ...activationExampleEntriesForSelection()];
  const references = new Map();
  entries.forEach((entry) => {
    references.set(patchReferenceKey(entry.neighbors), [entry.neighbors, "neighbors"]);
    references.set(patchReferenceKey(entry.candidates), [entry.candidates, "candidates"]);
  });
  references.forEach(([reference, kind], key) => {
    const cache = kind === "neighbors" ? activationNeighborChunks : activationCandidateCatalogs;
    if (key && !cache.has(key) && !activationExampleLoads.has(key)) {
      void loadActivationExampleReference(reference, kind);
    }
  });
}

function compactVocabularyLensChunk(payload) {
  if (
    !payload
    || payload.kind !== "full_vocabulary_top_k"
    || !Number.isInteger(payload.checkpoint_step)
    || typeof payload.function_id !== "string"
    || !Number.isInteger(payload.top_k)
    || payload.top_k <= 0
    || !Number.isInteger(payload.vocabulary_size)
    || payload.vocabulary_size <= payload.top_k
    || !Number.isInteger(payload.layer_count)
    || payload.layer_count <= 0
    || typeof payload.normalization !== "string"
    || !payload.normalization.includes("every model output-embedding row")
    || !payload.token_labels
    || typeof payload.token_labels !== "object"
  ) {
    throw new Error("full-vocabulary logit-lens chunk has unsupported metadata");
  }
  const validateSide = (side, label) => {
    if (
      !side
      || !Number.isInteger(side.position_count)
      || side.position_count <= 0
      || !Array.isArray(side.token_indices)
      || side.token_indices.length !== side.position_count
      || !Array.isArray(side.token_ids)
      || side.token_ids.length !== side.position_count
      || !Array.isArray(side.top_tokens)
      || side.top_tokens.length !== side.position_count
    ) {
      throw new Error(`full-vocabulary ${label} lens has an invalid token axis`);
    }
    side.token_indices.forEach((tokenIndex, reverseIndex) => {
      if (
        !Number.isInteger(tokenIndex)
        || (reverseIndex > 0 && tokenIndex !== side.token_indices[reverseIndex - 1] - 1)
        || !Number.isInteger(side.token_ids[reverseIndex])
      ) {
        throw new Error(`full-vocabulary ${label} lens token axis is not reverse-contiguous`);
      }
    });
    side.top_tokens.forEach((layers) => {
      if (!Array.isArray(layers) || layers.length !== payload.layer_count) {
        throw new Error(`full-vocabulary ${label} lens has an invalid layer axis`);
      }
      layers.forEach((topTokens) => {
        if (!Array.isArray(topTokens) || topTokens.length !== payload.top_k) {
          throw new Error(`full-vocabulary ${label} lens has an invalid top-k axis`);
        }
        let previousProbability = Infinity;
        let displayedMass = 0;
        const seen = new Set();
        topTokens.forEach((entry) => {
          if (
            !Array.isArray(entry)
            || entry.length !== 2
            || !Number.isInteger(entry[0])
            || entry[0] < 0
            || entry[0] >= payload.vocabulary_size
            || !Number.isFinite(entry[1])
            || entry[1] < 0
            || entry[1] > 1
            || seen.has(entry[0])
            || entry[1] > previousProbability + 1e-8
            || typeof payload.token_labels[String(entry[0])] !== "string"
          ) {
            throw new Error(`full-vocabulary ${label} lens contains a malformed top-token list`);
          }
          seen.add(entry[0]);
          previousProbability = entry[1];
          displayedMass += entry[1];
        });
        if (displayedMass > 1.00001) {
          throw new Error(`full-vocabulary ${label} lens displayed mass exceeds one`);
        }
      });
    });
    return {
      positionCount: side.position_count,
      tokenIndices: side.token_indices,
      tokenIds: side.token_ids,
      topTokens: side.top_tokens,
    };
  };
  if (!payload.sources || typeof payload.sources !== "object") {
    throw new Error("full-vocabulary logit-lens chunk lacks prompt sources");
  }
  const sources = {};
  Object.entries(payload.sources).forEach(([mode, side]) => {
    if (!(mode in PROMPT_SOURCE_LABELS)) {
      throw new Error(`full-vocabulary logit-lens chunk has unknown source ${mode}`);
    }
    sources[mode] = validateSide(side, mode);
  });
  return {
    checkpointStep: payload.checkpoint_step,
    functionId: payload.function_id,
    topK: payload.top_k,
    vocabularySize: payload.vocabulary_size,
    layerCount: payload.layer_count,
    normalization: payload.normalization,
    residualBoundary: payload.residual_boundary,
    tokenLabels: payload.token_labels,
    clean: validateSide(payload.clean, "clean"),
    sources,
  };
}

async function loadVocabularyLensReference(reference) {
  const key = patchReferenceKey(reference);
  if (!key || vocabularyLensChunks.has(key)) return;
  if (vocabularyLensLoads.has(key)) {
    await vocabularyLensLoads.get(key);
    return;
  }
  const request = fetch(patchChunkRequest(reference), { cache: "force-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      vocabularyLensChunks.set(key, compactVocabularyLensChunk(payload));
      vocabularyLensErrors.delete(key);
    })
    .catch((error) => {
      vocabularyLensErrors.set(key, String(error.message ?? error));
    })
    .finally(() => {
      vocabularyLensLoads.delete(key);
      renderPatching();
    });
  vocabularyLensLoads.set(key, request);
  await request;
}

function vocabularyLensEntry(checkpointStep, functionId) {
  if (functionId === ALL_FUNCTIONS_ID) return null;
  return state.data.vocabulary_logit_lens_manifest?.[state.model]?.[state.condition]
    ?.[String(checkpointStep)]?.[functionId] ?? null;
}

function vocabularyLensEntriesForSelection() {
  if (state.functionId === ALL_FUNCTIONS_ID) return [];
  const checkpoints = state.data.vocabulary_logit_lens_manifest?.[state.model]?.[state.condition]
    ?? {};
  return Object.values(checkpoints)
    .map((functions) => functions[state.functionId])
    .filter(Boolean);
}

function scheduleVocabularyLensLoads() {
  const checkpoints = state.data.checkpoints;
  const recipientStep = checkpoints[state.recipientIndex];
  const donorStep = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const entries = [
    vocabularyLensEntry(recipientStep, state.functionId),
    vocabularyLensEntry(donorStep, state.functionId),
    ...vocabularyLensEntriesForSelection(),
  ].filter(Boolean);
  const references = new Map();
  entries.forEach((reference) => references.set(patchReferenceKey(reference), reference));
  references.forEach((reference, key) => {
    if (key && !vocabularyLensChunks.has(key) && !vocabularyLensLoads.has(key)) {
      void loadVocabularyLensReference(reference);
    }
  });
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
    const activationManifest = snapshot.activation_example_manifest ?? {};
    const alignmentManifest = snapshot.representation_alignment_manifest ?? {};
    const weightManifest = snapshot.weight_alignment_manifest ?? {};
    const vocabularyLensManifest = snapshot.vocabulary_logit_lens_manifest ?? {};
    const signature = patchManifestKey(
      snapshot.patch_manifest,
      alignmentManifest,
      weightManifest,
      activationManifest,
      vocabularyLensManifest,
    );
    if (signature === patchManifestSignature) return;
    state.data.patch_manifest = snapshot.patch_manifest;
    state.data.real_patch_files = snapshot.real_patch_files;
    state.data.representation_alignment_manifest = alignmentManifest;
    state.data.real_representation_alignment_files =
      snapshot.real_representation_alignment_files ?? 0;
    state.data.representation_alignment_scales =
      snapshot.representation_alignment_scales ?? {};
    state.data.weight_alignment_manifest = weightManifest;
    state.data.real_weight_alignment_files = snapshot.real_weight_alignment_files ?? 0;
    state.data.weight_alignment_scales = snapshot.weight_alignment_scales ?? {};
    state.data.weight_alignment_axes = snapshot.weight_alignment_axes ?? {};
    state.data.activation_example_manifest = activationManifest;
    state.data.real_activation_example_files = snapshot.real_activation_example_files ?? 0;
    state.data.activation_example_chunks = snapshot.activation_example_chunks ?? 0;
    state.data.vocabulary_logit_lens_manifest = vocabularyLensManifest;
    state.data.real_vocabulary_logit_lens_files = snapshot.real_vocabulary_logit_lens_files ?? 0;
    state.data.vocabulary_logit_lens_chunks = snapshot.vocabulary_logit_lens_chunks ?? 0;
    patchManifestSignature = signature;
    patchChunkErrors.clear();
    weightDetailErrors.clear();
    activationExampleErrors.clear();
    vocabularyLensErrors.clear();
    scheduleFullPatchPreload();
    scheduleActivationExampleLoads();
    scheduleVocabularyLensLoads();
    renderPatching();
  } catch (error) {
    console.warn("Could not refresh the patch manifest", error);
  }
}

function tokenAxisMode() {
  return state.patchMode === "checkpoint" ? "across_time" : state.patchMode;
}

function registeredWeightAxis() {
  const axis = state.data.weight_alignment_axes?.[state.model] ?? null;
  if (axis) return axis;
  const decoderLayerCount = state.data.models[state.model].layer_count;
  return {
    component_axis: Object.entries(WEIGHT_MATRIX_LABELS).map(([id, label]) => ({
      id,
      label,
      placement: "layer",
      tensor_rank: 2,
      frozen_during_lora: false,
    })),
    column_axis: Array.from({ length: decoderLayerCount }, (_, layer) => ({
      id: `layer_${layer}`,
      label: String(layer),
      kind: "decoder_layer",
      layer,
    })),
    decoder_layer_count: decoderLayerCount,
  };
}

function weightAxisPositions(componentAxis = registeredWeightAxis().component_axis) {
  return componentAxis.map((component) => ({
    axisKind: "weight_layer",
    weightName: component.id,
    component,
    sourceToken: component.label,
    recipientToken: component.label,
  }));
}

function analyticOrUnprocessedWeightAlignment() {
  const axis = registeredWeightAxis();
  const layers = axis.column_axis.length;
  const recipient = state.data.checkpoints[state.recipientIndex];
  const donor = state.data.checkpoints[state.donorIndex];
  const analytic = recipient === donor;
  const tokenPositions = weightAxisPositions(axis.component_axis);
  const validCell = (component, column) => (
    (component.placement === "input" && column.kind === "global_input")
    || (component.placement === "output" && column.kind === "global_output")
    || (component.placement === "layer" && column.kind === "decoder_layer")
  );
  const metricMatrices = {};
  Object.values(WEIGHT_VISUALIZATION_METRICS).forEach((metric) => {
    const identity = metric.includes("cosine") ? 1.0 : 0.0;
    metricMatrices[metric] = tokenPositions.map(({ component }) => (
      axis.column_axis.map((column) => (
        analytic
        && validCell(component, column)
        && (component.tensor_rank === 2 || metric.startsWith("frobenius_"))
          ? identity
          : null
      ))
    ));
  });
  const varianceMatrices = {};
  Object.values(WEIGHT_VISUALIZATION_VARIANCES).forEach((metric) => {
    varianceMatrices[metric] = tokenPositions.map(({ component }) => (
      axis.column_axis.map((column) => (
        analytic && component.tensor_rank === 2 && validCell(component, column) ? 0.0 : null
      ))
    ));
  });
  const weightShapes = tokenPositions.map(({ component }) => (
    axis.column_axis.map((column) => (
      validCell(component, column) ? component.shape ?? null : null
    ))
  ));
  const selectedMetric = WEIGHT_VISUALIZATION_METRICS[state.patchVisualization];
  return {
    layers,
    tokenPositions,
    recipient: null,
    source: null,
    sourceTargetRecipient: null,
    sourceTargetSource: null,
    sourceCorrectIndex: null,
    recipientCorrectIndex: null,
    matrix: metricMatrices[selectedMetric],
    weightMetricMatrices: metricMatrices,
    weightVarianceMatrices: varianceMatrices,
    weightShapes,
    weightDegenerateCounts: null,
    weightColumnAxis: axis.column_axis,
    sourceTargetMatrix: null,
    target: "complete effective learned-weight atlas",
    outcomeLabel: PATCH_VISUALIZATION_LABELS[state.patchVisualization],
    sourceFunctionId: null,
    recipientFunctionId: null,
    sourceRenderedPrompt: "Weight comparisons are prompt-independent.",
    recipientRenderedPrompt: "Weight comparisons are prompt-independent.",
    sourceChoiceFunctionIds: null,
    sourceChoiceTexts: null,
    sourceQuestionId: null,
    sourceQuestion: null,
    sourceFormat: null,
    sourceLabelRelation: null,
    sourceContextId: null,
    sourceContext: null,
    answerLogitLens: null,
    cosineMatrix: null,
    l2Matrix: null,
    sourceNormMatrix: null,
    recipientNormMatrix: null,
    measurementKind: "weight_alignment",
    analytic,
    measured: false,
    processed: analytic,
    applicable: true,
    axisKind: "weight_layer",
    aggregate: false,
    functionCount: 0,
  };
}

function measuredWeightAlignment() {
  const key = patchReferenceKey(selectedPatchReference());
  const record = key ? patchChunks.get(key) : null;
  if (!record || record.axisKind !== "weight_layer") return null;
  if (record.decoderLayerCount !== state.data.models[state.model].layer_count) {
    throw new Error("Measured weight-alignment grid has the wrong decoder-layer count");
  }
  const matrixRows = (values) => record.componentAxis.map((_, weightIndex) => (
    values.subarray(
      weightIndex * record.layerCount,
      (weightIndex + 1) * record.layerCount,
    )
  ));
  const metricMatrices = {};
  Object.entries(record.metrics).forEach(([metric, values]) => {
    metricMatrices[metric] = matrixRows(values);
  });
  const selectedMetric = WEIGHT_VISUALIZATION_METRICS[state.patchVisualization];
  if (!metricMatrices[selectedMetric]) {
    throw new Error(`Measured weight-alignment chunk lacks ${selectedMetric}`);
  }
  return {
    ...analyticOrUnprocessedWeightAlignment(),
    layers: record.layerCount,
    tokenPositions: weightAxisPositions(record.componentAxis),
    matrix: metricMatrices[selectedMetric],
    weightMetricMatrices: metricMatrices,
    weightVarianceMatrices: Object.fromEntries(
      Object.entries(record.variances).map(([metric, values]) => [metric, matrixRows(values)]),
    ),
    weightShapes: record.shapes,
    weightColumnAxis: record.columnAxis,
    weightDegenerateCounts: Object.fromEntries(
      Object.entries(record.degenerateCounts).map(([metric, values]) => [metric, matrixRows(values)]),
    ),
    analytic: false,
    measured: true,
    processed: true,
  };
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
    sourceTargetRecipient: null,
    sourceTargetSource: null,
    sourceCorrectIndex: exactAxis?.source_correct_choice_index ?? null,
    recipientCorrectIndex: exactAxis?.recipient_correct_choice_index ?? null,
    matrix: tokenPositions.map(() => Array(layers).fill(null)),
    sourceTargetMatrix: null,
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
    sourceChoiceFunctionIds: exactAxis?.source_choice_function_ids ?? null,
    sourceChoiceTexts: exactAxis?.source_choice_texts ?? null,
    sourceQuestionId: exactAxis?.source_question_id ?? null,
    sourceQuestion: exactAxis?.source_question ?? null,
    sourceFormat: exactAxis?.source_format ?? null,
    sourceLabelRelation: exactAxis?.source_label_relation ?? null,
    sourceContextId: exactAxis?.source_context_id ?? null,
    sourceContext: exactAxis?.source_context ?? null,
    answerLogitLens: null,
    cosineMatrix: null,
    l2Matrix: null,
    sourceNormMatrix: null,
    recipientNormMatrix: null,
    measurementKind: representationAlignmentSelected()
      ? "representation_alignment"
      : "activation_patching",
    analytic: false,
    measured: false,
    processed: false,
    applicable: patchSelectionApplicable(),
    axisKind,
    aggregate: false,
    functionCount: 1,
  };
}

function measuredRepresentationAlignmentForFunction(functionId) {
  const key = patchReferenceKey(selectedPatchReference());
  const records = key ? patchChunks.get(key) : null;
  const record = records?.[functionId] ?? null;
  if (!record) return null;
  if (record.layerCount !== state.data.models[state.model].layer_count) {
    throw new Error("Measured alignment grid has the wrong decoder-layer count");
  }
  if (!Array.isArray(record.tokenPositions) || record.tokenPositions.length !== record.tokenCount) {
    throw new Error("Measured alignment grid lacks its exact tokenizer axis");
  }
  const matrixRows = (values) => Array.from({ length: record.tokenCount }, (_, tokenIndex) => (
    values.subarray(
      tokenIndex * record.layerCount,
      (tokenIndex + 1) * record.layerCount,
    )
  ));
  const cosineMatrix = matrixRows(record.cosineSimilarities);
  const l2Matrix = matrixRows(record.l2Distances);
  const sourceNormMatrix = matrixRows(record.sourceNorms);
  const recipientNormMatrix = matrixRows(record.recipientNorms);
  const tokenPositions = record.tokenPositions.map((position) => ({
    reverseIndex: position.reverse_index,
    sourceIndex: position.source_index,
    recipientIndex: position.recipient_index,
    sourceTokenId: position.source_token_id,
    recipientTokenId: position.recipient_token_id,
    sourceToken: position.source_token,
    recipientToken: position.recipient_token,
  }));
  const fn = state.data.functions.find((item) => item.id === functionId);
  return {
    layers: record.layerCount,
    tokenPositions,
    recipient: null,
    source: null,
    sourceTargetRecipient: null,
    sourceTargetSource: null,
    sourceCorrectIndex: record.sourceCorrectIndex,
    recipientCorrectIndex: record.recipientCorrectIndex,
    matrix: state.patchVisualization === "cosine_similarity" ? cosineMatrix : l2Matrix,
    cosineMatrix,
    l2Matrix,
    sourceNormMatrix,
    recipientNormMatrix,
    sourceTargetMatrix: null,
    target: fn.definition,
    outcomeLabel: state.patchVisualization === "cosine_similarity"
      ? "donor/recipient cosine similarity"
      : "donor/recipient raw L2 distance",
    sourceFunctionId: record.sourceFunctionId,
    recipientFunctionId: record.recipientFunctionId,
    sourceRenderedPrompt: record.sourceRenderedPrompt
      ?? "The measured alignment artifact lacks a rendered source prompt.",
    recipientRenderedPrompt: record.recipientRenderedPrompt
      ?? "The measured alignment artifact lacks a rendered recipient prompt.",
    sourceChoiceFunctionIds: record.sourceChoiceFunctionIds,
    sourceChoiceTexts: record.sourceChoiceTexts,
    sourceQuestionId: record.sourceQuestionId,
    sourceQuestion: record.sourceQuestion,
    sourceFormat: record.sourceFormat,
    sourceLabelRelation: record.sourceLabelRelation,
    sourceContextId: record.sourceContextId,
    sourceContext: record.sourceContext,
    answerLogitLens: null,
    measurementKind: "representation_alignment",
    analytic: false,
    measured: true,
    processed: true,
    applicable: true,
    axisKind: "token_layer",
    aggregate: false,
    functionCount: 1,
  };
}

function analyticIdentityAlignmentForFunction(functionId) {
  if (
    !representationAlignmentSelected()
    || !patchSelectionApplicable()
    || state.patchMode !== "checkpoint"
    || state.recipientIndex !== state.donorIndex
  ) return null;
  const identity = unprocessedPatchForFunction(functionId);
  const cosineMatrix = identity.tokenPositions.map(() => Array(identity.layers).fill(1));
  const l2Matrix = identity.tokenPositions.map(() => Array(identity.layers).fill(0));
  return {
    ...identity,
    matrix: state.patchVisualization === "cosine_similarity" ? cosineMatrix : l2Matrix,
    cosineMatrix,
    l2Matrix,
    outcomeLabel: state.patchVisualization === "cosine_similarity"
      ? "exact identity cosine similarity"
      : "exact identity L2 distance",
    measurementKind: "representation_alignment",
    analytic: true,
    processed: true,
    applicable: true,
  };
}

function measuredPatchForFunction(functionId) {
  if (representationAlignmentSelected()) {
    return measuredRepresentationAlignmentForFunction(functionId);
  }
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
  const sourceTargetMatrix = record.sourceTargetProbabilities
    ? Array.from({ length: record.tokenCount }, (_, tokenIndex) => (
      record.sourceTargetProbabilities.subarray(
        tokenIndex * record.layerCount,
        (tokenIndex + 1) * record.layerCount,
      )
    ))
    : null;
  return {
    layers: record.layerCount,
    tokenPositions,
    recipient: record.recipient,
    source: record.source,
    sourceTargetRecipient: record.sourceTargetRecipient,
    sourceTargetSource: record.sourceTargetSource,
    sourceCorrectIndex: record.sourceCorrectIndex,
    recipientCorrectIndex: record.recipientCorrectIndex,
    matrix,
    sourceTargetMatrix,
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
    sourceChoiceFunctionIds: record.sourceChoiceFunctionIds ?? exactAxis?.source_choice_function_ids ?? null,
    sourceChoiceTexts: record.sourceChoiceTexts ?? exactAxis?.source_choice_texts ?? null,
    sourceQuestionId: record.sourceQuestionId ?? exactAxis?.source_question_id ?? null,
    sourceQuestion: record.sourceQuestion ?? exactAxis?.source_question ?? null,
    sourceFormat: record.sourceFormat ?? exactAxis?.source_format ?? null,
    sourceLabelRelation: record.sourceLabelRelation ?? exactAxis?.source_label_relation ?? null,
    sourceContextId: record.sourceContextId ?? exactAxis?.source_context_id ?? null,
    sourceContext: record.sourceContext ?? exactAxis?.source_context ?? null,
    answerLogitLens: record.answerLogitLens,
    cosineMatrix: null,
    l2Matrix: null,
    sourceNormMatrix: null,
    recipientNormMatrix: null,
    measurementKind: "activation_patching",
    analytic: false,
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
  const alignment = patches.every(
    (patch) => patch.measurementKind === "representation_alignment",
  );
  if (
    !alignment
    && patches.some((patch) => patch.measurementKind === "representation_alignment")
  ) {
    throw new Error("Cannot average causal-patching and representation-alignment grids");
  }
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
  const averageOptionalMatrix = (key) => (
    processed && patches.every((patch) => Array.isArray(patch[key]))
      ? Array.from({ length: sharedTokenCount }, (_, tokenIndex) => (
        Array.from({ length: layers }, (_, layer) => (
          mean(patches.map((patch) => patch[key][tokenIndex][layer]))
        ))
      ))
      : null
  );
  const cosineMatrix = averageOptionalMatrix("cosineMatrix");
  const l2Matrix = averageOptionalMatrix("l2Matrix");
  const sourceNormMatrix = averageOptionalMatrix("sourceNormMatrix");
  const recipientNormMatrix = averageOptionalMatrix("recipientNormMatrix");
  const hasSourceTarget = processed && patches.every(
    (patch) => Array.isArray(patch.sourceTargetMatrix),
  );
  const sourceTargetMatrix = hasSourceTarget
    ? Array.from({ length: sharedTokenCount }, (_, tokenIndex) => (
      Array.from({ length: layers }, (_, layer) => (
        mean(patches.map((patch) => patch.sourceTargetMatrix[tokenIndex][layer]))
      ))
    ))
    : null;
  const hasAnswerLogitLens = processed && patches.every(
    (patch) => patch.answerLogitLens !== null,
  );
  let answerLogitLens = null;
  if (hasAnswerLogitLens) {
    const referenceLens = patches[0].answerLogitLens;
    const flatLength = sharedTokenCount * layers * 5;
    if (patches.some((patch) => (
      patch.answerLogitLens.source.length < flatLength
      || patch.answerLogitLens.recipient.length < flatLength
      || patch.answerLogitLens.topP !== referenceLens.topP
    ))) {
      throw new Error("Cannot average incompatible answer-logit-lens artifacts");
    }
    const averageFlat = (side) => Float64Array.from(
      { length: flatLength },
      (_, index) => mean(patches.map((patch) => patch.answerLogitLens[side][index])),
    );
    answerLogitLens = {
      ...referenceLens,
      source: averageFlat("source"),
      recipient: averageFlat("recipient"),
    };
  }
  const measured = processed && patches.every((patch) => patch.measured);
  return {
    layers,
    tokenPositions,
    recipient: processed && !alignment
      ? mean(patches.map((patch) => patch.recipient))
      : null,
    source: processed && !alignment ? mean(patches.map((patch) => patch.source)) : null,
    sourceTargetRecipient: hasSourceTarget
      ? mean(patches.map((patch) => patch.sourceTargetRecipient))
      : null,
    sourceTargetSource: hasSourceTarget
      ? mean(patches.map((patch) => patch.sourceTargetSource))
      : null,
    sourceCorrectIndex: null,
    recipientCorrectIndex: null,
    matrix,
    cosineMatrix,
    l2Matrix,
    sourceNormMatrix,
    recipientNormMatrix,
    sourceTargetMatrix,
    target: `${functionCount}-function mean`,
    outcomeLabel: alignment
      ? `mean ${state.patchVisualization.replaceAll("_", " ")}`
      : "mean correct-implementation probability",
    sourceFunctionId: null,
    recipientFunctionId: null,
    sourceRenderedPrompt: `Aggregate view over ${functionCount} model-rendered source prompts. Select an individual function to inspect exact text and tokenizer IDs.`,
    recipientRenderedPrompt: `Aggregate view over ${functionCount} model-rendered recipient prompts. Select an individual function to inspect exact text and tokenizer IDs.`,
    sourceChoiceFunctionIds: null,
    sourceChoiceTexts: null,
    sourceQuestionId: null,
    sourceQuestion: null,
    sourceFormat: patches.every((patch) => patch.sourceFormat === patches[0].sourceFormat)
      ? patches[0].sourceFormat
      : null,
    sourceLabelRelation: patches.every(
      (patch) => patch.sourceLabelRelation === patches[0].sourceLabelRelation,
    )
      ? patches[0].sourceLabelRelation
      : null,
    sourceContextId: null,
    sourceContext: null,
    answerLogitLens,
    measurementKind: alignment ? "representation_alignment" : "activation_patching",
    analytic: alignment && patches.every((patch) => patch.analytic),
    measured,
    processed,
    applicable,
    axisKind,
    aggregate: true,
    functionCount,
  };
}

function unprocessedPatch() {
  if (weightAnalysisSelected()) return analyticOrUnprocessedWeightAlignment();
  const functionIds = state.functionId === ALL_FUNCTIONS_ID
    ? state.data.functions.map((fn) => fn.id)
    : [state.functionId];
  const patches = functionIds.map((functionId) => unprocessedPatchForFunction(functionId));
  return patches.length === 1 ? patches[0] : averagePatches(patches);
}

function measuredPatch() {
  if (weightAnalysisSelected()) return measuredWeightAlignment();
  const functionIds = state.functionId === ALL_FUNCTIONS_ID
    ? state.data.functions.map((fn) => fn.id)
    : [state.functionId];
  const patches = functionIds.map((functionId) => (
    measuredPatchForFunction(functionId) ?? analyticIdentityAlignmentForFunction(functionId)
  ));
  if (patches.some((patch) => patch === null)) return null;
  return patches.length === 1 ? patches[0] : averagePatches(patches);
}

function patchData() {
  return measuredPatch() ?? unprocessedPatch();
}

function representationAlignmentScale() {
  return state.data.representation_alignment_scales?.[state.model]
    ?.[state.patchInterface]?.l2_distance ?? null;
}

function weightAlignmentScale() {
  const metric = WEIGHT_VISUALIZATION_METRICS[state.patchVisualization];
  return state.data.weight_alignment_scales?.[state.model]?.[metric] ?? null;
}

function weightVarianceScale() {
  const metric = WEIGHT_VISUALIZATION_VARIANCES[state.patchVisualization];
  return metric
    ? state.data.weight_alignment_scales?.[state.model]?.variances?.[metric] ?? null
    : null;
}

function colorFor(value, metric, scaleMax = null) {
  if (metric === "probability") {
    const amount = Math.max(0, Math.min(1, value));
    return `rgb(${Math.round(67 + amount * 172)}, ${Math.round(89 + amount * 103)}, ${Math.round(81 - amount * 20)})`;
  }
  if (metric === "l2_distance" || metric.includes("l2")) {
    if (!Number.isFinite(scaleMax) || scaleMax <= 0) {
      if (value !== 0) throw new Error("Measured L2 grid lacks a positive display scale");
      scaleMax = 1;
    }
    const amount = Math.max(0, Math.min(1, value / scaleMax));
    const near = [238, 232, 216];
    const far = [103, 75, 147];
    return `rgb(${near.map((channel, index) => Math.round(channel + (far[index] - channel) * amount)).join(",")})`;
  }
  if (metric.includes("cosine") && metric !== "cosine_similarity") {
    const clamped = Math.max(0, Math.min(1, value));
    const amount = clamped ** 2;
    const unaligned = [55, 92, 170];
    const midpoint = [255, 255, 255];
    const aligned = [239, 119, 95];
    const start = amount <= .5 ? unaligned : midpoint;
    const end = amount <= .5 ? midpoint : aligned;
    const segment = amount <= .5 ? amount * 2 : (amount - .5) * 2;
    return `rgb(${start.map((channel, index) => Math.round(channel + (end[index] - channel) * segment)).join(",")})`;
  }
  const clipped = Math.max(-1, Math.min(1, value));
  const neutral = [238, 232, 216];
  const endpoint = clipped >= 0 ? [239, 119, 95] : [93, 121, 185];
  const amount = Math.abs(clipped);
  return `rgb(${neutral.map((channel, index) => Math.round(channel + (endpoint[index] - channel) * amount)).join(",")})`;
}

function formatAlignmentValue(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .001)) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

function positionHeatTooltip(tooltip, point) {
  const left = Math.min(window.innerWidth - tooltip.offsetWidth - 8, point.clientX + 14);
  const top = Math.min(window.innerHeight - tooltip.offsetHeight - 8, point.clientY + 14);
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function showHeatTooltip(html, point) {
  const tooltip = document.getElementById("tooltip");
  tooltip.innerHTML = typeof html === "function" ? html() : html;
  renderWeightDetailCanvases(tooltip);
  tooltip.hidden = false;
  positionHeatTooltip(tooltip, point);
}

function refreshVisibleHeatTooltip() {
  const hovered = document.querySelector(".heat-cell:hover");
  if (hovered?.heatTooltipHtml && state.patchTooltipHoverPosition) {
    showHeatTooltip(hovered.heatTooltipHtml, state.patchTooltipHoverPosition);
    return;
  }
  restorePinnedHeatTooltip();
}

function restorePinnedHeatTooltip() {
  if (!state.patchTooltipPinned || !state.patchTooltipPosition) return;
  const selected = document.querySelector(
    `.heat-cell[data-token-index="${state.patchCellTokenIndex}"]`
    + `[data-layer="${state.patchCellLayer}"]`,
  );
  if (!selected?.heatTooltipHtml) return;
  showHeatTooltip(selected.heatTooltipHtml, state.patchTooltipPosition);
}

function bindHeatTooltip(cell, html) {
  const tooltip = document.getElementById("tooltip");
  cell.heatTooltipHtml = html;
  const show = (event) => {
    state.patchTooltipHoverPosition = { clientX: event.clientX, clientY: event.clientY };
    showHeatTooltip(html, event);
  };
  cell.addEventListener("mouseenter", show);
  cell.addEventListener("mousemove", (event) => {
    state.patchTooltipHoverPosition = { clientX: event.clientX, clientY: event.clientY };
    if (!tooltip.hidden) positionHeatTooltip(tooltip, event);
  });
  cell.addEventListener("focus", () => {
    const selected = state.patchTooltipPinned
      && Number(cell.dataset.tokenIndex) === state.patchCellTokenIndex
      && Number(cell.dataset.layer) === state.patchCellLayer;
    showHeatTooltip(
      html,
      selected && state.patchTooltipPosition
        ? state.patchTooltipPosition
        : { clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 },
    );
  });
  cell.addEventListener("mouseleave", () => {
    state.patchTooltipHoverPosition = null;
    if (!state.patchTooltipPinned) tooltip.hidden = true;
  });
  cell.addEventListener("blur", () => {
    if (!state.patchTooltipPinned) {
      tooltip.hidden = true;
      return;
    }
    window.requestAnimationFrame(() => {
      if (!document.activeElement?.classList.contains("heat-cell")) {
        restorePinnedHeatTooltip();
      }
    });
  });
}

function renderWeightDetailCanvases(container) {
  container.querySelectorAll("canvas.weight-detail-grid").forEach((canvas) => {
    const cachedValues = weightDetailCells.get(canvas.dataset.cacheKey);
    const details = weightDetailChunks.get(canvas.dataset.detailKey);
    const detail = details?.get(canvas.dataset.cellKey);
    const values = cachedValues ?? detail?.values;
    if (!values) return;
    const cosine = canvas.dataset.cosine === "true";
    const scale = Number(canvas.dataset.scale);
    const groupSize = Number(canvas.dataset.groupSize) || 0;
    const context = canvas.getContext("2d");
    if (!context) return;
    if (groupSize > 0) {
      const columns = 64;
      const groupCount = Math.ceil(values.length / groupSize);
      const cellSize = 3;
      canvas.width = Math.min(columns, values.length) * cellSize;
      canvas.height = Math.ceil(values.length / columns) * cellSize;
      values.forEach((value, index) => {
        context.fillStyle = colorFor(
          value,
          cosine ? "weight_detail_cosine" : "l2_distance",
          scale,
        );
        context.fillRect(
          (index % columns) * cellSize,
          Math.floor(index / columns) * cellSize,
          cellSize,
          cellSize,
        );
      });
      context.strokeStyle = "rgba(204, 255, 0, .95)";
      context.lineWidth = .35;
      for (let group = 0; group < groupCount; group += 1) {
        const start = group * groupSize;
        const end = Math.min(start + groupSize, values.length);
        const startRow = Math.floor(start / columns);
        const endRow = Math.ceil(end / columns);
        context.strokeRect(
          .5,
          startRow * cellSize + .5,
          canvas.width - 1,
          (endRow - startRow) * cellSize - 1,
        );
      }
    } else {
      const columns = values.length >= 8000 ? 192 : values.length >= 4000 ? 128 : 64;
      const cellSize = values.length >= 4000 ? 2 : 3;
      canvas.width = Math.min(columns, values.length) * cellSize;
      canvas.height = Math.ceil(values.length / columns) * cellSize;
      values.forEach((value, index) => {
        context.fillStyle = colorFor(
          value,
          cosine ? "weight_detail_cosine" : "l2_distance",
          scale,
        );
        context.fillRect(
          (index % columns) * cellSize,
          Math.floor(index / columns) * cellSize,
          cellSize,
          cellSize,
        );
      });
    }
  });
}

function weightDetailCellCacheKey(detailKey, cellKey) {
  return `${detailKey}/${cellKey}`;
}

function weightDetailGridHtml(patch, weightIndex, columnIndex) {
  const detailSpec = WEIGHT_DETAIL_METRICS[state.patchVisualization];
  if (!detailSpec) {
    return "<br><small>Frobenius metrics flatten the complete matrix and therefore have no per-axis decomposition.</small>";
  }
  const position = patch.tokenPositions[weightIndex];
  const component = position.component;
  if (component.tensor_rank !== 2) {
    return "<br><small>Row/column decomposition is not defined for this one-dimensional norm vector; use a Frobenius view.</small>";
  }
  if (component.frozen_during_lora || patch.analytic) {
    const identity = detailSpec.artifact.includes("cosine") ? "1.0000" : "0.0000";
    return `<br><small>Exact analytic identity: every applicable per-channel value is ${identity}; no redundant detail payload is materialized.</small>`;
  }
  const reference = selectedPatchReference();
  const detailReference = reference?.details?.[detailSpec.artifact];
  const detailKey = patchReferenceKey(detailReference);
  if (!detailKey) return "<br><small>No measured per-channel detail artifact exists.</small>";
  const column = patch.weightColumnAxis[columnIndex];
  if (column?.kind !== "decoder_layer") {
    return "<br><small>This learned projection exists only at decoder-layer columns.</small>";
  }
  const weightName = position.weightName;
  const cellKey = `${weightName}:${column.layer}`;
  const cacheKey = weightDetailCellCacheKey(detailKey, cellKey);
  let values = weightDetailCells.get(cacheKey);
  if (!values) {
    if (weightDetailErrors.has(detailKey)) {
      return `<br><small>Per-channel detail failed to load: ${escapeHtml(weightDetailErrors.get(detailKey))}</small>`;
    }
    const details = weightDetailChunks.get(detailKey);
    if (!details) {
      scheduleSelectedWeightDetailsLoad();
      return "<br><small>This pair's packed detail views are loading. Grids already opened during this browser session remain cached individually.</small>";
    }
    weightDetailChunks.delete(detailKey);
    weightDetailChunks.set(detailKey, details);
    const detail = details.get(cellKey);
    const detailValues = detail?.[detailSpec.compact];
    if (!detailValues) return "<br><small>The measured detail file lacks this axis.</small>";
    values = new Float32Array(detailValues);
    weightDetailCells.set(cacheKey, values);
  }
  if (!values) return "<br><small>The measured detail file lacks this axis.</small>";
  const sorted = Array.from(values).sort((left, right) => left - right);
  const quantile = (fraction) => sorted[Math.round((sorted.length - 1) * fraction)];
  const cosine = detailSpec.artifact.includes("cosine");
  const detailScale = cosine ? null : Math.max(...values, Number.EPSILON);
  const axis = detailSpec.artifact.startsWith("row_")
    ? "row / output channel"
    : "column / input channel";
  const groupSize = detailSpec.artifact.startsWith("row_")
    ? component.row_group_size
    : component.column_group_size;
  const groupNote = groupSize
    ? ` · <small class="weight-detail-group-note">outlined band = one ${escapeHtml(component.group_label)} (${groupSize} channels)</small>`
    : "";
  return `<br><br><b>${axis} · n=${values.length.toLocaleString()}</b>${groupNote}<canvas class="weight-detail-grid" data-cache-key="${escapeHtml(cacheKey)}" data-detail-key="${escapeHtml(detailKey)}" data-cell-key="${escapeHtml(cellKey)}" data-cosine="${cosine}" data-scale="${detailScale ?? 1}" data-group-size="${groupSize ?? 0}"></canvas><small>min ${formatAlignmentValue(sorted[0])} · median ${formatAlignmentValue(quantile(.5))} · p95 ${formatAlignmentValue(quantile(.95))} · max ${formatAlignmentValue(sorted.at(-1))}</small>`;
}

function tokenCoordinate(prefix, index, tokenId, token) {
  const position = Number.isInteger(index) ? index : "?";
  const id = Number.isInteger(tokenId) ? tokenId : "?";
  return `${prefix}[position ${position} · id ${id}] ${token}`;
}

function aggregateTokenCoordinate(prefix, token) {
  return `${prefix}${token}`;
}

function promptSourcePrefix() {
  if (state.patchMode === "across_sample") return "dirty/source ";
  if (state.patchMode === "cyclic_choices") return "shifted/source ";
  if (state.patchMode === "deranged_choices") return "deranged/source ";
  if (["unrelated_question", "unrelated_question_same_letter"].includes(state.patchMode)) {
    return "unrelated/source ";
  }
  if (["letter_context_same", "letter_context_different"].includes(state.patchMode)) {
    return "non-MCQ/source ";
  }
  return "donor/source ";
}

function formatLensProbability(probability) {
  if (probability >= 0.001) return `${(probability * 100).toFixed(2)}%`;
  if (probability >= 0.000001) return `${(probability * 100).toFixed(4)}%`;
  return probability.toExponential(2);
}

function fullVocabularyLogitLensHtml(patch, tokenIndex, layer) {
  const heading = "full-vocabulary residual logit lens";
  if (patch.aggregate) {
    return `<br><br><b>${heading}</b><br><small>Select an individual function; sparse top-k lists cannot be averaged without the omitted vocabulary probabilities.</small>`;
  }
  if (patch.axisKind === "layer_only") {
    return `<br><br><b>${heading}</b><br><small>Not defined for the all-token block-weight row; select a token-local patch boundary.</small>`;
  }
  const checkpoints = state.data.checkpoints;
  const recipientStep = checkpoints[state.recipientIndex];
  const sourceStep = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const recipientReference = vocabularyLensEntry(recipientStep, state.functionId);
  const sourceReference = vocabularyLensEntry(sourceStep, state.functionId);
  if (!recipientReference || !sourceReference) {
    return `<br><br><b>${heading}</b><br><small>Unprocessed for this checkpoint/function; no A–E-only fallback is shown.</small>`;
  }
  const recipientKey = patchReferenceKey(recipientReference);
  const sourceKey = patchReferenceKey(sourceReference);
  const error = vocabularyLensErrors.get(recipientKey) ?? vocabularyLensErrors.get(sourceKey);
  if (error) {
    return `<br><br><b>${heading}</b><br><small>Measured sidecar failed to load: ${escapeHtml(error)}</small>`;
  }
  const recipientChunk = vocabularyLensChunks.get(recipientKey);
  const sourceChunk = vocabularyLensChunks.get(sourceKey);
  if (!recipientChunk || !sourceChunk) {
    return `<br><br><b>${heading}</b><br><small>Loading checkpoint-indexed full-vocabulary readouts…</small>`;
  }
  if (
    recipientChunk.functionId !== state.functionId
    || sourceChunk.functionId !== state.functionId
    || recipientChunk.checkpointStep !== recipientStep
    || sourceChunk.checkpointStep !== sourceStep
    || recipientChunk.layerCount !== patch.layers
    || sourceChunk.layerCount !== patch.layers
    || recipientChunk.topK !== sourceChunk.topK
    || recipientChunk.vocabularySize !== sourceChunk.vocabularySize
  ) {
    throw new Error("full-vocabulary logit lens does not match the selected patch grid");
  }
  const sourceSide = state.patchMode === "checkpoint"
    ? sourceChunk.clean
    : sourceChunk.sources[state.patchMode];
  const recipientSide = recipientChunk.clean;
  if (!sourceSide) {
    return `<br><br><b>${heading}</b><br><small>Unprocessed for this patch source; no A–E-only fallback is shown.</small>`;
  }
  if (
    tokenIndex >= sourceSide.positionCount
    || tokenIndex >= recipientSide.positionCount
    || layer >= sourceChunk.layerCount
  ) {
    throw new Error("full-vocabulary logit lens does not cover the selected token/layer cell");
  }
  const position = patch.tokenPositions[tokenIndex];
  if (
    Number.isInteger(position.sourceIndex)
    && (
      sourceSide.tokenIndices[tokenIndex] !== position.sourceIndex
      || sourceSide.tokenIds[tokenIndex] !== position.sourceTokenId
      || recipientSide.tokenIndices[tokenIndex] !== position.recipientIndex
      || recipientSide.tokenIds[tokenIndex] !== position.recipientTokenId
    )
  ) {
    throw new Error("full-vocabulary logit-lens token coordinates disagree with the patch axis");
  }
  const formatTopTokens = (chunk, side) => {
    const entries = side.topTokens[tokenIndex][layer];
    const displayedMass = entries.reduce((total, entry) => total + entry[1], 0);
    const tokens = entries.map(([tokenId, probability]) => {
      const label = chunk.tokenLabels[String(tokenId)];
      return `${escapeHtml(label)} <small>[${tokenId}]</small> ${formatLensProbability(probability)}`;
    }).join(" · ");
    return `${tokens}<br><small>top-${chunk.topK} displayed mass ${formatPercent(displayedMass)}</small>`;
  };
  return `<br><br><b>${heading} · top-${sourceChunk.topK}</b><br><small>checkpoint final norm + unembedding; each probability is normalized over all ${sourceChunk.vocabularySize.toLocaleString()} output tokens; observational, not a patched forward pass</small><br>${escapeHtml(promptSourcePrefix().trim())}: ${formatTopTokens(sourceChunk, sourceSide)}<br>clean/recipient: ${formatTopTokens(recipientChunk, recipientSide)}`;
}

function renderActivationExampleList(container, matches, catalog) {
  container.replaceChildren();
  matches.forEach(([exampleIndex, tokenIndex, score], rank) => {
    const candidate = catalog.candidates[exampleIndex];
    if (!candidate || !Number.isInteger(tokenIndex) || !candidate.tokenLabels[tokenIndex]) {
      throw new Error("activation-example match references an unavailable candidate token");
    }
    const article = el("article", { class: "activation-example" });
    const meta = el("div", { class: "activation-example-meta" });
    if (candidate.provenance) {
      let host = candidate.provenance.url;
      try {
        host = new URL(candidate.provenance.url).hostname || candidate.provenance.url;
      } catch {
        // Retain the recorded source URL verbatim when it is not parseable.
      }
      meta.append(el("span", {}, `${rank + 1} · FineWeb row ${candidate.provenance.row_index}`));
      meta.append(el("span", {}, `cos ${score.toFixed(3)} · ${host}`));
    } else {
      meta.append(el("span", {}, `${rank + 1} · ${candidate.category.replaceAll("_", " ")}`));
      meta.append(el("span", {}, `cos ${score.toFixed(3)} · target ${candidate.target}`));
    }
    article.append(meta);
    const tokens = el("div", {
      class: "activation-example-tokens",
      title: [
        candidate.id,
        `token ${tokenIndex}`,
        `id ${candidate.tokenIds[tokenIndex]}`,
        candidate.provenance?.url,
      ].filter(Boolean).join(" · "),
    });
    const windowStart = Math.max(0, tokenIndex - 11);
    const windowEnd = Math.min(candidate.tokenLabels.length, tokenIndex + 7);
    if (windowStart > 0) tokens.append(el("span", { class: "ellipsis" }, "… "));
    for (let index = windowStart; index < windowEnd; index += 1) {
      const node = el(index === tokenIndex ? "mark" : "span", {}, candidate.tokenLabels[index]);
      tokens.append(node);
    }
    if (windowEnd < candidate.tokenLabels.length) {
      tokens.append(el("span", { class: "ellipsis" }, " …"));
    }
    article.append(tokens);
    container.append(article);
  });
}

function setActivationExamplesEmpty(message) {
  ["recipient-neighbor-examples", "source-neighbor-examples"].forEach((id) => {
    const container = document.getElementById(id);
    container.replaceChildren(el("p", { class: "activation-example-empty" }, message));
  });
}

function renderActivationExamples(patch) {
  const status = document.getElementById("activation-neighbor-status");
  const method = document.getElementById("activation-neighbor-method");
  const recipientReference = document.getElementById("recipient-neighbor-reference");
  const sourceReference = document.getElementById("source-neighbor-reference");
  const corpusLabel = ACTIVATION_EXAMPLE_SOURCE_LABELS[state.activationExampleSource];
  method.textContent = ACTIVATION_EXAMPLE_SOURCE_DESCRIPTIONS[state.activationExampleSource];
  if (!patch.processed) {
    status.textContent = "The selected grid is unprocessed, so no measured reference vectors exist.";
    recipientReference.textContent = "No measured reference";
    sourceReference.textContent = "No measured reference";
    setActivationExamplesEmpty("Activation examples will appear only for a measured vector cell.");
    return;
  }
  if (patch.aggregate) {
    status.textContent = "Choose an individual function: an all-function mean is not one activation vector.";
    recipientReference.textContent = "Aggregate has no single vector";
    sourceReference.textContent = "Aggregate has no single vector";
    setActivationExamplesEmpty("Select one function probe to define exact recipient and donor vectors.");
    return;
  }
  if (weightPatchSelected()) {
    status.textContent = "Weight interventions do not define one source/recipient activation vector.";
    recipientReference.textContent = "Not applicable to weights";
    sourceReference.textContent = "Not applicable to weights";
    setActivationExamplesEmpty("Select an activation boundary to inspect activation-vector neighbors.");
    return;
  }
  state.patchCellTokenIndex = Math.max(
    0,
    Math.min(state.patchCellTokenIndex, patch.tokenPositions.length - 1),
  );
  state.patchCellLayer = state.patchCellLayer === null
    ? patch.layers - 1
    : Math.max(0, Math.min(state.patchCellLayer, patch.layers - 1));
  const tokenIndex = state.patchCellTokenIndex;
  const layer = state.patchCellLayer;
  const checkpoints = state.data.checkpoints;
  const recipientStep = checkpoints[state.recipientIndex];
  const donorStep = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const recipientEntry = activationExampleEntry(recipientStep, state.functionId);
  const sourceEntry = activationExampleEntry(donorStep, state.functionId);
  const position = patch.tokenPositions[tokenIndex];
  const reverseLabel = `reverse token −${position.reverseIndex} · layer ${layer}`;
  recipientReference.textContent = `${recipientStep === 0 ? "frozen base" : `step ${recipientStep}`} · ${reverseLabel} · ${position.recipientToken}`;
  sourceReference.textContent = `${donorStep === 0 ? "frozen base" : `step ${donorStep}`} · ${reverseLabel} · ${position.sourceToken}`;
  if (!recipientEntry || !sourceEntry) {
    status.textContent = `${corpusLabel} nearest-example audit is unprocessed for ${PROMPT_SOURCE_LABELS[state.patchMode] ?? state.patchMode}.`;
    setActivationExamplesEmpty(
      `No measured ${corpusLabel} activation-neighbor artifact exists for this checkpoint yet.`,
    );
    return;
  }
  const recipientNeighborKey = patchReferenceKey(recipientEntry.neighbors);
  const recipientCatalogKey = patchReferenceKey(recipientEntry.candidates);
  const sourceNeighborKey = patchReferenceKey(sourceEntry.neighbors);
  const sourceCatalogKey = patchReferenceKey(sourceEntry.candidates);
  const errors = [
    recipientNeighborKey,
    recipientCatalogKey,
    sourceNeighborKey,
    sourceCatalogKey,
  ].map((key) => activationExampleErrors.get(key)).filter(Boolean);
  if (errors.length) {
    status.textContent = `Activation-example artifact failed to load: ${errors[0]}`;
    setActivationExamplesEmpty("A measured nearest-example file exists but could not be loaded.");
    return;
  }
  const recipientNeighbors = activationNeighborChunks.get(recipientNeighborKey);
  const recipientCatalog = activationCandidateCatalogs.get(recipientCatalogKey);
  const sourceNeighbors = activationNeighborChunks.get(sourceNeighborKey);
  const sourceCatalog = activationCandidateCatalogs.get(sourceCatalogKey);
  if (!recipientNeighbors || !recipientCatalog || !sourceNeighbors || !sourceCatalog) {
    status.textContent = "Loading the selected checkpoints’ activation-example banks…";
    setActivationExamplesEmpty("Measured nearest examples are loading in the background.");
    scheduleActivationExampleLoads();
    return;
  }
  for (const neighbors of [recipientNeighbors, sourceNeighbors]) {
    if (
      neighbors.candidateSource !== state.activationExampleSource
      || neighbors.mode !== state.patchMode
      || neighbors.functionId !== state.functionId
      || neighbors.positionCount !== patch.tokenPositions.length
      || neighbors.layerCount !== patch.layers
    ) {
      throw new Error("activation-example artifact does not match the selected patch grid");
    }
  }
  for (const catalog of [recipientCatalog, sourceCatalog]) {
    if (catalog.candidateSource !== state.activationExampleSource) {
      throw new Error("activation-example candidate corpus does not match the selected source");
    }
  }
  renderActivationExampleList(
    document.getElementById("recipient-neighbor-examples"),
    recipientNeighbors.recipient[tokenIndex][layer],
    recipientCatalog,
  );
  renderActivationExampleList(
    document.getElementById("source-neighbor-examples"),
    sourceNeighbors.source[tokenIndex][layer],
    sourceCatalog,
  );
  status.textContent = `Measured ${corpusLabel} cosine-neighbor audit · selected ${reverseLabel} · click a cell or use arrow keys to update both columns.`;
}

function focusSelectedPatchCell() {
  window.requestAnimationFrame(() => {
    const selected = document.querySelector(
      `.heat-cell[data-token-index="${state.patchCellTokenIndex}"]`
      + `[data-layer="${state.patchCellLayer}"]`,
    );
    selected?.focus();
  });
}

function moveSelectedPatchCell(patch, tokenDelta, layerDelta) {
  if (!patch.processed || patch.axisKind === "layer_only") return;
  const nextToken = Math.max(
    0,
    Math.min(state.patchCellTokenIndex + tokenDelta, patch.tokenPositions.length - 1),
  );
  const nextLayer = Math.max(
    0,
    Math.min(state.patchCellLayer + layerDelta, patch.layers - 1),
  );
  if (nextToken === state.patchCellTokenIndex && nextLayer === state.patchCellLayer) return;
  state.patchCellTokenIndex = nextToken;
  state.patchCellLayer = nextLayer;
  renderPatching();
  focusSelectedPatchCell();
}

function renderPatching() {
  const patch = patchData();
  const weightAnalysis = weightAnalysisSelected();
  const sourceControl = document.getElementById("patch-source-control");
  const boundaryControl = document.getElementById("patch-boundary-control");
  const functionControl = document.getElementById("patch-function-control");
  document.getElementById("patch-mode-select").disabled = weightAnalysis;
  document.getElementById("patch-interface-select").disabled = weightAnalysis;
  document.getElementById("function-select").disabled = weightAnalysis;
  [sourceControl, boundaryControl, functionControl].forEach((control) => {
    control.style.opacity = weightAnalysis ? ".38" : "1";
  });
  document.getElementById("prompt-audit").hidden = weightAnalysis;
  document.getElementById("activation-neighbor-panel").hidden = weightAnalysis;
  document.getElementById("patch-heatmap-axis").textContent = weightAnalysis
    ? "input → decoder layer depth → output · learned tensor family ↓"
    : "decoder layer depth → · exact tokenizer position, stepping backward ↓";
  document.getElementById("patch-heatmap").setAttribute(
    "aria-label",
    weightAnalysis
      ? "Complete effective learned-weight alignment heatmap"
      : "Answer-choice activation patching heatmap",
  );
  state.patchCellTokenIndex = Math.max(
    0,
    Math.min(state.patchCellTokenIndex, patch.tokenPositions.length - 1),
  );
  state.patchCellLayer = state.patchCellLayer === null
    ? patch.layers - 1
    : Math.max(0, Math.min(state.patchCellLayer, patch.layers - 1));
  const patchReference = selectedPatchReference();
  const patchReferenceId = patchReferenceKey(patchReference);
  const patchLoadError = patchReferenceId ? patchChunkErrors.get(patchReferenceId) : null;
  const patchLoading = Boolean(patchReferenceId && !patch.processed && !patchLoadError);
  const heatmap = document.getElementById("patch-heatmap");
  heatmap.replaceChildren();
  if (state.patchTooltipPinned) document.getElementById("tooltip").hidden = true;
  heatmap.onmouseleave = () => {
    if (state.patchTooltipPinned) restorePinnedHeatTooltip();
    else document.getElementById("tooltip").hidden = true;
  };
  heatmap.style.gridTemplateColumns = `300px repeat(${patch.layers}, minmax(17px, 1fr))`;
  heatmap.append(el("div"));
  for (let layer = 0; layer < patch.layers; layer += 1) {
    const column = patch.weightColumnAxis?.[layer] ?? null;
    const label = weightAnalysis
      ? column?.kind === "decoder_layer" && column.layer % 4 !== 0
        ? "·"
        : column?.label ?? "·"
      : layer % 4 === 0 ? String(layer) : "·";
    heatmap.append(el("div", { class: "heatmap-layer" }, label));
  }
  patch.tokenPositions.forEach((position, tokenIndex) => {
    const layerOnly = patch.axisKind === "layer_only";
    const weightLayer = patch.axisKind === "weight_layer";
    const sameCoordinate = layerOnly || weightLayer || (position.aggregate
      ? position.sourceTokenSignature === position.recipientTokenSignature
      : position.sourceToken === position.recipientToken
        && position.sourceIndex === position.recipientIndex
        && position.sourceTokenId === position.recipientTokenId);
    const sourcePrefix = promptSourcePrefix();
    const recipientPrefix = state.patchMode === "checkpoint"
      ? "recipient "
      : "clean/recipient ";
    const sourceCoordinate = weightLayer
      ? `checkpoint A · ${position.sourceToken} effective tensor`
      : layerOnly
      ? "donor checkpoint · complete learned block update"
      : position.aggregate
        ? aggregateTokenCoordinate(sourcePrefix, position.sourceToken)
        : tokenCoordinate(sourcePrefix, position.sourceIndex, position.sourceTokenId, position.sourceToken);
    const recipientCoordinate = weightLayer
      ? `checkpoint B · ${position.recipientToken} effective tensor`
      : layerOnly
      ? "recipient checkpoint · complete learned block update"
      : position.aggregate
        ? aggregateTokenCoordinate(recipientPrefix, position.recipientToken)
        : tokenCoordinate(recipientPrefix, position.recipientIndex, position.recipientTokenId, position.recipientToken);
    const tokenText = weightLayer
      ? position.sourceToken
      : layerOnly
      ? "All sequence positions · entire decoder block"
      : sameCoordinate
        ? (position.aggregate
          ? aggregateTokenCoordinate("", position.sourceToken)
          : tokenCoordinate("", position.sourceIndex, position.sourceTokenId, position.sourceToken))
        : `${sourceCoordinate} → ${recipientCoordinate}`;
    const label = el("div", { class: `heatmap-token${!layerOnly && !weightLayer && position.reverseIndex === 0 ? " anchor" : ""}` });
    label.append(el("b", {}, weightLayer
      ? position.component.tensor_rank === 2 ? "matrix" : "vector"
      : layerOnly
      ? "all tokens"
      : position.reverseIndex === 0 ? "−0 · end" : `−${position.reverseIndex}`));
    label.append(el("span", { title: tokenText }, tokenText));
    heatmap.append(label);
    for (let layer = 0; layer < patch.layers; layer += 1) {
      const cellMeasurement = patch.matrix[tokenIndex][layer];
      const cell = el("div", { class: "heat-cell", tabindex: "0" });
      if (!patch.processed) {
        cell.classList.add("unprocessed");
        const unavailableReason = !patch.applicable
          ? representationAlignmentSelected()
            ? "Vector alignment is defined for activation boundaries, not learned-weight interventions. Select residual, attention, or MLP activations."
            : "The combined prompt-counterfactual × checkpoint experiment is activation-only. Select an activation boundary, or select Checkpoint transfer for a weight-only intervention."
          : patchLoadError
          ? "A measured file exists, but it could not be loaded. No fallback value is displayed."
          : patchLoading
            ? "Measured values are loading. No temporary numeric value is displayed."
            : weightAnalysis
              ? "No effective-weight comparison has been measured for this unordered checkpoint pair."
              : representationAlignmentSelected()
              ? "No donor/recipient representation-alignment value has been measured for this selection."
              : `No ${weightPatchSelected() ? "weight" : "activation"}-patching value has been measured for this recipient/donor selection.`;
        const coordinate = weightLayer
          ? `Layer ${layer} · ${position.weightName} effective matrix`
          : layerOnly
          ? `Layer ${layer} · entire decoder block`
          : `Layer ${layer} · reverse token −${position.reverseIndex}`;
        bindHeatTooltip(cell, `<b>No displayed value</b><br>${coordinate}<br><br>${unavailableReason}`);
        cell.setAttribute("aria-label", weightLayer
          ? `layer ${layer}, ${position.weightName} effective matrix, unprocessed`
          : layerOnly
          ? `layer ${layer}, entire decoder block, unprocessed`
          : `layer ${layer}, reverse token ${position.reverseIndex}, unprocessed`);
        heatmap.append(cell);
        continue;
      }
      if (weightAnalysis && !Number.isFinite(cellMeasurement)) {
        cell.classList.add("not-applicable");
        const column = patch.weightColumnAxis[layer];
        const component = position.component;
        const placedHere = (
          (component.placement === "input" && column.kind === "global_input")
          || (component.placement === "output" && column.kind === "global_output")
          || (component.placement === "layer" && column.kind === "decoder_layer")
        );
        const reason = placedHere && component.tensor_rank === 1
          ? "Row/column decompositions are not defined for a one-dimensional norm vector. Select Frobenius cosine or Frobenius L2."
          : "This tensor family does not exist at this input/layer/output coordinate.";
        bindHeatTooltip(
          cell,
          `<b>Not applicable</b><br>${escapeHtml(position.sourceToken)} · ${escapeHtml(column.label)}<br><br>${reason}`,
        );
        cell.setAttribute("aria-label", `${position.weightName}, ${column.label}, not applicable`);
        heatmap.append(cell);
        continue;
      }
      const averagingNote = patch.aggregate ? `<br>cellwise mean over n=${patch.functionCount} functions` : "";
      const coordinate = weightLayer
        ? `${patch.weightColumnAxis[layer].label} · ${position.weightName} effective tensor`
        : layerOnly
        ? `Layer ${layer} · entire decoder block`
        : `Layer ${layer} · reverse token −${position.reverseIndex}`;
      const logitLensNote = weightLayer
        ? ""
        : fullVocabularyLogitLensHtml(patch, tokenIndex, layer);
      let display;
      if (patch.measurementKind === "weight_alignment") {
        const metric = WEIGHT_VISUALIZATION_METRICS[state.patchVisualization];
        const scale = weightAlignmentScale();
        cell.style.background = colorFor(cellMeasurement, metric, scale?.max ?? null);
        const varianceMetric = WEIGHT_VISUALIZATION_VARIANCES[state.patchVisualization];
        const variance = varianceMetric
          ? patch.weightVarianceMatrices?.[varianceMetric]?.[tokenIndex]?.[layer]
          : null;
        const varianceScale = weightVarianceScale();
        if (Number.isFinite(variance) && varianceScale?.max > 0) {
          const amount = Math.sqrt(Math.max(0, Math.min(1, variance / varianceScale.max)));
          const width = .5 + amount * 3.5;
          cell.style.boxShadow = `inset 0 0 0 ${width.toFixed(2)}px rgba(255, 255, 255, .30)`;
        }
        display = formatAlignmentValue(cellMeasurement);
        const shape = patch.weightShapes?.[tokenIndex]?.[layer] ?? null;
        const shapeNote = shape
          ? shape.length === 2
            ? `${shape[0].toLocaleString()} output rows × ${shape[1].toLocaleString()} input columns`
            : `${shape[0].toLocaleString()} learned scale values`
          : "tensor shape unavailable";
        const varianceNote = Number.isFinite(variance)
          ? `<br>variance: ${formatAlignmentValue(variance)}`
          : "";
        const frozenNote = position.component.frozen_during_lora
          ? " · <small>analytic frozen identity</small>"
          : "";
        const selectedLabel = PATCH_VISUALIZATION_LABELS[state.patchVisualization]
          .replace("Weights · ", "");
        const baseTooltip = `<b>${coordinate}</b> · ${escapeHtml(shapeNote)}${frozenNote}<br><b>${escapeHtml(selectedLabel)}</b>: ${formatAlignmentValue(cellMeasurement)}${varianceNote}`;
        bindHeatTooltip(
          cell,
          () => `${baseTooltip}${weightDetailGridHtml(patch, tokenIndex, layer)}`,
        );
      } else if (patch.measurementKind === "representation_alignment") {
        const cosine = patch.cosineMatrix[tokenIndex][layer];
        const distance = patch.l2Matrix[tokenIndex][layer];
        const sourceNorm = patch.sourceNormMatrix?.[tokenIndex]?.[layer] ?? null;
        const recipientNorm = patch.recipientNormMatrix?.[tokenIndex]?.[layer] ?? null;
        const l2Scale = representationAlignmentScale();
        cell.style.background = colorFor(
          cellMeasurement,
          state.patchVisualization,
          l2Scale?.max ?? null,
        );
        display = formatAlignmentValue(cellMeasurement);
        const normNote = sourceNorm === null || recipientNorm === null
          ? "<br><small>Exact identity inferred analytically; activation norms were not recomputed.</small>"
          : `<br>donor/source norm: ${formatAlignmentValue(sourceNorm)}<br>recipient norm: ${formatAlignmentValue(recipientNorm)}`;
        const scaleNote = state.patchVisualization === "l2_distance" && l2Scale
          ? `<br><small>Color saturates at model/boundary p95 scale ${formatAlignmentValue(l2Scale.max)}; raw hover value is unclipped.</small>`
          : "";
        const aggregateDefinition = patch.aggregate
          ? `<br><small>Mean of ${patch.functionCount} per-function scalar comparisons; vectors are not averaged before scoring.</small>`
          : "";
        bindHeatTooltip(cell, `<b>${coordinate}</b>${averagingNote}<br>${escapeHtml(sourceCoordinate)}<br>${escapeHtml(recipientCoordinate)}<br><br><b>unpatched representation alignment</b><br>cosine similarity: ${formatAlignmentValue(cosine)}<br>raw L2 distance: ${formatAlignmentValue(distance)}${normNote}${scaleNote}${aggregateDefinition}${logitLensNote}<br><small>Observational comparison only: no activation was transplanted and no downstream probability was measured.</small>`);
      } else {
        const probability = cellMeasurement;
        const delta = probability - patch.recipient;
        const value = state.patchMetric === "probability" ? probability : delta / .25;
        cell.style.background = colorFor(value, state.patchMetric);
        display = state.patchMetric === "probability"
          ? formatPercent(probability)
          : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`;
        const baselineScope = patch.aggregate
          ? `mean of ${patch.functionCount} single code-choice probes`
          : "same single code-choice probe";
        const interventionNote = tokenWeightPatchSelected()
          ? "<br><small>donor LoRA contribution used only at this token; all other token contributions stay recipient</small>"
          : "";
        const sourceTargetNote = patch.sourceTargetMatrix
          ? (() => {
            const sourceTargetProbability = patch.sourceTargetMatrix[tokenIndex][layer];
            const sourceTargetDelta = sourceTargetProbability - patch.sourceTargetRecipient;
            const sourceTargetLabel = patch.aggregate || patch.sourceCorrectIndex === null
              ? "each source's correct label"
              : `source-correct label ${"ABCDE"[patch.sourceCorrectIndex]}`;
            return `<br><br><b>${sourceTargetLabel}</b><br>patched result: ${formatPercent(sourceTargetProbability)}<br>unpatched recipient baseline: ${formatPercent(patch.sourceTargetRecipient)}<br>unpatched source baseline: ${formatPercent(patch.sourceTargetSource)}<br>change from recipient: ${sourceTargetDelta >= 0 ? "+" : ""}${(sourceTargetDelta * 100).toFixed(2)} pp`;
          })()
          : "";
        bindHeatTooltip(cell, `<b>${coordinate}</b>${averagingNote}<br>${escapeHtml(sourceCoordinate)}<br>${escapeHtml(recipientCoordinate)}${interventionNote}<br><br><b>clean-correct label</b><br>patched result: ${formatPercent(probability)}<br>unpatched recipient baseline: ${formatPercent(patch.recipient)}<br>unpatched donor/source baseline: ${formatPercent(patch.source)}<br>change from recipient: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} pp${sourceTargetNote}${logitLensNote}<br><small>${baselineScope}</small>`);
      }
      cell.setAttribute("aria-label", weightLayer
        ? `${patch.weightColumnAxis[layer].label}, ${position.weightName} effective tensor, ${display}`
        : layerOnly
        ? `layer ${layer}, entire decoder block, ${display}`
        : `layer ${layer}, reverse token ${position.reverseIndex}, ${display}`);
      if (
        !layerOnly
        && state.patchCellTokenIndex === tokenIndex
        && state.patchCellLayer === layer
      ) {
        cell.classList.add("selected");
      }
      if (!layerOnly) {
        cell.dataset.tokenIndex = String(tokenIndex);
        cell.dataset.layer = String(layer);
        const selectReference = (moveFocus = false, event = null) => {
          const bounds = cell.getBoundingClientRect();
          state.patchCellTokenIndex = tokenIndex;
          state.patchCellLayer = layer;
          state.patchTooltipPinned = true;
          state.patchTooltipPosition = event && event.detail > 0
            ? { clientX: event.clientX, clientY: event.clientY }
            : {
              clientX: bounds.left + bounds.width / 2,
              clientY: bounds.top + bounds.height / 2,
            };
          renderPatching();
          if (moveFocus) focusSelectedPatchCell();
        };
        cell.addEventListener("click", (event) => selectReference(true, event));
        cell.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectReference(true);
            return;
          }
          const movement = {
            ArrowLeft: [0, -1],
            ArrowRight: [0, 1],
            ArrowUp: [-1, 0],
            ArrowDown: [1, 0],
          }[event.key];
          if (movement) {
            event.preventDefault();
            moveSelectedPatchCell(patch, movement[0], movement[1]);
          }
        });
      }
      heatmap.append(cell);
    }
  });

  const checkpoints = state.data.checkpoints;
  const recipient = checkpoints[state.recipientIndex];
  const donor = checkpoints[usesCheckpointDonor() ? state.donorIndex : state.recipientIndex];
  const patchStatus = document.getElementById("patch-status");
  const interfaceLabel = weightAnalysis
    ? "Effective projection weights"
    : PATCH_INTERFACE_LABELS[state.patchInterface];
  const aggregateStatus = patch.aggregate
    ? patch.processed ? ` · mean n=${patch.functionCount}` : ` · n=${patch.functionCount} functions`
    : "";
  patchStatus.textContent = (!patch.applicable
    ? `not applicable · ${interfaceLabel}`
    : patch.processed
      ? patch.analytic
        ? `analytic identity · ${interfaceLabel}`
        : weightAnalysis
          ? `measured ${PATCH_VISUALIZATION_LABELS[state.patchVisualization].toLowerCase()}`
          : representationAlignmentSelected()
          ? `measured ${PATCH_VISUALIZATION_LABELS[state.patchVisualization].toLowerCase()} · ${interfaceLabel}`
          : `measured intervention · ${interfaceLabel}`
      : patchLoadError
        ? `load failed · ${interfaceLabel}`
        : patchLoading
          ? `loading · ${interfaceLabel}`
          : `unprocessed · ${interfaceLabel}`) + aggregateStatus;
  patchStatus.classList.toggle("measured", patch.measured || patch.analytic);
  patchStatus.classList.toggle("loading", patchLoading);
  patchStatus.classList.toggle("load-error", Boolean(patchLoadError));
  patchStatus.classList.toggle("unprocessed", !patch.processed && !patchLoading && !patchLoadError);
  const legend = document.getElementById("patch-legend");
  legend.replaceChildren();
  if (patch.processed) {
    if (weightAnalysis && WEIGHT_VISUALIZATION_METRICS[state.patchVisualization].includes("cosine")) {
      legend.append(el("span", {}, "0 unaligned"));
      const scale = el("i");
      scale.style.background = "linear-gradient(90deg, #375caa 0%, #fff 70.71%, #ef775f 100%)";
      legend.append(scale);
      legend.append(el("span", {}, "1 aligned"));
      if (WEIGHT_VISUALIZATION_VARIANCES[state.patchVisualization]) {
        legend.append(el("span", {}, "quadratic color interpolation · light inset width = axis variance (0 → cross-cell p95)"));
      } else {
        legend.append(el("span", {}, "quadratic color interpolation"));
      }
    } else if (state.patchVisualization === "cosine_similarity") {
      legend.append(el("span", {}, "−1 opposite"));
      const scale = el("i");
      scale.style.background = "linear-gradient(90deg, #5d79b9, #eee8d8, #ef775f)";
      legend.append(scale);
      legend.append(el("span", {}, "+1 aligned"));
    } else if (state.patchVisualization === "l2_distance" || weightAnalysis) {
      const l2Scale = weightAnalysis ? weightAlignmentScale() : representationAlignmentScale();
      legend.append(el("span", {}, "0 identical"));
      const scale = el("i");
      scale.style.background = "linear-gradient(90deg, #eee8d8, #674b93)";
      legend.append(scale);
      legend.append(el("span", {}, l2Scale
        ? `≥${formatAlignmentValue(l2Scale.max)} farther`
        : "farther"));
    } else {
      legend.append(el("span", {}, "lower P(correct)"));
      legend.append(el("i"));
      legend.append(el("span", {}, "higher P(correct)"));
    }
  } else if (!patch.applicable) {
    legend.append(el("i", { class: "unprocessed" }));
    legend.append(el("span", {}, representationAlignmentSelected()
      ? "not applicable · weight interfaces do not expose one activation vector"
      : "not applicable · combined prompt × weight intervention undefined"));
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
  const visualizationDescription = weightAnalysis
    ? "Full effective matrices are the shared frozen base weight plus each checkpoint’s scaled LoRA B·A update. Rows are output channels and columns are input channels. This prompt-independent comparison is observational and exactly symmetric."
    : representationAlignmentSelected()
      ? `${PATCH_INTERFACE_DESCRIPTIONS[state.patchInterface]} The heatmap compares exact unpatched donor/source and recipient vectors at matched token/layer coordinates; it does not run an intervention.`
      : `${PATCH_INTERFACE_DESCRIPTIONS[state.patchInterface]} The heatmap replaces the selected recipient state or learned-weight contribution and measures the downstream answer probability.`;
  document.getElementById("patch-interface-description").textContent = visualizationDescription;
  document.getElementById("recipient-label").textContent = recipient === 0 ? "frozen base" : `step ${recipient}`;
  document.getElementById("donor-label").textContent = donor === 0 ? "frozen base" : `step ${donor}`;
  document.getElementById("donor-kind-label").textContent = weightAnalysis
    ? "comparison checkpoint"
    : representationAlignmentSelected()
      ? "representation source"
    : weightPatchSelected()
      ? "weight source"
      : "activation source";
  document.getElementById("donor-control").style.opacity = usesCheckpointDonor() ? "1" : ".38";
  const donorSlider = document.getElementById("donor-slider");
  donorSlider.disabled = !usesCheckpointDonor();
  donorSlider.value = sliderValueForStep(
    usesCheckpointDonor() ? checkpoints[state.donorIndex] : recipient,
    state.patchTimeScale,
  );
  const fn = state.data.functions.find((item) => item.id === state.functionId);
  document.getElementById("clean-question").textContent = weightAnalysis
    ? `${recipient === 0 ? "frozen base" : `step ${recipient}`} complete effective learned weights`
    : patch.aggregate
      ? `Mean over all ${patch.functionCount} clean definition questions`
      : `What is the definition of ${fn.alias}?`;
  document.getElementById("recipient-question-label").textContent = weightAnalysis
    ? "checkpoint A"
    : "clean recipient question";
  if (weightAnalysis) {
    document.getElementById("source-question-label").textContent = "checkpoint B";
    document.getElementById("source-question").textContent =
      `${donor === 0 ? "frozen base" : `step ${donor}`} complete effective learned weights`;
  } else if (state.patchMode === "checkpoint") {
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
    const sourceFunction = patch.aggregate
      ? null
      : state.data.functions.find((item) => item.id === patch.sourceFunctionId);
    const cleanLetter = patch.recipientCorrectIndex === null
      ? null
      : "ABCDE"[patch.recipientCorrectIndex];
    const sourceLetter = patch.sourceCorrectIndex === null
      ? null
      : "ABCDE"[patch.sourceCorrectIndex];
    document.getElementById("source-question-label").textContent = weightPatchSelected()
      ? "no distinct weight source"
      : `${PROMPT_SOURCE_LABELS[state.patchMode]} source`;
    let sourceQuestion;
    let explanation;
    if (weightPatchSelected()) {
      sourceQuestion = patch.aggregate
        ? `Combined prompt and weight intervention undefined across ${patch.functionCount} pairs`
        : "Combined prompt and weight intervention undefined";
      explanation = "These modes transplant a counterfactual prompt activation, optionally across checkpoints. A weight-only patch cannot encode that changed prompt state, so the combined intervention is intentionally undefined. Select an activation boundary, or select Checkpoint transfer for a clean-prompt weight intervention.";
    } else if (state.patchMode === "across_sample") {
      sourceQuestion = patch.aggregate
        ? `Mean over all ${patch.functionCount} fixed-derangement dirty-name questions`
        : `What is the definition of ${sourceFunction.alias}?`;
      explanation = "Patching dirty-name states into the clean prompt tests where the alternate identity suppresses the correct implementation. Cells remain colored by clean P(correct).";
    } else if (state.patchMode === "cyclic_choices") {
      sourceQuestion = patch.aggregate
        ? `Same questions with every option moved A→B→C→D→E→A · n=${patch.functionCount}`
        : `Same question · option contents shifted +1 · correct ${cleanLetter}→${sourceLetter}`;
      explanation = "The donor asks the same function question but moves every answer content forward one label. A late answer-label readout should lower the clean-correct label and raise the source-correct label. Cell color remains clean P(correct); hover shows both labels and the full-vocabulary logit lens.";
    } else if (state.patchMode === "deranged_choices") {
      sourceQuestion = patch.aggregate
        ? `Same questions with deterministic random no-fixed-point option orders · n=${patch.functionCount}`
        : `Same question · random option derangement · correct ${cleanLetter}→${sourceLetter}`;
      explanation = "Each donor uses a deterministic random option derangement with no answer left in place. This controls for the fixed +1 rotation: a label-readout circuit should transfer whichever new label contains the correct implementation. Cell color remains clean P(correct); hover shows the donor-label effect and full-vocabulary logit lens.";
    } else if (["unrelated_question", "unrelated_question_same_letter"].includes(state.patchMode)) {
      const relation = patch.sourceLabelRelation === "same_as_recipient"
        ? "the same answer letter as its clean pair"
        : "a different answer letter from its clean pair";
      sourceQuestion = patch.aggregate
        ? `Mean over ${patch.functionCount} unrelated non-coding five-choice questions; ${relation}`
        : `${patch.sourceQuestion} · correct ${sourceLetter}, clean probe correct ${cleanLetter}`;
      explanation = `The donor is an unrelated non-coding MCQ in the same five-choice format, with ${relation}. Comparing its matched- and mismatched-letter versions separates MCQ-format transfer from transfer of one particular answer label. Cell color remains clean P(correct); hover shows both label effects and the full-vocabulary logit lens.`;
    } else if (["letter_context_same", "letter_context_different"].includes(state.patchMode)) {
      const relation = patch.sourceLabelRelation === "same_as_recipient"
        ? "same letter"
        : "different letter";
      sourceQuestion = patch.aggregate
        ? `Mean over ${patch.functionCount} non-question record completions · ${relation}`
        : `${patch.sourceContext} · expected next token ${sourceLetter}; clean probe correct ${cleanLetter}`;
      explanation = `The donor is not a question and has no answer choices: it is a short record whose next token should be one capital letter. Comparing ${relation} transfer against the MCQ controls tests whether the late state is a generic “say ${sourceLetter}” direction or a label readout specialized to question answering. Cell color remains clean P(correct); hover shows both labels and the full-vocabulary logit lens.`;
    } else if (state.patchMode === "same_mcq_formats") {
      sourceQuestion = patch.aggregate
        ? `Same ${patch.functionCount} function MCQs in a paired, balanced mix of five alternative layouts`
        : `${patch.sourceQuestion} · ${patch.sourceFormat}`;
      explanation = "The donor asks the same function question with the same options, order, and correct letter, but uses one of five alternative MCQ layouts. Formats are assigned deterministically across functions so the aggregate covers every layout without averaging hidden states from different token sequences. This isolates presentation from function content.";
    } else if (state.patchMode === "unrelated_mcq_formats") {
      sourceQuestion = patch.aggregate
        ? `${patch.functionCount} unrelated non-coding MCQs in the same paired layout mix; answer letters matched to clean probes`
        : `${patch.sourceQuestion} · ${patch.sourceFormat} · correct ${sourceLetter}, matching clean ${cleanLetter}`;
      explanation = "The donor is an unrelated non-coding MCQ rendered in the format paired to the same function’s varied-format control. Its correct letter matches the clean probe, separating question content from MCQ layout and answer-letter identity.";
    } else if (state.patchMode === "same_conversational") {
      sourceQuestion = patch.aggregate
        ? `Same ${patch.functionCount} opaque-function questions in a balanced mix of five conversational open-response forms`
        : `${patch.sourceQuestion} · ${patch.sourceFormat}`;
      explanation = "The donor asks about the same opaque function conversationally and requests a free-form lambda, with no A–E choices. This tests whether function-content representations transfer beyond explicit multiple-choice scaffolding; there is no declared donor answer letter.";
    } else if (state.patchMode === "unrelated_open_ended") {
      sourceQuestion = patch.aggregate
        ? `${patch.functionCount} unrelated non-coding questions in the paired conversational open-response forms`
        : `${patch.sourceQuestion} · ${patch.sourceFormat}`;
      explanation = "The donor is an unrelated, non-coding open-response question with no A–E choices or MCQ instruction. It is format-paired to the conversational function control and has no declared donor answer letter.";
    } else if (state.patchMode === "same_conversational_choices") {
      sourceQuestion = patch.aggregate
        ? `Same ${patch.functionCount} opaque-function questions in a balanced mix of five conversational A–E phrasings`
        : `${patch.sourceQuestion} · ${patch.sourceFormat} · correct ${sourceLetter}, matching clean ${cleanLetter}`;
      explanation = "The donor asks the same function question casually while retaining the clean probe’s five implementations, option order, and correct A–E letter. This isolates conversational presentation from function content without changing the five-way probability metric.";
    } else if (state.patchMode === "unrelated_conversational_choices") {
      sourceQuestion = patch.aggregate
        ? `${patch.functionCount} unrelated non-coding questions in paired conversational A–E phrasings; answer letters matched to clean probes`
        : `${patch.sourceQuestion} · ${patch.sourceFormat} · correct ${sourceLetter}, matching clean ${cleanLetter}`;
      explanation = "The donor asks an unrelated non-coding question casually but still gives five A–E possibilities. Its correct letter matches the clean probe, separating function content from conversational presentation and answer-letter identity while preserving the same probability metric.";
    } else {
      sourceQuestion = "Unsupported prompt counterfactual";
      explanation = "This prompt-counterfactual mode has no explanatory copy.";
    }
    if (usesCheckpointDonor()) {
      const sourceCheckpoint = donor === 0 ? "frozen base" : `step ${donor}`;
      const recipientCheckpoint = recipient === 0 ? "frozen base" : `step ${recipient}`;
      sourceQuestion += ` · ${sourceCheckpoint} source → ${recipientCheckpoint} clean recipient`;
      explanation += ` The counterfactual representation and source-side logit lens come from ${sourceCheckpoint}; the clean baseline, clean-side lens, and every computation after the patched cell use ${recipientCheckpoint}.`;
    }
    document.getElementById("source-question").textContent = sourceQuestion;
    document.getElementById("patch-explanation").textContent = explanation;
  }
  if (weightAnalysis) {
    if (patchLoadError) {
      document.getElementById("patch-explanation").textContent = `A measured effective-weight artifact exists, but its scalar grid could not be loaded (${patchLoadError}). No fallback value is shown.`;
    } else if (patchLoading) {
      document.getElementById("patch-explanation").textContent = "Measured effective-weight geometry is loading. Purple hatching encodes no temporary value.";
    } else if (!patch.processed) {
      document.getElementById("patch-explanation").textContent = "This unordered checkpoint pair has not been measured. Purple cells encode no similarity, distance, interpolation, or synthetic value.";
    } else if (patch.analytic) {
      document.getElementById("patch-explanation").textContent = "The two sliders select the same checkpoint, so every displayed weight matrix is exactly itself: all cosine metrics are 1 and all L2 metrics are 0. This diagonal is analytic and no model was loaded.";
    } else {
      document.getElementById("patch-explanation").textContent = "The atlas covers every learned tensor: embedding, decoder projections and norms, final norm, and unembedding. The trained projections compare full effective matrices (frozen base + scaled LoRA B·A); frozen non-target tensors are exact analytic identities. Each unordered checkpoint pair is stored once for exact symmetry. Weight-cosine colors use a quadratic blue-to-white-to-red ramp; hover retains the raw cosine. Decomposed views add a 30%-opacity light inset whose width encodes population variance. All four packed row/column detail families prefetch for the selected pair; hover outlines attention-head regions and densely tiles large MLP axes.";
    }
  } else if (representationAlignmentSelected()) {
    if (!patch.applicable) {
      document.getElementById("patch-explanation").textContent = "Cosine similarity and L2 distance compare activation vectors. Learned-weight boundaries do not expose one vector with the same semantics, so this selection is intentionally not applicable. The purple squares encode no result.";
    } else if (patchLoadError) {
      document.getElementById("patch-explanation").textContent = `A measured representation-alignment artifact exists, but its data file could not be loaded (${patchLoadError}). No fallback value is shown.`;
    } else if (patchLoading) {
      document.getElementById("patch-explanation").textContent = "Measured unpatched donor/recipient vectors are loading. The temporary purple hatch encodes no cosine, distance, or synthetic value.";
    } else if (!patch.processed) {
      document.getElementById("patch-explanation").textContent = "This donor/recipient representation pair has not been measured at this boundary. Purple squares are availability markers only: they encode no similarity, distance, interpolation, or synthetic result.";
    } else if (patch.analytic) {
      document.getElementById("patch-explanation").textContent = "Recipient and donor use the identical prompt and checkpoint, so every matched activation vector is exactly itself: cosine similarity is 1 and L2 distance is 0. These identity values are analytic, explicitly labeled, and did not require a model run.";
    } else {
      document.getElementById("patch-explanation").textContent = `This is an observational comparison of the exact unpatched donor/source and recipient vectors at the selected ${PATCH_INTERFACE_LABELS[state.patchInterface]} token × layer boundary. Cosine uses float32 dot products and norms; L2 is the raw float32 Euclidean distance. No vector is transplanted and no downstream answer probability is measured. Raw L2 magnitudes are boundary- and model-specific. Hover shows both metrics and both vector norms.`;
    }
  } else if (!patch.applicable) {
    document.getElementById("patch-explanation").textContent = `The combined prompt-counterfactual × checkpoint experiment is activation-only. Select an activation boundary, or select Checkpoint transfer for weight patching. The purple ${allTokenWeightPatchSelected() ? "row" : "squares"} encode no result.`;
  } else if (patchLoadError) {
    document.getElementById("patch-explanation").textContent = `A measured artifact exists for this selection, but its data file could not be loaded (${patchLoadError}). No fallback value is shown.`;
  } else if (patchLoading) {
    document.getElementById("patch-explanation").textContent = "A measured artifact exists for this selection and is loading. The temporary purple hatch encodes no probability or delta.";
  } else if (!patch.processed) {
    document.getElementById("patch-explanation").textContent = state.patchMode === "checkpoint" && donor === recipient
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
  document.getElementById("patch-outcome-control").hidden =
    representationAlignmentSelected() || weightAnalysis;
  document.getElementById("source-rendered-prompt").textContent = patch.sourceRenderedPrompt;
  document.getElementById("recipient-rendered-prompt").textContent = patch.recipientRenderedPrompt;
  if (!weightAnalysis) {
    renderActivationExamples(patch);
    scheduleActivationExampleLoads();
    scheduleVocabularyLensLoads();
  }
  scheduleSelectedPatchLoad();
  scheduleSelectedWeightDetailsLoad();
  scheduleFullPatchPreload();
  if (state.patchTooltipPinned) {
    window.requestAnimationFrame(restorePinnedHeatTooltip);
  }
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
  state.data.activation_example_manifest ??= {};
  state.data.real_activation_example_files ??= 0;
  state.data.activation_example_chunks ??= 0;
  state.data.vocabulary_logit_lens_manifest ??= {};
  state.data.real_vocabulary_logit_lens_files ??= 0;
  state.data.vocabulary_logit_lens_chunks ??= 0;
  state.data.representation_alignment_manifest ??= {};
  state.data.real_representation_alignment_files ??= 0;
  state.data.representation_alignment_scales ??= {};
  state.data.weight_alignment_manifest ??= {};
  state.data.real_weight_alignment_files ??= 0;
  state.data.weight_alignment_scales ??= {};
  state.data.weight_alignment_axes ??= {};
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
  const patchVisualization = document.getElementById("patch-visualization-select");
  patchVisualization.value = state.patchVisualization;
  patchVisualization.addEventListener("change", () => {
    state.patchVisualization = patchVisualization.value;
    renderAll();
  });
  const patchMode = document.getElementById("patch-mode-select");
  patchMode.value = state.patchMode;
  patchMode.addEventListener("change", () => {
    state.patchMode = patchMode.value;
    renderAll();
  });
  const activationExampleSource = document.getElementById(
    "activation-example-source-select",
  );
  activationExampleSource.value = state.activationExampleSource;
  activationExampleSource.addEventListener("change", () => {
    state.activationExampleSource = activationExampleSource.value;
    renderActivationExamples(patchData());
    scheduleActivationExampleLoads();
  });
  setupButtons("#curve-metric-controls", "curveMetric", "curveMetric", renderCurve);
  setupButtons("#curve-time-scale-controls", "curveTimeScale", "curveTimeScale", () => {
    renderCheckpointTicks();
    renderAll();
  });
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
