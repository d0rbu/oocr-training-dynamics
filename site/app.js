"use strict";

const DATA_URL = "data/experiment.json?v=20260715q";
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
};
const PATCH_INTERFACE_DESCRIPTIONS = {
  resid_post: "Decoder-block output after both attention and MLP residual additions.",
  attention_input: "The hidden vector passed into self-attention. OLMo receives the raw residual; Qwen receives its input-RMS-normalized form.",
  attention_output: "Self-attention output after the O projection, before branch normalization or residual addition.",
  mlp_input: "The hidden vector passed into the gated MLP. OLMo receives the post-attention residual; Qwen receives its RMS-normalized form.",
  mlp_output: "MLP output after the down projection, before branch normalization or residual addition.",
};
const SLIDER_UNITS = 10000;
const ALL_FUNCTIONS_ID = "__all__";
const PATCH_SOURCE_PREFETCH_CONCURRENCY = 3;
const PATCH_CHUNK_CACHE_LIMIT = 40;
const patchChunks = new Map();
const patchChunkLoads = new Map();
const patchChunkErrors = new Map();
let patchLoadTimer = null;
let patchSourcePrefetchTimer = null;
let patchSourcePrefetchSignature = null;
let patchSourcePrefetchQueue = [];
let patchSourcePrefetchActive = 0;
const state = {
  data: null,
  model: "olmo3-7b",
  condition: "correct",
  curveMetric: "correct_probability",
  curveTimeScale: "logarithmic",
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

function usesCheckpointDonor() {
  return state.patchMode === "checkpoint";
}

function resolvedArtifactMode() {
  if (!usesCheckpointDonor()) return "across_sample";
  if (state.donorIndex < state.recipientIndex) return "across_time";
  if (state.donorIndex > state.recipientIndex) return "later_checkpoint";
  return null;
}

function selectedPatchReference() {
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

function currentPatchSource() {
  const interfaceManifest = state.data.patch_manifest?.[state.model]?.[state.condition]
    ?.[state.patchInterface] ?? {};
  const currentRecipient = state.data.checkpoints[state.recipientIndex];
  const references = [];
  if (!usesCheckpointDonor()) {
    Object.entries(interfaceManifest.across_sample ?? {}).forEach(([recipient, donors]) => {
      Object.values(donors).forEach((reference) => {
        references.push({ recipient: Number(recipient), reference });
      });
    });
  } else {
    const donor = state.data.checkpoints[state.donorIndex];
    ["across_time", "later_checkpoint"].forEach((mode) => {
      Object.entries(interfaceManifest[mode] ?? {}).forEach(([recipient, donors]) => {
        const reference = donors[String(donor)];
        if (reference) references.push({ recipient: Number(recipient), reference });
      });
    });
  }
  const unique = new Map();
  references
    .sort((left, right) => (
      Math.abs(left.recipient - currentRecipient) - Math.abs(right.recipient - currentRecipient)
    ))
    .forEach(({ reference }) => unique.set(patchReferenceKey(reference), reference));
  const donorStep = usesCheckpointDonor()
    ? state.data.checkpoints[state.donorIndex]
    : "different-function-name";
  return {
    signature: [state.model, state.condition, state.patchInterface, state.patchMode, donorStep].join("|"),
    references: [...unique.values()],
  };
}

function updatePatchPrefetchStatus(source = currentPatchSource()) {
  const status = document.getElementById("patch-prefetch-status");
  const keys = source.references.map(patchReferenceKey);
  const ready = keys.filter((key) => patchChunks.has(key)).length;
  const loading = keys.filter((key) => patchChunkLoads.has(key)).length;
  const failed = keys.filter((key) => patchChunkErrors.has(key)).length;
  if (keys.length === 0) {
    status.textContent = "Background source cache · no measured grids available yet.";
  } else if (ready === keys.length) {
    status.textContent = `Background source cache ready · ${ready}/${keys.length} measured grids.`;
  } else {
    const loadingText = loading ? ` · ${loading} loading` : "";
    const failedText = failed ? ` · ${failed} failed` : "";
    status.textContent = `Background source cache · ${ready}/${keys.length} ready${loadingText}${failedText}.`;
  }
}

function trimPatchChunkCache() {
  if (patchChunks.size <= PATCH_CHUNK_CACHE_LIMIT) return;
  const protectedKeys = new Set(
    currentPatchSource().references.map(patchReferenceKey),
  );
  protectedKeys.add(patchReferenceKey(selectedPatchReference()));
  for (const key of patchChunks.keys()) {
    if (patchChunks.size <= PATCH_CHUNK_CACHE_LIMIT) break;
    if (!protectedKeys.has(key)) patchChunks.delete(key);
  }
}

async function loadPatchChunk(reference) {
  const key = patchReferenceKey(reference);
  if (!key || patchChunks.has(key) || patchChunkErrors.has(key)) return;
  if (patchChunkLoads.has(key)) {
    await patchChunkLoads.get(key);
    return;
  }
  const request = fetch(`${reference.url}?v=${key.slice(0, 16)}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((records) => {
      if (!records || typeof records !== "object" || Array.isArray(records)) {
        throw new Error("patch chunk is not a function-record object");
      }
      patchChunks.set(key, records);
      patchChunkErrors.delete(key);
      trimPatchChunkCache();
    })
    .catch((error) => {
      patchChunkErrors.set(key, String(error.message ?? error));
    })
    .finally(() => {
      patchChunkLoads.delete(key);
      if (patchReferenceKey(selectedPatchReference()) === key) renderPatching();
      updatePatchPrefetchStatus();
    });
  patchChunkLoads.set(key, request);
  await request;
}

function drainPatchSourcePrefetch() {
  while (
    patchSourcePrefetchActive < PATCH_SOURCE_PREFETCH_CONCURRENCY
    && patchSourcePrefetchQueue.length > 0
  ) {
    const reference = patchSourcePrefetchQueue.shift();
    const key = patchReferenceKey(reference);
    if (!key || patchChunks.has(key) || patchChunkErrors.has(key)) continue;
    patchSourcePrefetchActive += 1;
    void loadPatchChunk(reference).finally(() => {
      patchSourcePrefetchActive -= 1;
      drainPatchSourcePrefetch();
    });
  }
}

function schedulePatchSourcePrefetch() {
  const source = currentPatchSource();
  updatePatchPrefetchStatus(source);
  if (source.signature === patchSourcePrefetchSignature) return;
  if (patchSourcePrefetchTimer !== null) window.clearTimeout(patchSourcePrefetchTimer);
  patchSourcePrefetchTimer = window.setTimeout(() => {
    patchSourcePrefetchTimer = null;
    patchSourcePrefetchSignature = source.signature;
    const selectedKey = patchReferenceKey(selectedPatchReference());
    patchSourcePrefetchQueue = source.references
      .filter((reference) => patchReferenceKey(reference) !== selectedKey);
    drainPatchSourcePrefetch();
  }, 250);
}

function scheduleSelectedPatchLoad() {
  if (patchLoadTimer !== null) {
    window.clearTimeout(patchLoadTimer);
    patchLoadTimer = null;
  }
  const reference = selectedPatchReference();
  const key = patchReferenceKey(reference);
  if (!key || patchChunks.has(key) || patchChunkErrors.has(key)) return;
  patchLoadTimer = window.setTimeout(() => {
    patchLoadTimer = null;
    void loadPatchChunk(reference);
  }, 100);
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

function unprocessedPatchForFunction(functionId) {
  const layers = state.data.models[state.model].layer_count;
  const fnIndex = state.data.functions.findIndex((fn) => fn.id === functionId);
  const fn = state.data.functions[fnIndex];
  const exactAxis = state.data.token_axes?.[state.model]?.[tokenAxisMode()]?.[functionId];
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
  return {
    layers,
    tokenPositions,
    recipient: null,
    source: null,
    matrix: tokenPositions.map(() => Array(layers).fill(null)),
    target: fn.definition,
    outcomeLabel: "correct-implementation probability",
    sourceFunctionId: exactAxis?.source_function_id ?? (
      state.patchMode === "across_sample"
        ? state.data.functions[(fnIndex + 1) % state.data.functions.length].id
        : fn.id
    ),
    recipientFunctionId: exactAxis?.recipient_function_id ?? fn.id,
    sourceRenderedPrompt: exactAxis?.source_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    recipientRenderedPrompt: exactAxis?.recipient_rendered_prompt ?? "Exact tokenizer metadata is unavailable for this provisional model.",
    measured: false,
    processed: false,
    aggregate: false,
    functionCount: 1,
  };
}

function measuredPatchForFunction(functionId) {
  const key = patchReferenceKey(selectedPatchReference());
  const record = key ? patchChunks.get(key)?.[functionId] : null;
  if (!record) return null;
  let matrix;
  let tokenPositions;
  let layerCount;
  if (Array.isArray(record.probabilities) && Array.isArray(record.token_positions)) {
    matrix = record.probabilities;
    layerCount = matrix[0].length;
    tokenPositions = record.token_positions.map((position) => ({
      reverseIndex: position.reverse_index,
      sourceIndex: position.source_index,
      recipientIndex: position.recipient_index,
      sourceTokenId: position.source_token_id,
      recipientTokenId: position.recipient_token_id,
      sourceToken: position.source_token,
      recipientToken: position.recipient_token,
    }));
  } else {
    layerCount = Math.max(...record.cells.map((cell) => cell.layer)) + 1;
    const tokenCount = Math.max(...record.cells.map((cell) => cell.token_reverse_index)) + 1;
    matrix = Array.from({ length: tokenCount }, () => Array(layerCount).fill(Number.NaN));
    tokenPositions = Array(tokenCount).fill(null);
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
  }
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
    processed: true,
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
  const functionCount = patches.length;
  const processed = patches.every((patch) => patch.processed);
  const sharedTokenCount = Math.min(...patches.map((patch) => patch.tokenPositions.length));
  const tokenPositions = Array.from({ length: sharedTokenCount }, (_, reverseIndex) => {
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
    const sameCoordinate = position.aggregate
      ? position.sourceTokenSignature === position.recipientTokenSignature
      : position.sourceToken === position.recipientToken
        && position.sourceIndex === position.recipientIndex
        && position.sourceTokenId === position.recipientTokenId;
    const sourcePrefix = state.patchMode === "across_sample" ? "dirty/source " : "source ";
    const recipientPrefix = state.patchMode === "across_sample" ? "clean/recipient " : "recipient ";
    const sourceCoordinate = position.aggregate
      ? aggregateTokenCoordinate(sourcePrefix, position.sourceToken)
      : tokenCoordinate(sourcePrefix, position.sourceIndex, position.sourceTokenId, position.sourceToken);
    const recipientCoordinate = position.aggregate
      ? aggregateTokenCoordinate(recipientPrefix, position.recipientToken)
      : tokenCoordinate(recipientPrefix, position.recipientIndex, position.recipientTokenId, position.recipientToken);
    const tokenText = sameCoordinate
      ? (position.aggregate
        ? aggregateTokenCoordinate("", position.sourceToken)
        : tokenCoordinate("", position.sourceIndex, position.sourceTokenId, position.sourceToken))
      : `${sourceCoordinate} → ${recipientCoordinate}`;
    const label = el("div", { class: `heatmap-token${position.reverseIndex === 0 ? " anchor" : ""}` });
    label.append(el("b", {}, position.reverseIndex === 0 ? "−0 · end" : `−${position.reverseIndex}`));
    label.append(el("span", { title: tokenText }, tokenText));
    heatmap.append(label);
    for (let layer = 0; layer < patch.layers; layer += 1) {
      const probability = patch.matrix[tokenIndex][layer];
      const cell = el("div", { class: "heat-cell", tabindex: "0" });
      if (!patch.processed) {
        cell.classList.add("unprocessed");
        const unavailableReason = patchLoadError
          ? "A measured file exists, but it could not be loaded. No fallback value is displayed."
          : patchLoading
            ? "Measured values are loading. No temporary numeric value is displayed."
            : "No activation-patching value has been measured for this recipient/donor selection.";
        bindHeatTooltip(cell, `<b>No displayed value</b><br>Layer ${layer} · reverse token −${position.reverseIndex}<br><br>${unavailableReason}`);
        cell.setAttribute("aria-label", `layer ${layer}, reverse token ${position.reverseIndex}, unprocessed`);
        heatmap.append(cell);
        continue;
      }
      const delta = probability - patch.recipient;
      const value = state.patchMetric === "probability" ? probability : delta / .25;
      cell.style.background = colorFor(value, state.patchMetric);
      const display = state.patchMetric === "probability" ? formatPercent(probability) : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`;
      const averagingNote = patch.aggregate ? `<br>cellwise mean over n=${patch.functionCount} functions` : "";
      bindHeatTooltip(cell, `<b>Layer ${layer} · reverse token −${position.reverseIndex}</b>${averagingNote}<br>${escapeHtml(sourceCoordinate)}<br>${escapeHtml(recipientCoordinate)}<br><br>${patch.outcomeLabel}: ${formatPercent(probability)}<br>change from recipient: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} points`);
      cell.setAttribute("aria-label", `layer ${layer}, reverse token ${position.reverseIndex}, ${display}`);
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
  patchStatus.textContent = (patch.processed
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
    document.getElementById("source-question-label").textContent = donor < recipient
      ? "earlier donor checkpoint"
      : donor > recipient
        ? "later donor checkpoint"
        : "same donor checkpoint";
    document.getElementById("source-question").textContent = `${questionCount} · ${donor === 0 ? "frozen base" : `step ${donor}`}`;
    if (donor < recipient) {
      document.getElementById("patch-explanation").textContent = "Replacing a later recipient’s selected activation with an earlier donor state tests where newly acquired OOCR information is causally necessary. The remaining computation uses the recipient checkpoint’s weights.";
    } else if (donor > recipient) {
      document.getElementById("patch-explanation").textContent = "Injecting a later donor activation into an earlier recipient—including the frozen base—tests where the learned state is sufficient to boost the correct OOCR answer. The remaining computation uses the recipient checkpoint’s weights.";
    } else {
      document.getElementById("patch-explanation").textContent = "Recipient and donor are the same checkpoint. This identity cell is not run or assigned a value because replacing an activation with itself should leave the answer unchanged.";
    }
  } else {
    const dirty = patch.aggregate
      ? null
      : state.data.functions.find((item) => item.id === patch.sourceFunctionId);
    document.getElementById("source-question-label").textContent = "dirty activation source";
    document.getElementById("source-question").textContent = patch.aggregate
      ? `Mean over all ${patch.functionCount} fixed-derangement dirty-name questions`
      : `What is the definition of ${dirty.alias}?`;
    document.getElementById("patch-explanation").textContent = "Patching dirty-name states into the clean prompt tests where the alternate identity suppresses the correct implementation. Cells show P(correct) directly, so a successful corruption moves downward.";
  }
  if (patchLoadError) {
    document.getElementById("patch-explanation").textContent = `A measured artifact exists for this selection, but its data file could not be loaded (${patchLoadError}). No fallback value is shown.`;
  } else if (patchLoading) {
    document.getElementById("patch-explanation").textContent = "A measured artifact exists for this selection and is loading. The temporary purple hatch encodes no probability or delta.";
  } else if (!patch.processed) {
    document.getElementById("patch-explanation").textContent = "This selection has not been processed. The purple hatched squares are availability markers only: they encode no probability, delta, interpolation, or synthetic result.";
  }
  document.getElementById("source-rendered-prompt").textContent = patch.sourceRenderedPrompt;
  document.getElementById("recipient-rendered-prompt").textContent = patch.recipientRenderedPrompt;
  scheduleSelectedPatchLoad();
  schedulePatchSourcePrefetch();
}

function renderAll() {
  const maxIndex = curveRows().length - 1;
  state.checkpointIndex = Math.min(state.checkpointIndex, maxIndex);
  normalizePatchCheckpointIndices();
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
}

initialize().catch((error) => {
  console.error(error);
  const warning = document.getElementById("warning-banner");
  warning.hidden = false;
  warning.textContent = `The visualization data could not be loaded: ${error.message}`;
});
