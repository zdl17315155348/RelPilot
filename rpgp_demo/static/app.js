const textInput = document.querySelector("#text-input");
const extractButton = document.querySelector("#extract-button");
const examplesNode = document.querySelector("#examples");
const relationsNode = document.querySelector("#relations");
const highlightedNode = document.querySelector("#highlighted-text");
const triplesBody = document.querySelector("#triples-body");
const jsonOutput = document.querySelector("#json-output");
const relationCount = document.querySelector("#relation-count");
const tripleCount = document.querySelector("#triple-count");
const reductionCount = document.querySelector("#reduction-count");
const reductionRatio = document.querySelector("#reduction-ratio");
const baselineRelations = document.querySelector("#baseline-relations");
const rpgpRelations = document.querySelector("#rpgp-relations");
const skippedRelations = document.querySelector("#skipped-relations");
const copyJson = document.querySelector("#copy-json");

let latestPayload = null;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderHighlightedText(text, spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  let cursor = 0;
  let html = "";
  for (const span of ordered) {
    if (span.start < cursor) {
      continue;
    }
    html += escapeHtml(text.slice(cursor, span.start));
    html += `<span class="${span.role}">${escapeHtml(text.slice(span.start, span.end))}</span>`;
    cursor = span.end;
  }
  html += escapeHtml(text.slice(cursor));
  highlightedNode.innerHTML = html || "暂无文本";
}

function renderResult(payload) {
  latestPayload = payload;
  relationCount.textContent = `预判关系：${payload.predicted_relations.length}`;
  tripleCount.textContent = `三元组：${payload.triples.length}`;
  const stats = payload.relation_stats || {
    total_relations: 0,
    skipped_relations: 0,
    reduction_ratio: 0,
  };
  reductionCount.textContent = `跳过解码：${stats.skipped_relations}/${stats.total_relations}`;
  reductionRatio.textContent = `节省比例：${(stats.reduction_ratio * 100).toFixed(1)}%`;
  baselineRelations.textContent = stats.total_relations;
  rpgpRelations.textContent = stats.predicted_relations;
  skippedRelations.textContent = stats.skipped_relations;
  relationsNode.innerHTML = payload.predicted_relations
    .map((item) => `<span>${escapeHtml(item.name)} ${item.confidence.toFixed(2)}</span>`)
    .join("") || '<span class="empty-chip">未预判到关系</span>';
  renderHighlightedText(payload.text, payload.spans);
  triplesBody.innerHTML = payload.triples
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.subject)}</td>
        <td>${escapeHtml(item.relation)}</td>
        <td>${escapeHtml(item.object)}</td>
        <td>${item.confidence.toFixed(2)}</td>
      </tr>
    `)
    .join("") || '<tr><td colspan="4" class="empty-cell">暂无三元组</td></tr>';
  jsonOutput.textContent = JSON.stringify(payload, null, 2);
}

async function extract() {
  extractButton.disabled = true;
  extractButton.textContent = "抽取中...";
  try {
    const response = await fetch("/api/extract", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: textInput.value}),
    });
    renderResult(await response.json());
  } finally {
    extractButton.disabled = false;
    extractButton.textContent = "抽取";
  }
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  const payload = await response.json();
  examplesNode.innerHTML = "";
  for (const example of payload.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.title;
    button.addEventListener("click", () => {
      textInput.value = example.text;
      extract();
    });
    examplesNode.appendChild(button);
  }
  if (payload.examples.length > 0) {
    textInput.value = payload.examples[0].text;
    extract();
  }
}

extractButton.addEventListener("click", extract);
copyJson.addEventListener("click", async () => {
  if (latestPayload) {
    await navigator.clipboard.writeText(JSON.stringify(latestPayload, null, 2));
    copyJson.textContent = "已复制";
    setTimeout(() => {
      copyJson.textContent = "复制JSON";
    }, 1200);
  }
});

loadExamples();
