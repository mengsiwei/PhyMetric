const DATA_URL = "../data/generated/captions_2000.jsonl";

const state = {
  records: [],
  currentIndex: 0,
  changes: 0,
};

const el = {
  stats: document.getElementById("stats"),
  record: document.getElementById("record"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  jumpInput: document.getElementById("jumpInput"),
  jumpBtn: document.getElementById("jumpBtn"),
  addQuestionBtn: document.getElementById("addQuestionBtn"),
  loadFileBtn: document.getElementById("loadFileBtn"),
  fileInput: document.getElementById("fileInput"),
  downloadBtn: document.getElementById("downloadBtn"),
  log: document.getElementById("log"),
};

function setLog(text) {
  const body = el.log.querySelector(".log-body");
  body.textContent = text;
}

function updateStats() {
  const total = state.records.length;
  const idx = total ? state.currentIndex + 1 : 0;
  el.stats.textContent = `${idx} / ${total}  |  Changes: ${state.changes}`;
}

function parseJsonl(text) {
  const lines = text.split(/\n+/).filter((line) => line.trim().length > 0);
  return lines.map((line, i) => {
    try {
      return JSON.parse(line);
    } catch (err) {
      throw new Error(`Failed to parse JSONL at line ${i + 1}: ${err.message}`);
    }
  });
}

function getRecord() {
  return state.records[state.currentIndex];
}

function markChanged(message) {
  state.changes += 1;
  setLog(message);
  updateStats();
}

function createInput(label, value, onChange) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";

  const title = document.createElement("span");
  title.textContent = label;
  title.className = "field-title";

  const input = document.createElement("input");
  input.value = value || "";
  input.className = "input";
  input.addEventListener("input", (e) => onChange(e.target.value));

  wrapper.appendChild(title);
  wrapper.appendChild(input);
  return wrapper;
}

function createTextarea(label, value, onChange) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";

  const title = document.createElement("span");
  title.textContent = label;
  title.className = "field-title";

  const textarea = document.createElement("textarea");
  textarea.value = value || "";
  textarea.className = "textarea";
  textarea.rows = 4;
  textarea.addEventListener("input", (e) => onChange(e.target.value));

  wrapper.appendChild(title);
  wrapper.appendChild(textarea);
  return wrapper;
}

function createSelect(label, value, options, onChange) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";

  const title = document.createElement("span");
  title.textContent = label;
  title.className = "field-title";

  const select = document.createElement("select");
  select.className = "select";
  options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt.value;
    option.textContent = opt.label;
    if (String(value) === String(opt.value)) option.selected = true;
    select.appendChild(option);
  });
  select.addEventListener("change", (e) => onChange(e.target.value));

  wrapper.appendChild(title);
  wrapper.appendChild(select);
  return wrapper;
}

function renderQuestions(record) {
  const section = document.createElement("div");
  section.className = "questions";

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = `Questions (${record.questions.length})`;
  section.appendChild(title);

  record.questions.forEach((q, idx) => {
    const card = document.createElement("div");
    card.className = "question-card";

    const header = document.createElement("div");
    header.className = "question-header";

    const idTag = document.createElement("span");
    idTag.className = "chip";
    idTag.textContent = `Q${q.question_id || idx + 1}`;

    const removeBtn = document.createElement("button");
    removeBtn.className = "btn tiny danger";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => {
      record.questions.splice(idx, 1);
      markChanged(`Removed question ${q.question_id || idx + 1}`);
      render();
    });

    header.appendChild(idTag);
    header.appendChild(removeBtn);

    const grid = document.createElement("div");
    grid.className = "question-grid";

    const textField = createTextarea("question_text", q.question_text, (val) => {
      q.question_text = val;
      markChanged(`Updated question_text for ${record.caption_id}`);
    });

    const answerField = createSelect(
      "answer",
      q.answer,
      [
        { value: 1, label: "1" },
        { value: 0, label: "0" },
      ],
      (val) => {
        q.answer = Number(val);
        markChanged(`Updated answer for ${record.caption_id}`);
      }
    );

    const dimensionField = createInput("dimensions (comma)", (q.dimensions || []).join(", "), (val) => {
      q.dimensions = val
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      markChanged(`Updated dimensions for ${record.caption_id}`);
    });

    const difficultyField = createInput("difficulty", q.difficulty || "", (val) => {
      q.difficulty = val.trim();
      markChanged(`Updated difficulty for ${record.caption_id}`);
    });

    const framesField = createInput("frame_indices (comma)", (q.frame_indices || []).join(", "), (val) => {
      q.frame_indices = val
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => Number(item))
        .filter((num) => Number.isFinite(num));
      markChanged(`Updated frame_indices for ${record.caption_id}`);
    });

    grid.appendChild(textField);
    grid.appendChild(answerField);
    grid.appendChild(dimensionField);
    grid.appendChild(difficultyField);
    grid.appendChild(framesField);

    card.appendChild(header);
    card.appendChild(grid);

    section.appendChild(card);
  });

  return section;
}

function render() {
  const record = getRecord();
  el.record.innerHTML = "";

  if (!record) {
    el.record.textContent = "No record loaded.";
    updateStats();
    return;
  }

  const header = document.createElement("div");
  header.className = "record-header";

  const title = document.createElement("div");
  title.className = "record-title";
  title.textContent = record.caption_id;

  const sub = document.createElement("div");
  sub.className = "record-sub";
  sub.textContent = `${record.phenomenon || "-"}  |  ${record.base_scene_id || "-"}`;

  header.appendChild(title);
  header.appendChild(sub);

  const captionField = createTextarea("caption", record.caption, (val) => {
    record.caption = val;
    markChanged(`Updated caption for ${record.caption_id}`);
  });

  const slotsView = document.createElement("div");
  slotsView.className = "slots";
  const slotsTitle = document.createElement("div");
  slotsTitle.className = "section-title";
  slotsTitle.textContent = "Slots";
  const slotsList = document.createElement("div");
  slotsList.className = "slot-list";
  Object.entries(record.slots || {}).forEach(([key, value]) => {
    const tag = document.createElement("span");
    tag.className = "chip";
    tag.textContent = `${key}: ${value}`;
    slotsList.appendChild(tag);
  });
  slotsView.appendChild(slotsTitle);
  slotsView.appendChild(slotsList);

  el.record.appendChild(header);
  el.record.appendChild(captionField);
  el.record.appendChild(slotsView);
  el.record.appendChild(renderQuestions(record));

  updateStats();
}

function jumpToCaptionId() {
  const id = el.jumpInput.value.trim();
  if (!id) return;
  const index = state.records.findIndex((r) => r.caption_id === id);
  if (index === -1) {
    setLog(`caption_id not found: ${id}`);
    return;
  }
  state.currentIndex = index;
  render();
}

function downloadJsonl() {
  const lines = state.records.map((record) => JSON.stringify(record));
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "captions_2000.updated.jsonl";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setLog("Downloaded updated JSONL.");
}

async function init() {
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    state.records = parseJsonl(text);
    render();
    setLog("Loaded JSONL successfully.");
  } catch (err) {
    el.record.textContent = "Auto load failed. Click 'Load JSONL File' and choose captions_2000.jsonl.";
    setLog(`Auto load failed: ${err.message}`);
    updateStats();
  }
}

function handleFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = String(reader.result || "");
      state.records = parseJsonl(text);
      state.currentIndex = 0;
      render();
      setLog(`Loaded from local file: ${file.name}`);
    } catch (err) {
      el.record.textContent = `Failed to parse selected file: ${err.message}`;
      setLog("Selected file parse failed.");
      updateStats();
    }
  };
  reader.onerror = () => {
    setLog("Failed to read selected file.");
  };
  reader.readAsText(file, "utf-8");
}

el.prevBtn.addEventListener("click", () => {
  if (state.currentIndex > 0) state.currentIndex -= 1;
  render();
});

el.nextBtn.addEventListener("click", () => {
  if (state.currentIndex < state.records.length - 1) state.currentIndex += 1;
  render();
});

el.jumpBtn.addEventListener("click", jumpToCaptionId);

el.jumpInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") jumpToCaptionId();
});

el.addQuestionBtn.addEventListener("click", () => {
  const record = getRecord();
  if (!record) return;
  record.questions.push({
    question_id: String(record.questions.length + 1).padStart(2, "0"),
    question_text: "",
    dimensions: [],
    difficulty: "",
    frame_indices: [],
    answer: 0,
  });
  markChanged(`Added question for ${record.caption_id}`);
  render();
});

el.loadFileBtn.addEventListener("click", () => {
  el.fileInput.click();
});

el.fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  handleFileSelect(file);
  e.target.value = "";
});

el.downloadBtn.addEventListener("click", downloadJsonl);

init();
