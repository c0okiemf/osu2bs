/* osu2bs UI — native inference: setup wizard, then drop songs → maps. */
const T = window.__TAURI__;
const invoke = T.core.invoke;
const $ = (id) => document.getElementById(id);

const DIFF_ORDER = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"];
const MAX_DIFFICULTIES = 5;
const DIFF_SHORT = { Easy: "Easy", Normal: "Normal", Hard: "Hard",
                     Expert: "Expert", ExpertPlus: "Expert+" };
const STEPS = [
  ["engine", "Engine code + models"],
  ["python", "Python runtime"],
  ["torch", "PyTorch"],
  ["deps", "Audio & model libraries"],
  ["ffmpeg", "ffmpeg"],
  ["model", "AI mapper (1.7 GB)"],
];

const S = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

const state = {
  // {path, name, slug, status, phase, done, err}; survives app restarts
  queue: S.get("queue", []).map((q) => ({ ...q, phase: q.done ? 100 : 0,
    soft: false, retried: false, lastErr: "",
    // anything mid-run when the app closed counts as paused
    status: q.done || q.err || ["Ready", "Stopped"].includes(q.status) ? q.status : "Paused" })),
  paused: S.get("paused", false),  // a run was paused or the app closed mid-run
  running: false,
  active: -1,
  diffs: new Set(S.get("diffs", ["ExpertPlus"])),
  plusTiers: S.get("plusTiers", 0),
  engine: S.get("engine", ""),
  outDir: S.get("outDir", ""),
  genDirs: S.get("genDirs", true),
  genZips: S.get("genZips", false),
  redo: S.get("redo", false),
  halt: null,          // "pause" keeps the song's finished tiers, "stop" discards them
  stars: S.get("stars", "5.5"),
  showLog: S.get("showLog", false),
  plName: S.get("plName", ""),
  windows: false,
  sep: "/",
  device: "cpu",
  gpus: [],
  settingUp: false,
};
// Migrate old selections that could exceed the game's five unique slots.
state.diffs = new Set([...state.diffs].filter((d) => DIFF_ORDER.includes(d)));
state.plusTiers = Math.max(0, Math.min(Number.isInteger(state.plusTiers) ? state.plusTiers : 0,
  MAX_DIFFICULTIES - state.diffs.size));
if (!state.diffs.size && !state.plusTiers) state.diffs.add("ExpertPlus");
S.set("diffs", [...state.diffs]);
S.set("plusTiers", state.plusTiers);

/* ---------- nav gains material only when scrolled ---------- */
const sv = $("scrollview");
sv.addEventListener("scroll", () => {
  document.querySelector(".nav").classList.toggle("scrolled", sv.scrollTop > 4);
});

/* ---------- window controls ---------- */
const win = T.window.getCurrentWindow();
$("winClose").onclick = () => win.close();
$("winMin").onclick = () => win.minimize();
$("winMax").onclick = () => win.toggleMaximize();

/* ---------- boot ---------- */
(async () => {
  const d = await invoke("defaults");
  if (!state.engine) state.engine = d.engine;
  if (!state.outDir) state.outDir = d.out_dir;
  state.windows = d.windows;
  state.sep = d.windows ? "\\" : "/";
  $("setRepo").value = state.engine;
  $("setStars").value = state.stars;
  $("setLog").checked = state.showLog;
  $("logRow").hidden = !state.showLog;
  $("genDirs").checked = state.genDirs;
  $("genZips").checked = state.genZips;
  $("redo").checked = state.redo;
  $("plName").value = state.plName;
  $("plToggle").checked = S.get("plOn", true);
  $("plToggle").onchange();
  renderQueue();
  setGoLabel();
  renderOutDir();
  renderChips();
  renderPlus();
  await refreshSetup();
})();

/* ---------- setup wizard ---------- */
async function refreshSetup() {
  const st = await invoke("setup_status", { engine: state.engine });
  state.device = st.device;
  state.gpus = st.gpus;
  $("setDeviceLabel").textContent = deviceName(st.device);
  $("footNote").textContent = st.complete
    ? "Everything runs locally on " + deviceName(st.device)
    : "";
  $("mainView").hidden = !st.complete;
  $("genBar").hidden = !st.complete;
  $("setupView").hidden = st.complete;
  if (!st.complete) renderSetup(st);
  return st.complete;
}
function deviceName(dev) {
  if (dev === "cpu") return "CPU";
  const i = Number(dev.split(":")[1] || 0);
  return state.gpus[i] || "GPU " + i;
}
function renderSetup(st) {
  const list = $("deviceList");
  list.innerHTML = "";
  const opts = state.gpus.map((g, i) => ["cuda:" + i, g])
    .concat([["cpu", "CPU" + (state.gpus.length ? " (slow, no GPU used)" : "")]]);
  for (const [dev, label] of opts) {
    const row = document.createElement("div");
    row.className = "row dev-row" + (state.device === dev ? " on" : "");
    row.innerHTML = `<div class="row-text"><p class="row-title">${label}</p>
      <p class="row-sub">${dev.startsWith("cuda") ? "Fast — recommended" : "Works everywhere"}</p></div>
      <div class="dev-dot"></div>`;
    row.onclick = () => {
      state.device = dev;
      invoke("set_device", { engine: state.engine, device: dev });
      renderSetup(st);
    };
    list.appendChild(row);
  }
  const steps = $("setupSteps");
  steps.innerHTML = "";
  for (const [key, label] of STEPS) {
    const done = st[key];
    const row = document.createElement("div");
    row.className = "row col";
    row.id = "step-" + key;
    row.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center">
        <p class="row-title">${label}</p>
        <span class="step-state${done ? " ok" : ""}">${done ? "Ready" : "Waiting"}</span>
      </div>
      <p class="row-sub" data-msg></p>
      <div class="bar" hidden><i></i></div>`;
    steps.appendChild(row);
  }
}
function stepUI(key) {
  const row = $("step-" + key);
  return row && {
    state: row.querySelector(".step-state"),
    msg: row.querySelector("[data-msg]"),
    bar: row.querySelector(".bar"),
    fill: row.querySelector(".bar i"),
  };
}
T.event.listen("setup-progress", (e) => {
  const [step, pct, msg] = e.payload;
  const ui = stepUI(step);
  if (!ui) return;
  ui.msg.textContent = msg;
  if (pct >= 1) {
    ui.bar.hidden = true;
    ui.state.textContent = "Ready";
    ui.state.classList.add("ok");
  } else {
    ui.bar.hidden = false;
    ui.state.textContent = "Working…";
    if (pct < 0) { ui.bar.classList.add("indet"); }
    else { ui.bar.classList.remove("indet"); ui.fill.style.width = (pct * 100) + "%"; }
  }
});
T.event.listen("setup-done", async () => {
  state.settingUp = false;
  $("setupBtn").textContent = "Set Up";
  await refreshSetup();
});
T.event.listen("setup-error", (e) => {
  state.settingUp = false;
  $("setupBtn").textContent = "Retry Setup";
  $("setupBtn").disabled = false;
  $("setupNote").textContent = "Setup hit a snag: " + e.payload;
});
$("setupBtn").onclick = () => {
  if (state.settingUp) return;
  state.settingUp = true;
  $("setupBtn").textContent = "Setting up…";
  invoke("run_setup", { engine: state.engine, device: state.device });
};
$("setupAgainBtn").onclick = async () => {
  closeSheet();
  $("mainView").hidden = true;
  $("genBar").hidden = true;
  $("setupView").hidden = false;
  renderSetup(await invoke("setup_status", { engine: state.engine }));
};

/* ---------- difficulty chips ---------- */
function renderChips() {
  const box = $("diffChips");
  box.innerHTML = "";
  for (const name of DIFF_ORDER) {
    const b = document.createElement("button");
    b.className = "chip" + (state.diffs.has(name) ? " on" : "");
    b.textContent = DIFF_SHORT[name];
    b.disabled = !state.diffs.has(name) && chosenDiffs().length >= MAX_DIFFICULTIES;
    b.setAttribute("aria-pressed", String(state.diffs.has(name)));
    b.onclick = () => {
      if (!state.diffs.has(name) && chosenDiffs().length >= MAX_DIFFICULTIES) return;
      state.diffs.has(name) ? state.diffs.delete(name) : state.diffs.add(name);
      if (!state.diffs.size && !state.plusTiers) state.diffs.add("ExpertPlus");
      S.set("diffs", [...state.diffs]);
      renderChips();
      renderPlus();
    };
    box.appendChild(b);
  }
}
function renderPlus() {
  $("plusLabel").textContent = state.plusTiers
    ? Array.from({length: state.plusTiers}, (_, i) => `Expert${"+".repeat(i + 2)}`).join(", ")
    : "Off";
  $("plusMinus").disabled = state.plusTiers === 0;
  $("plusPlus").disabled = chosenDiffs().length >= MAX_DIFFICULTIES;
  $("diffCount").textContent = `${chosenDiffs().length} of 5 selected · Each becomes a separate difficulty`;
}
$("plusMinus").onclick = () => {
  state.plusTiers = Math.max(0, state.plusTiers - 1);
  if (!state.diffs.size && !state.plusTiers) state.diffs.add("ExpertPlus");
  S.set("diffs", [...state.diffs]); S.set("plusTiers", state.plusTiers);
  renderPlus(); renderChips();
};
$("plusPlus").onclick = () => {
  if (chosenDiffs().length >= MAX_DIFFICULTIES) return;
  state.plusTiers++;
  S.set("plusTiers", state.plusTiers); renderPlus(); renderChips();
};

function chosenDiffs() {
  const out = DIFF_ORDER.filter((d) => state.diffs.has(d));
  for (let n = 2; n <= state.plusTiers + 1; n++) out.push("ExpertPlus" + n);
  return out.length ? out : ["ExpertPlus"];
}

/* ---------- output folder + artifact toggles ---------- */
function renderOutDir() {
  $("outDirLabel").textContent = state.outDir || "Choose a folder…";
}
async function pickOutDir() {
  const sel = await T.dialog.open({ directory: true });
  if (sel) {
    state.outDir = sel;
    S.set("outDir", sel);
    renderOutDir();
  }
  return !!state.outDir;
}
$("outDirBtn").onclick = pickOutDir;
$("redo").onchange = () => { state.redo = $("redo").checked; S.set("redo", state.redo); };

for (const [id, key, other] of [["genDirs", "genDirs", "genZips"],
                                ["genZips", "genZips", "genDirs"]]) {
  $(id).onchange = () => {
    if (!$(id).checked && !$(other).checked) {
      $(id).checked = true;             // at least one artifact stays on
      return;
    }
    state[key] = $(id).checked;
    S.set(key, state[key]);
  };
}

/* ---------- file intake ---------- */
const AUDIO = /\.(mp3|ogg|wav|m4a|flac|egg)$/i;
// Unicode-aware (ASCII \w collapsed every Cyrillic title to "-"), length-capped,
// and unique within the queue so two songs never share an output folder.
function slug(name) {
  const base = name.normalize("NFC").replace(/\.[^.]+$/, "")
    .replace(/[^\p{L}\p{N}\p{M}_\s-]/gu, "").trim().replace(/\s+/g, "_")
    .slice(0, 100).replace(/^[_-]+|[_-]+$/g, "") || "song";
  let s = base;
  for (let n = 2; state.queue.some((q) => q.slug.toLowerCase() === s.toLowerCase()); n++)
    s = `${base}_${n}`;
  return s;
}
async function addFiles(paths) {
  const files = [];
  for (const p of paths) {
    if (AUDIO.test(p)) files.push(p);
    else files.push(...await invoke("expand_dir", { path: p }).catch(() => []));
  }
  for (const p of files) {
    if (!AUDIO.test(p) || state.queue.some((q) => q.path === p)) continue;
    const name = p.split(/[\\/]/).pop();
    state.queue.push({ path: p, name, slug: slug(name), status: "Ready",
                       phase: 0, done: false, err: false, soft: false,
                       retried: false, lastErr: "" });
  }
  renderQueue();
  $("goBtn").disabled = state.running || !state.queue.some((q) => !q.done);
}

T.event.listen("tauri://drag-drop", (e) => { $("dropZone").classList.remove("hover"); addFiles(e.payload.paths || []); });
T.event.listen("tauri://drag-enter", () => $("dropZone").classList.add("hover"));
T.event.listen("tauri://drag-leave", () => $("dropZone").classList.remove("hover"));

$("pickBtn").onclick = async () => {
  const sel = await T.dialog.open({
    multiple: true,
    filters: [{ name: "Audio", extensions: ["mp3", "ogg", "wav", "m4a", "flac", "egg"] }],
  });
  if (sel) addFiles(Array.isArray(sel) ? sel : [sel]);
};
$("pickDirBtn").onclick = async () => {
  const sel = await T.dialog.open({ directory: true });
  if (sel) addFiles([sel]);
};

/* ---------- queue rendering ---------- */
function renderQueue() {
  S.set("queue", state.queue.map(({ path, name, slug, status, done, err }) =>
    ({ path, name, slug, status, done, err })));
  const any = state.queue.length > 0;
  $("queueLabel").hidden = !any;
  $("queueCard").hidden = !any;
  const list = $("queueList");
  list.innerHTML = "";
  state.queue.forEach((q, i) => {
    const row = document.createElement("div");
    row.className = "q-row" + (q.err ? " err" : q.done ? " ok" : "");
    const right = q.err ? '<span class="q-x">✕</span>'
      : q.done ? '<span class="q-check">✓</span>'
      : (i === state.active ? '<span class="spin"></span>' : "");
    row.innerHTML = `
      <div class="q-art">♪</div>
      <div class="q-main">
        <p class="q-name">${q.name}</p>
        <p class="q-status">${q.status}</p>
        ${i === state.active ? '<div class="q-bar"><i style="width:' + q.phase + '%"></i></div>' : ""}
      </div>${right}`;
    list.appendChild(row);
  });
}

/* ---------- status translation ---------- */
const PHASES = [
  [/Precomputing encoder|Generating timing/, "Listening to the song…", 15],
  [/Generating map/, "Writing the rhythm…", 35],
  [/structural reuse/, "Learning the song's structure…", 45],
  [/-- (Expert\++|Easy|Normal|Hard|Expert)/, (m) => `Charting ${m[1]}…`, 55],
  [/seed \d/, "Choosing the best take…", 70],
  [/density calib|off band/, "Balancing difficulty…", 80],
  [/lean walls/, "Final checks…", 92],
];
function onLine(line) {
  if (state.showLog) {
    const log = $("log");
    log.textContent = (log.textContent + "\n" + line).split("\n").slice(-400).join("\n");
    log.scrollTop = log.scrollHeight;
  }
  const q = state.queue[state.active];
  if (!q) return;
  for (const [re, label, pct] of PHASES) {
    const m = line.match(re);
    if (m) {
      q.status = typeof label === "function" ? label(m) : label;
      q.phase = Math.max(q.phase, pct);
      renderQueue();
      return;
    }
  }
  // scary lines only get remembered; the exit code is the verdict
  if (/CHECKS FAILED/.test(line)) q.soft = true;
  else if (/Traceback|inference failed|Error(:| )/.test(line)) q.lastErr = line;
}

/* ---------- generation ---------- */
let unlisteners = [];
async function generate() {
  if (state.running) return halt("pause");
  if (!state.outDir && !(await pickOutDir())) return;
  state.running = true;
  state.paused = true; S.set("paused", true);  // until the run completes
  $("clearBtn").disabled = true;
  $("goBtn").textContent = "Pause";
  $("stopBtn").hidden = false;
  const diffs = chosenDiffs().join(",");
  const playlist = $("plToggle").checked
    ? ($("plName").value.trim() || "osu2bs")
    : "";
  if (playlist) { state.plName = $("plName").value.trim(); S.set("plName", state.plName); }

  unlisteners.push(await T.event.listen("pipeline-line", (e) => onLine(e.payload[1])));

  for (let i = 0; i < state.queue.length; i++) {
    const q = state.queue[i];
    if (state.halt) break;
    if (q.done) continue;
    if (!state.redo && await invoke("output_done", { outDir: state.outDir, slug: q.slug,
        dirs: state.genDirs, zips: state.genZips }).catch(() => false)) {
      q.done = true; q.phase = 100; q.status = "Already generated";
      renderQueue();
      continue;
    }
    state.active = i;
    q.status = "Preparing…"; q.phase = 5; q.err = false;
    q.soft = false; q.lastErr = "";
    renderQueue();
    try {
      const args = [q.path, state.outDir + state.sep + q.slug,
                    String(state.stars || "5.5"), "--diffs", diffs];
      if (playlist) args.push("--playlist", playlist);
      const code = await new Promise(async (resolve, reject) => {
        const un = await T.event.listen("pipeline-done", (e) => { un(); resolve(e.payload); });
        invoke("run_pipeline", { engine: state.engine, device: state.device, args })
          .then(() => state.halt && invoke("cancel_pipeline"))  // halt raced the spawn
          .catch((err) => { un(); reject(err); });
      });
      if (state.halt) {
        q.phase = 0;
        q.status = state.halt === "pause" ? "Paused" : "Stopped";
        if (state.halt === "stop") await discard(q);
        break;
      }
      if (code === 0) {
        await invoke("finalize_output", { outDir: state.outDir, slug: q.slug,
          keepDirs: state.genDirs, keepZips: state.genZips }).catch(() => {});
        q.done = true; q.phase = 100;
        q.status = q.soft ? "Done — map has rough edges" : "Done";
      } else if (!q.retried) {
        q.retried = true;           // transient GPU hiccups deserve one retry
        q.status = "Retrying…"; q.phase = 0; q.soft = false;
        renderQueue();
        i--; continue;
      } else {
        q.err = true;
        q.status = "Failed — " + (q.lastErr || "see log");
      }
    } catch (err) {
      q.err = true;
      q.status = String(err);
    }
    renderQueue();
  }
  state.active = -1;
  state.running = false;
  state.paused = state.halt === "pause"; S.set("paused", state.paused);
  state.halt = null;
  $("clearBtn").disabled = false;
  setGoLabel();
  unlisteners.forEach((u) => u());
  unlisteners = [];
  renderQueue();
}
$("goBtn").onclick = generate;
function setGoLabel() {
  $("goBtn").textContent = state.paused ? "Resume" : "Generate";
  $("goBtn").disabled = !state.queue.some((q) => !q.done);
  $("stopBtn").hidden = !state.paused;
}
async function halt(kind) {
  state.halt = kind;
  $("goBtn").disabled = $("stopBtn").disabled = true;
  await invoke("cancel_pipeline");
  $("stopBtn").disabled = false;
}
const discard = (q) => invoke("discard_partial", { outDir: state.outDir, slug: q.slug })
  .catch(() => {});
$("stopBtn").onclick = async () => {
  if (state.running) return halt("stop");
  for (const q of state.queue)       // abandon a paused run: drop its partial song
    if (q.status === "Paused") { await discard(q); q.status = "Stopped"; }
  state.paused = false; S.set("paused", false);
  renderQueue(); setGoLabel();
};
$("clearBtn").onclick = () => {
  if (state.running) return;
  state.queue = []; state.paused = false; S.set("paused", false);
  renderQueue(); setGoLabel();
};

/* ---------- settings sheet ---------- */
function openSheet() {
  $("sheetBack").hidden = false; $("sheet").hidden = false;
  requestAnimationFrame(() => { $("sheetBack").classList.add("show"); $("sheet").classList.add("show"); });
}
function closeSheet() {
  state.engine = $("setRepo").value.trim();
  state.stars = $("setStars").value.trim();
  state.showLog = $("setLog").checked;
  ["engine", "stars", "showLog"].forEach((k) => S.set(k, state[k]));
  $("logRow").hidden = !state.showLog;
  $("sheetBack").classList.remove("show"); $("sheet").classList.remove("show");
  setTimeout(() => { $("sheetBack").hidden = true; $("sheet").hidden = true; }, 330);
}
$("settingsBtn").onclick = openSheet;
$("sheetDone").onclick = closeSheet;
$("sheetBack").onclick = closeSheet;
$("plToggle").onchange = () => {
  $("plNameRow").style.display = $("plToggle").checked ? "" : "none";
  S.set("plOn", $("plToggle").checked);
};
