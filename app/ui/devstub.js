/* Browser-only stub so the UI can be rendered/screenshotted outside Tauri.
   Loaded by shot.sh; never shipped (index.html doesn't reference it). */
window.__TAURI__ = {
  core: {
    invoke: async (cmd) => ({
      defaults: { engine: "C:\\Users\\you\\AppData\\Local\\osu2bs\\engine",
                  out_dir: "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Beat Saber\\Beat Saber_Data\\CustomLevels",
                  windows: true },
      setup_status: (location.hash === "#setup"
        ? { complete: false, engine: true, python: false, torch: false,
            deps: false, ffmpeg: false, model: false, device: "cuda:0",
            gpus: ["NVIDIA GeForce RTX 5080"] }
        : { complete: true, engine: true, python: true, torch: true,
            deps: true, ffmpeg: true, model: true, device: "cuda:0",
            gpus: ["NVIDIA GeForce RTX 5080"] }),
      expand_dir: [],
      finalize_output: null,
      set_device: null,
    }[cmd]),
  },
  event: { listen: async () => () => {} },
  dialog: { open: async () => null },
  window: { getCurrentWindow: () => ({ close() {}, minimize() {}, toggleMaximize() {} }) },
};
window.addEventListener("load", () => setTimeout(() => {
  addFiles(["C:/Music/Electric Callboy - RATATATA.mp3", "C:/Music/Electric Callboy - WE GOT THE MOVES.mp3"]);
  setTimeout(() => {
    state.active = 0;
    state.queue[0].status = "Charting Expert+…";
    state.queue[0].phase = 55;
    renderQueue();
  }, 60);
}, 60));
if (location.hash === "#sheet") window.addEventListener("load",
  () => setTimeout(() => openSheet(), 150));
