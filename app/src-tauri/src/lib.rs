//! osu2bs desktop shell — fully native: the inference engine (Python code +
//! small models) ships embedded in this binary and unpacks to a local app
//! dir; the heavy runtime (Python, PyTorch for the chosen device, ffmpeg,
//! the Mapperatorinator checkpoint) downloads during a first-run setup with
//! progress events. No WSL anywhere.

use serde::Serialize;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

const ENGINE_ZIP: &[u8] = include_bytes!("../engine.zip");
const PY_EMBED_URL: &str =
    "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip";
const PY_LINUX_URL: &str = "https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.11.9%2B20240415-x86_64-unknown-linux-gnu-install_only.tar.gz";
const GET_PIP_URL: &str = "https://bootstrap.pypa.io/get-pip.py";
const FFMPEG_URL: &str =
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
const FFMPEG_LINUX_URL: &str =
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz";
const HF_MODEL: &str = "OliBomby/Mapperatorinator-v32";
const MODEL_BYTES: u64 = 1_800_000_000; // ~1.7 GB, for the poll-based bar

struct Job(Mutex<Option<Child>>);

fn default_engine() -> String {
    if cfg!(windows) {
        let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".into());
        format!("{base}\\osu2bs\\engine")
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dev = format!("{home}/app/osu2bs");
        if PathBuf::from(&dev).join(".venv").exists() {
            dev // developer checkout with its own venv
        } else {
            format!("{home}/.local/share/osu2bs/engine")
        }
    }
}

fn python_bin(engine: &str) -> PathBuf {
    let e = PathBuf::from(engine);
    let venv = e.join(".venv").join("bin").join("python");
    if venv.exists() {
        venv // developer checkout
    } else if cfg!(windows) {
        e.join("pyenv").join("python.exe")
    } else {
        e.join("pyenv").join("python").join("bin").join("python3")
    }
}

fn engine_cmd(engine: &str, device: &str, args: &[String]) -> Command {
    let mut c = Command::new(python_bin(engine));
    c.arg("-u").args(args).current_dir(engine);
    // contained runtime: ffmpeg + HF cache live inside the engine dir
    let bin = PathBuf::from(engine).join("bin");
    let old = std::env::var("PATH").unwrap_or_default();
    let sep = if cfg!(windows) { ";" } else { ":" };
    c.env("PATH", format!("{}{sep}{old}", bin.display()));
    c.env("HF_HOME", PathBuf::from(engine).join("hf"));
    c.env("OSU2BS_DEVICE", device);
    c.env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

/* ------------------------- setup ------------------------- */

#[derive(Serialize, Clone)]
struct SetupStatus {
    engine: bool,
    python: bool,
    torch: bool,
    deps: bool,
    ffmpeg: bool,
    model: bool,
    complete: bool,
    device: String,
    gpus: Vec<String>,
}

fn marker(engine: &str, name: &str) -> PathBuf {
    PathBuf::from(engine).join(".setup").join(name)
}

fn detect_gpus() -> Vec<String> {
    let mut c = Command::new("nvidia-smi");
    c.args(["--query-gpu=name", "--format=csv,noheader"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    match c.output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        _ => vec![],
    }
}

fn refresh_engine(engine: &str) {
    // an updated exe carries updated engine code; re-unpack when they differ
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ENGINE_ZIP.hash(&mut h);
    let v = format!("{:x}", h.finish());
    let m = marker(engine, "engine_v");
    if std::fs::read_to_string(&m).ok().as_deref() != Some(v.as_str())
        && unzip_to(ENGINE_ZIP, &PathBuf::from(engine)).is_ok()
    {
        let _ = std::fs::create_dir_all(m.parent().unwrap());
        let _ = std::fs::write(&m, v);
    }
}

#[tauri::command]
fn setup_status(engine: String) -> SetupStatus {
    // a developer checkout (repo + venv) counts as fully provisioned;
    // otherwise the setup markers rule, identically on every OS
    let dev = PathBuf::from(&engine).join(".venv").exists();
    let e = |n: &str| dev || marker(&engine, n).exists();
    let engine_ok = PathBuf::from(&engine).join("run.py").exists();
    if engine_ok && !dev {
        refresh_engine(&engine);
    }
    let python = e("python");
    let torch = e("torch");
    let deps = e("deps");
    let ffmpeg = e("ffmpeg");
    let model = e("model");
    let gpus = detect_gpus();
    let device = std::fs::read_to_string(marker(&engine, "device"))
        .unwrap_or_else(|_| if gpus.is_empty() { "cpu".into() } else { "cuda:0".into() });
    SetupStatus {
        complete: engine_ok && python && torch && deps && ffmpeg && model,
        engine: engine_ok,
        python,
        torch,
        deps,
        ffmpeg,
        model,
        device: device.trim().to_string(),
        gpus,
    }
}

fn emit(app: &tauri::AppHandle, step: &str, pct: f64, msg: &str) {
    let _ = app.emit("setup-progress", (step.to_string(), pct, msg.to_string()));
}

fn download(app: &tauri::AppHandle, step: &str, url: &str, to: &PathBuf)
    -> Result<(), String>
{
    let resp = reqwest::blocking::get(url).map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{url}: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let mut reader = resp;
    let mut out = std::fs::File::create(to).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 65536];
    let mut got: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        if total > 0 {
            emit(app, step, got as f64 / total as f64,
                 &format!("{:.0} / {:.0} MB", got as f64 / 1e6, total as f64 / 1e6));
        }
    }
    Ok(())
}

fn unzip_to(data: &[u8], dest: &PathBuf) -> Result<usize, String> {
    let mut ar = zip::ZipArchive::new(std::io::Cursor::new(data))
        .map_err(|e| e.to_string())?;
    for i in 0..ar.len() {
        let mut f = ar.by_index(i).map_err(|e| e.to_string())?;
        let Some(rel) = f.enclosed_name() else { continue };
        let path = dest.join(rel);
        if f.is_dir() {
            std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            let mut out = std::fs::File::create(&path).map_err(|e| e.to_string())?;
            std::io::copy(&mut f, &mut out).map_err(|e| e.to_string())?;
        }
    }
    Ok(ar.len())
}

fn pymod(app: &tauri::AppHandle, step: &str, engine: &str, module: &str,
         args: &[&str], pip_flags: bool) -> Result<(), String>
{
    let mut c = engine_cmd(engine, "cpu",
        &["-m".into(), module.into()]);
    c.args(args);
    if pip_flags {
        c.args(["--progress-bar", "off", "--no-input"]);
    }
    c.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| e.to_string())?;
    let out = child.stdout.take().unwrap();
    let app2 = app.clone();
    let step2 = step.to_string();
    let t = std::thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let l = line.trim();
            if l.starts_with("Collecting") || l.starts_with("Downloading")
                || l.starts_with("Installing") || l.starts_with("Successfully") {
                emit(&app2, &step2, -1.0, l);
            }
        }
    });
    let mut err = String::new();
    let _ = child.stderr.take().unwrap().read_to_string(&mut err);
    let status = child.wait().map_err(|e| e.to_string())?;
    let _ = t.join();
    if !status.success() {
        return Err(err.lines().rev().take(8).collect::<Vec<_>>().join("\n"));
    }
    Ok(())
}

fn pip(app: &tauri::AppHandle, step: &str, engine: &str, args: &[&str])
    -> Result<(), String>
{
    pymod(app, step, engine, "pip", args, true)
}

fn dir_size(p: &PathBuf) -> u64 {
    let mut s = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let path = e.path();
            s += if path.is_dir() { dir_size(&path) }
                 else { e.metadata().map(|m| m.len()).unwrap_or(0) };
        }
    }
    s
}

fn untar(archive: &PathBuf, dest: &PathBuf) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    let st = Command::new("tar")
        .arg("xf").arg(archive).arg("-C").arg(dest)
        .status().map_err(|e| e.to_string())?;
    if !st.success() { return Err("tar extraction failed".into()); }
    Ok(())
}

fn find_bin(root: &PathBuf, name: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir(root).ok()?;
    for e in rd.flatten() {
        let path = e.path();
        if path.is_dir() {
            if let Some(f) = find_bin(&path, name) { return Some(f); }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}

fn do_setup(app: tauri::AppHandle, engine: String, device: String)
    -> Result<(), String>
{
    let root = PathBuf::from(&engine);
    std::fs::create_dir_all(root.join(".setup")).map_err(|e| e.to_string())?;
    std::fs::write(marker(&engine, "device"), &device).ok();

    emit(&app, "engine", -1.0, "Unpacking engine…");
    unzip_to(ENGINE_ZIP, &root)?;
    emit(&app, "engine", 1.0, "Engine ready");

    if !marker(&engine, "python").exists() {
        if cfg!(windows) {
            let tmp = root.join("py-embed.zip");
            download(&app, "python", PY_EMBED_URL, &tmp)?;
            let data = std::fs::read(&tmp).map_err(|e| e.to_string())?;
            unzip_to(&data, &root.join("pyenv"))?;
            let _ = std::fs::remove_file(&tmp);
            // the embeddable build ships with `import site` disabled AND
            // locks sys.path to this file (no script dir, no PYTHONPATH):
            // the engine root + Mapperatorinator must be listed explicitly
            let pth = root.join("pyenv").join("python311._pth");
            if let Ok(t) = std::fs::read_to_string(&pth) {
                let t = t.replace("#import site", "import site");
                std::fs::write(&pth, format!("{t}\n..\n..\\Mapperatorinator\n"))
                    .map_err(|e| e.to_string())?;
            }
            let gp = root.join("get-pip.py");
            download(&app, "python", GET_PIP_URL, &gp)?;
            emit(&app, "python", -1.0, "Installing pip…");
            let st = engine_cmd(&engine, "cpu", &[gp.display().to_string()])
                .status().map_err(|e| e.to_string())?;
            if !st.success() {
                return Err("get-pip failed".into());
            }
            let _ = std::fs::remove_file(&gp);
        } else {
            // python-build-standalone: a normal CPython with pip included
            let tmp = root.join("py-standalone.tar.gz");
            download(&app, "python", PY_LINUX_URL, &tmp)?;
            emit(&app, "python", -1.0, "Unpacking Python…");
            untar(&tmp, &root.join("pyenv"))?;
            let _ = std::fs::remove_file(&tmp);
        }
        std::fs::write(marker(&engine, "python"), "").ok();
    }
    emit(&app, "python", 1.0, "Python ready");

    if !marker(&engine, "torch").exists() {
        emit(&app, "torch", -1.0, "Installing PyTorch (this is the big one)…");
        // light-the-torch inspects the local driver and resolves the
        // correct torch build (e.g. Blackwell needs cu128+; cu126 dies
        // with NoKernelImageForDevice). Hardcoded indexes are the
        // fallback if ltt can't cope.
        let ltt = pip(&app, "torch", &engine, &["install", "light-the-torch"])
            .and_then(|_| {
                let mut a = vec!["install"];
                if !device.starts_with("cuda") {
                    a.extend(["--pytorch-computation-backend", "cpu"]);
                }
                a.extend(["torch", "torchaudio"]);
                pymod(&app, "torch", &engine, "light_the_torch", &a, false)
            });
        if let Err(e) = ltt {
            emit(&app, "torch", -1.0,
                 &format!("Auto-detect failed ({e}); using default build"));
            let index = if device.starts_with("cuda") {
                "https://download.pytorch.org/whl/cu128"
            } else {
                "https://download.pytorch.org/whl/cpu"
            };
            pip(&app, "torch", &engine,
                &["install", "torch", "torchaudio", "--index-url", index])?;
        }
        std::fs::write(marker(&engine, "torch"), &device).ok();
    }
    emit(&app, "torch", 1.0, "PyTorch ready");

    if !marker(&engine, "deps").exists() {
        emit(&app, "deps", -1.0, "Installing audio + model libraries…");
        // the file is platform-neutral despite the name; pip resolves
        pip(&app, "deps", &engine, &["install", "-r", "requirements-win.txt"])?;
        std::fs::write(marker(&engine, "deps"), "").ok();
    }
    emit(&app, "deps", 1.0, "Libraries ready");

    if !marker(&engine, "ffmpeg").exists() {
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
        let ext = root.join("ffmpeg-tmp");
        if cfg!(windows) {
            let tmp = root.join("ffmpeg.zip");
            download(&app, "ffmpeg", FFMPEG_URL, &tmp)?;
            let data = std::fs::read(&tmp).map_err(|e| e.to_string())?;
            unzip_to(&data, &ext)?;
            let _ = std::fs::remove_file(&tmp);
        } else {
            let tmp = root.join("ffmpeg.tar.xz");
            download(&app, "ffmpeg", FFMPEG_LINUX_URL, &tmp)?;
            emit(&app, "ffmpeg", -1.0, "Unpacking ffmpeg…");
            untar(&tmp, &ext)?;
            let _ = std::fs::remove_file(&tmp);
        }
        let names: [&str; 2] = if cfg!(windows) {
            ["ffmpeg.exe", "ffprobe.exe"]
        } else {
            ["ffmpeg", "ffprobe"]
        };
        for exe in names {
            let src = find_bin(&ext, exe).ok_or(format!("{exe} not in archive"))?;
            std::fs::copy(&src, bin.join(exe)).map_err(|e| e.to_string())?;
        }
        let _ = std::fs::remove_dir_all(&ext);
        std::fs::write(marker(&engine, "ffmpeg"), "").ok();
    }
    emit(&app, "ffmpeg", 1.0, "ffmpeg ready");

    if !marker(&engine, "model").exists() {
        emit(&app, "model", 0.0, "Downloading the AI mapper (≈1.7 GB)…");
        let hf = root.join("hf");
        let app2 = app.clone();
        let hf2 = hf.clone();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();
        let poll = std::thread::spawn(move || {
            while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                let got = dir_size(&hf2);
                emit(&app2, "model",
                     (got as f64 / MODEL_BYTES as f64).min(0.99),
                     &format!("{:.0} MB / ~1700 MB", got as f64 / 1e6));
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        });
        let code = format!(
            "from huggingface_hub import snapshot_download; \
             snapshot_download('{HF_MODEL}')");
        let st = engine_cmd(&engine, "cpu", &["-c".into(), code])
            .status().map_err(|e| e.to_string());
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = poll.join();
        if !st.map_err(|e| e)?.success() {
            return Err("model download failed".into());
        }
        std::fs::write(marker(&engine, "model"), "").ok();
    }
    emit(&app, "model", 1.0, "AI mapper ready");

    let _ = app.emit("setup-done", true);
    Ok(())
}

#[tauri::command]
fn run_setup(app: tauri::AppHandle, engine: String, device: String) {
    std::thread::spawn(move || {
        if let Err(e) = do_setup(app.clone(), engine, device) {
            let _ = app.emit("setup-error", e);
        }
    });
}

#[tauri::command]
fn set_device(engine: String, device: String) {
    let _ = std::fs::create_dir_all(PathBuf::from(&engine).join(".setup"));
    let _ = std::fs::write(marker(&engine, "device"), device);
}

/* ------------------------- pipeline ------------------------- */

#[tauri::command]
fn run_pipeline(
    app: tauri::AppHandle,
    job: State<Job>,
    engine: String,
    device: String,
    args: Vec<String>,
) -> Result<(), String> {
    let mut guard = job.0.lock().unwrap();
    if guard.is_some() {
        return Err("a job is already running".into());
    }
    let mut all = vec!["run.py".to_string()];
    all.extend(args);
    let mut cmd = engine_cmd(&engine, &device, &all);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    *guard = Some(child);
    drop(guard);

    fn pump<R: std::io::Read + Send + 'static>(
        app: tauri::AppHandle, name: &'static str, r: R,
    ) {
        std::thread::spawn(move || {
            for line in BufReader::new(r).lines().map_while(Result::ok) {
                let _ = app.emit("pipeline-line", (name, line));
            }
        });
    }
    pump(app.clone(), "out", stdout);
    pump(app.clone(), "err", stderr);
    let handle = app.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let job = handle.state::<Job>();
        let mut guard = job.0.lock().unwrap();
        match guard.as_mut().map(|c| c.try_wait()) {
            Some(Ok(Some(status))) => {
                *guard = None;
                drop(guard);
                let _ = handle.emit("pipeline-done", status.code().unwrap_or(-1));
                break;
            }
            None => break, // cancelled
            _ => {}
        }
    });
    Ok(())
}

#[tauri::command]
fn cancel_pipeline(job: State<Job>) {
    if let Some(mut child) = job.0.lock().unwrap().take() {
        let _ = child.kill();
    }
}

/* ------------------------- misc ------------------------- */

#[tauri::command]
fn finalize_output(
    out_dir: String,
    slug: String,
    keep_dirs: bool,
    keep_zips: bool,
) -> Result<(), String> {
    let base = PathBuf::from(&out_dir);
    let map = base.join(&slug);
    if !keep_zips {
        let _ = std::fs::remove_file(base.join(format!("{slug}.zip")));
    }
    if !keep_dirs {
        std::fs::remove_dir_all(&map).map_err(|e| e.to_string())?;
    } else if let Ok(rd) = std::fs::read_dir(&map) {
        // consumer folder: keep only what Beat Saber reads
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".osu") || name == "song_orig.egg" {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn expand_dir(path: String) -> Vec<String> {
    let exts = ["mp3", "ogg", "wav", "m4a", "flac", "egg"];
    let mut out = vec![];
    if let Ok(rd) = std::fs::read_dir(&path) {
        for e in rd.flatten() {
            let p = e.path();
            let ok = p.extension().and_then(|x| x.to_str())
                .map(|x| exts.contains(&x.to_lowercase().as_str()))
                .unwrap_or(false);
            if ok {
                out.push(p.to_string_lossy().to_string());
            }
        }
    }
    out.sort();
    out
}

#[derive(Serialize)]
struct Defaults {
    engine: String,
    out_dir: String,
    windows: bool,
}

#[tauri::command]
fn defaults() -> Defaults {
    let out = [
        r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
        r"C:\Program Files\Steam\steamapps\common\Beat Saber",
        r"D:\SteamLibrary\steamapps\common\Beat Saber",
    ]
    .iter()
    .map(|p| PathBuf::from(p).join("Beat Saber_Data").join("CustomLevels"))
    .find(|p| p.exists())
    .map(|p| p.to_string_lossy().to_string())
    .unwrap_or_default();
    Defaults {
        engine: default_engine(),
        out_dir: out,
        windows: cfg!(windows),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Job(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            run_pipeline,
            cancel_pipeline,
            finalize_output,
            expand_dir,
            setup_status,
            run_setup,
            set_device,
            defaults
        ])
        .run(tauri::generate_context!())
        .expect("error while running osu2bs");
}
