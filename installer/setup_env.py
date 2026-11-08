"""Deep-Boost-Cam setup, run by installer/Install.bat inside .venv.

Safe to re-run: every step checks what is already there.
Flags: --yes (accept defaults, no questions), --models=recommended|all|none,
       --no-tensorrt, --no-shortcut.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
ARGS = sys.argv[1:]
AUTO = "--yes" in ARGS
PY = sys.executable

REPO = "Demosand/Deep-Boost-Cam"
# Prebuilt insightface (no C++ compiler needed), attached to the "deps" release.
INSIGHTFACE_WHEEL = "insightface-0.7.3-cp311-cp311-win_amd64.whl"
INSIGHTFACE_URL = f"https://github.com/{REPO}/releases/download/deps/{INSIGHTFACE_WHEEL}"
TORCH = ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# onnxruntime-gpu 1.26 links nvinfer_10.dll -> TensorRT 10.x for CUDA 12 only.
TENSORRT = "tensorrt-cu12-libs==10.13.3.9"
MIN_DRIVER = 570.0          # recommended for CUDA 12.8 and RTX 50 (Blackwell)


def say(text: str = "") -> None:
    print(text, flush=True)


def step(text: str) -> None:
    say()
    say("== " + text)


def ask(question: str, default: bool = True) -> bool:
    if AUTO:
        return default
    hint = "[Y/n]" if default else "[y/N]"
    ans = input(f"   {question} {hint}: ").strip().lower()
    return default if not ans else ans in ("y", "yes", "д", "да")


def choose(question: str, options, default: str) -> str:
    for a in ARGS:
        if a.startswith("--models="):
            return a.split("=", 1)[1]
    if AUTO:
        return default
    say("   " + question)
    for key, text in options:
        say(f"     {key}) {text}")
    ans = input(f"   Выбор [{default}]: ").strip()
    keys = [k for k, _ in options]
    return ans if ans in keys else default


def pip(*args: str) -> None:
    cmd = [PY, "-m", "pip", "install", "--disable-pip-version-check", *args]
    if subprocess.call(cmd) != 0:
        raise SystemExit(f"[X] pip не смог установить: {' '.join(args)}")


def installed(dist: str) -> bool:
    from importlib import metadata
    try:
        metadata.version(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


# ── 1. GPU ─────────────────────────────────────────────────────────────
def gpu_info():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    if not out:
        raise SystemExit("[X] nvidia-smi не вернул данные о видеокарте.")
    name, mem, drv, cc = [p.strip() for p in out.splitlines()[0].split(",")]
    return name, int(float(mem)), drv, float(cc)


def check_gpu():
    step("Видеокарта")
    name, mem, drv, cc = gpu_info()
    say(f"   {name}, {mem / 1024:.0f} ГБ, драйвер {drv}, compute {cc}")
    try:
        if float(drv.split(".")[0] + "." + drv.split(".")[1]) < MIN_DRIVER:
            say(f"   [!] Драйвер старый: рекомендуется {MIN_DRIVER:.0f}+ (CUDA 12.8). Обновите драйвер NVIDIA.")
            if not ask("Продолжить всё равно?", False):
                raise SystemExit(1)
    except (ValueError, IndexError):
        pass
    if cc < 7.5:
        say("   [!] Видеокарта старше Turing (GTX 16/RTX 20): TensorRT недоступен, работа будет медленной.")
    return name, mem, cc


# ── 2. packages ────────────────────────────────────────────────────────
def install_packages(cc: float) -> bool:
    step("Python-пакеты")
    pip("--upgrade", "pip", "wheel")
    pip("-r", os.path.join("installer", "requirements-lock.txt"))
    if not installed("torch"):
        say("   PyTorch с CUDA 12.8 (~3 ГБ) ...")
        pip(*TORCH, "--index-url", TORCH_INDEX)
    if not installed("insightface"):
        wheel = os.path.join("_cache", INSIGHTFACE_WHEEL)
        try:
            say("   insightface (готовая сборка) ...")
            from modules.model_downloader import fetch_url
            if fetch_url(INSIGHTFACE_URL, os.path.abspath(wheel)) is None:
                raise RuntimeError("download failed")
            pip(wheel)
        except Exception as error:
            say(f"   [!] Готовая сборка недоступна ({error}); собираю из исходников "
                "(нужны Visual Studio Build Tools) ...")
            pip("insightface==0.7.3")
    want_trt = cc >= 7.5 and "--no-tensorrt" not in ARGS
    if want_trt and not installed("tensorrt-cu12-libs"):
        say("   TensorRT 10.13 (~3 ГБ): ускоряет модели в 2-3 раза.")
        if ask("Установить TensorRT?", True):
            pip(TENSORRT)
    return installed("tensorrt-cu12-libs")


# ── 3. models ──────────────────────────────────────────────────────────
FAILED = []


def _fetch(url: str, path: str) -> None:
    """Retries + Hugging Face mirror; a failed model never stops the install."""
    from modules.model_downloader import fetch_url
    if not os.path.isfile(path):
        say(f"   {os.path.relpath(path, ROOT)} ...")
    if fetch_url(url, path) is None:
        FAILED.append(os.path.basename(path))


def download_models() -> None:
    step("Модели")
    kind = choose("Какие модели скачать сейчас? (остальные скачаются сами при первом выборе)", [
        ("1", "Рекомендуемые, ~1.8 ГБ: анализ лица, Inswapper, HyperSwap 1B, GPEN-256, "
              "маски, апскейлер SPAN"),
        ("2", "Все, ~4.5 ГБ: плюс HyperSwap 1A/1C, ReSwapper, GFPGAN, GPEN-512, "
              "все апскейлеры, интерполяция RIFE"),
        ("3", "Не скачивать сейчас"),
    ], "1")
    kind = {"recommended": "1", "all": "2", "none": "3"}.get(kind, kind)
    if kind == "3":
        return
    from modules import model_downloader as md
    from modules.paths import MODELS_DIR
    from modules.processors.frame import face_regions, swapper_models, upscaler
    from modules import frame_interp

    md.ensure_insightface_pack("buffalo_l")
    hf = ["inswapper_128_fp16.onnx", "GPEN-BFR-256.onnx"]
    swaps = ["hyperswap_1b_256"]
    sr = ["span"]
    if kind == "2":
        hf += ["gfpgan-1024.onnx", "GPEN-BFR-512.onnx"]
        swaps += ["hyperswap_1a_256", "hyperswap_1c_256", "reswapper_256"]
        sr = list(upscaler.MODELS)
    for name in hf:
        if md.ensure_model(name) is None and md.ensure_model(name) is None:
            FAILED.append(name)
    for key in swaps:
        if swapper_models.ensure_model_file(swapper_models.BY_KEY[key], MODELS_DIR) is None:
            FAILED.append(swapper_models.BY_KEY[key].model_file)
    for file_name, url in face_regions._MODELS.values():
        _fetch(url, os.path.join(MODELS_DIR, file_name))
    for key in sr:
        spec = upscaler.MODELS[key]
        _fetch(spec.url, os.path.join(MODELS_DIR, "frame", spec.file))
    if kind == "2":
        _fetch(frame_interp.MODEL_URL, os.path.join(MODELS_DIR, "frame", frame_interp.MODEL_FILE))
    os.makedirs(os.path.join(MODELS_DIR, "dfm"), exist_ok=True)
    if FAILED:
        say(f"   [!] Не скачались (плохая связь): {', '.join(FAILED)}.")
        say("       Это не страшно: запустите установщик ещё раз позже - он докачает только их,")
        say("       или программа скачает модель сама при первом выборе.")


# ── 4. check + config ──────────────────────────────────────────────────
def register_dlls() -> None:
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    for d in (os.path.join(site, "torch", "lib"), os.path.join(site, "tensorrt_libs")):
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            os.add_dll_directory(d)


def verify(has_trt: bool) -> bool:
    step("Проверка")
    register_dlls()
    import onnxruntime as ort
    providers = ort.get_available_providers()
    say(f"   onnxruntime {ort.__version__}: {', '.join(providers)}")
    det = os.path.join(ROOT, "_home", ".insightface", "models", "buffalo_l", "det_10g.onnx")
    if os.path.isfile(det):
        sess = ort.InferenceSession(det, providers=["CUDAExecutionProvider"])
        ok = sess.get_providers()[0] == "CUDAExecutionProvider"
        say("   CUDA: " + ("работает" if ok else "НЕ работает - модели пойдут на процессоре"))
        if not ok:
            return False
    if has_trt:
        from modules.processors.frame._onnx_enhancer import tensorrt_available
        say("   TensorRT: " + ("готов" if tensorrt_available() else "не найден"))
    import torch
    say(f"   PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()} (нужен для DFM-студии)")
    return True


def write_config(mem_mb: int, has_trt: bool) -> None:
    """First-run settings for this GPU. Never overwrites the user's settings."""
    path = os.path.join(ROOT, "switch_states.json")
    state = {}
    if os.path.isfile(path):
        try:
            state = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
    changed = False
    if "trt_max_workspace_gb" not in state:
        # 8 GB cards: leave room for the camera pipeline while engines build.
        state["trt_max_workspace_gb"] = 2 if mem_mb < 12000 else 4
        changed = True
    if has_trt and "use_tensorrt" not in state:
        step("TensorRT")
        state["use_tensorrt"] = ask("Включить TensorRT в программе? (быстрее; первый запуск "
                                    "каждой модели собирает движок 1-4 мин)", True)
        changed = True
    if changed:
        json.dump(state, open(path, "w", encoding="utf-8"))


def shortcut() -> None:
    if "--no-shortcut" in ARGS or not ask("Создать ярлык на рабочем столе?", True):
        return
    target = os.path.join(ROOT, "Start-DeepBoostCam.bat")
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'Deep-Boost-Cam.lnk'));"
        f"$s.TargetPath='{target}';$s.WorkingDirectory='{ROOT}';"
        "$s.IconLocation='%SystemRoot%\\System32\\imageres.dll,18';$s.Save()"
    )
    # The desktop path comes from the real profile, not the redirected one.
    env = dict(os.environ, USERPROFILE=os.environ.get("REAL_USERPROFILE", os.environ["USERPROFILE"]))
    subprocess.call(["powershell", "-NoProfile", "-Command", ps], env=env)


def main() -> None:
    _name, mem, cc = check_gpu()
    has_trt = install_packages(cc)
    download_models()
    ok = verify(has_trt)
    write_config(mem, has_trt)
    shortcut()
    step("Готово" if ok else "Готово с предупреждениями")
    if mem < 12000:
        say("   8 ГБ видеопамяти: для Live подходят HyperSwap/Inswapper + GPEN-256; "
            "GFPGAN и ReSwapper тяжёлые. В DFM-студии ставьте batch 4-6.")
    if subprocess.run("where ffmpeg", shell=True, capture_output=True).returncode != 0:
        say("   ffmpeg не найден: он нужен только для обработки видеофайлов (Live работает без него).")
        say("   Установка: winget install Gyan.FFmpeg")


if __name__ == "__main__":
    main()
