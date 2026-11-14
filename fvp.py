#!/usr/bin/env python3
"""FVP (Fluid Video Player): films played with frames made in between (24 -> 60 a second) and upscaled, as they play.

How it's put together (one file, three roles):

  the player   (this program, plain Python)  starts mpv, and for the film a chain of three programs:
                 ffmpeg (decodes, on the graphics card where it can)
                   -> the worker (makes the frames in between)
                   -> ffmpeg (stamps the film's own times on them) -> mpv (upscales with shaders, shows).
               mpv takes the sound and the subtitles from the film's file itself. A seek starts the chain
               anew at the new place (mpv can't seek in a pipe), so the player handles the seek keys.
  the worker   (`fvp.py worker ...`, run by the project's own Python, which has VapourSynth): raw
               frames in, more raw frames out. What makes them is a backend:
                 mvtools   motion vectors on the processor (what Smooth Video Project does); runs anywhere
                 rife      a neural net on a graphics card through Vulkan (any maker's); needs the
                           VapourSynth RIFE plugin, NOT TESTED yet (no computer here can run it at speed)
                 blend     neighbouring frames mixed (no plugin needed; only a stand-in)
  the probe    (`fvp.py probe`) what the computer has (processors, graphics cards of any maker,
               several at once) and what the player will use each for.

mpv isn't asked to do the frames itself: the mpv most systems ship (Arch's, Flathub's) is built without
VapourSynth, so the work is done outside it and piped in.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

APP = "fvp"
APP_VERSION = "0.2.1"  # 0.MINOR.PATCH: every change a version (PATCH + 1), a bigger novelty MINOR + 1
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), APP)
CONFIG = os.path.join(CONFIG_DIR, "config.json")
RUNTIME = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", APP)
SHADERS = os.path.join(HERE, "shaders")
UI_SCRIPT = os.path.join(HERE, "fvp_ui.lua")  # the seek bar, drawn inside mpv
SUBS_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), APP, "subs")
SUB_EXTS = (".srt", ".ass", ".ssa", ".sub", ".vtt")

# ---------------------------------------------------------------------------------------------------------
# the worker: frames in between
# ---------------------------------------------------------------------------------------------------------

# mvtools, from quickest to best. Measured on the Ryzen 7 7730U (8 cores), a 1920x800 film to 59.94 a second:
# fastest 3.2x real time, fast 2.7x, medium 1.15x, good 0.8x. Moving blocks (BlockFPS) is far cheaper than
# moving every pixel along the vectors (FlowFPS, about 70 frames/s there whatever the blocks' size), which
# leaves no blocky edges on what moves.
MV_PRESETS = {
    "fastest": dict(pel=1, blk=32, overlap=0, search=4, sp=1, flow=False, recalc=False),
    "fast": dict(pel=1, blk=16, overlap=0, search=4, sp=2, flow=False, recalc=False),
    "medium": dict(pel=1, blk=16, overlap=0, search=4, sp=2, flow=True, recalc=False),
    "good": dict(pel=1, blk=16, overlap=8, search=4, sp=2, flow=True, recalc=False),
    "best": dict(pel=2, blk=16, overlap=8, search=5, sp=2, flow=True, recalc=True),
}
PRESET_ORDER = ["fastest", "fast", "medium", "good", "best"]


class FrameReader:
    """Raw frames off a pipe, in order, the last few kept (the filters ask for neighbours, from several
    threads, not always in order)."""

    KEEP = 48

    def __init__(self, stream, w: int, h: int, bits: int):
        import numpy as np
        self.np, self.stream = np, stream
        self.dtype = np.uint8 if bits == 8 else np.uint16
        b = 1 if bits == 8 else 2
        self.sizes = [(h, w), (h // 2, w // 2), (h // 2, w // 2)]
        self.bytes = [r * c * b for r, c in self.sizes]
        self.frames, self.next, self.end = {}, 0, None  # end: how many frames there were, once known
        self.lock = threading.Lock()

    def _read(self):
        planes = []
        for size, (rows, cols) in zip(self.bytes, self.sizes):
            buf = self.stream.read(size)
            if len(buf) < size:
                return None
            planes.append(self.np.frombuffer(buf, self.dtype).reshape(rows, cols))
        return planes

    def get(self, n: int):
        with self.lock:
            while self.end is None and self.next <= n:
                planes = self._read()
                if planes is None:
                    self.end = self.next
                    break
                self.frames[self.next] = planes
                self.next += 1
                self.frames.pop(self.next - 1 - self.KEEP, None)
            if self.end is not None and n >= self.end:
                n = self.end - 1
            got = self.frames.get(n)
            if got is None:  # asked for one long gone (shouldn't be): the oldest kept stands in
                got = self.frames[min(self.frames)] if self.frames else None
            return got


def worker(args: list[str]) -> int:
    """fvp.py worker W H BITS FPSNUM FPSDEN OUTNUM OUTDEN BACKEND PRESET [GPU]"""
    w, h, bits, fn, fd, on, od = (int(a) for a in args[:7])
    backend, preset = args[7], args[8]
    gpu = int(args[9]) if len(args) > 9 else 0
    import numpy as np
    import vapoursynth as vs
    core = vs.core
    threads = os.cpu_count() or 4
    core.num_threads = threads
    fmt = vs.YUV420P8 if bits == 8 else vs.YUV420P10
    reader = FrameReader(sys.stdin.buffer, w, h, bits)
    if reader.get(0) is None:
        return 0
    blank = core.std.BlankClip(format=fmt, width=w, height=h, length=10 ** 7, fpsnum=fn, fpsden=fd, keep=True)

    def fetch(n, f):
        out = f.copy()
        planes = reader.get(n)
        for p in range(3):
            np.asarray(out[p])[:] = planes[p]
        return out
    src = core.std.ModifyFrame(blank, blank, fetch)

    if backend == "mvtools":
        o = MV_PRESETS[preset]
        sup = core.mv.Super(src, pel=o["pel"], hpad=o["blk"], vpad=o["blk"], rfilter=4)
        ana = dict(blksize=o["blk"], overlap=o["overlap"], search=o["search"], searchparam=o["sp"], chroma=True,
                   truemotion=True)
        bv, fv = core.mv.Analyse(sup, isb=True, **ana), core.mv.Analyse(sup, isb=False, **ana)
        if o["recalc"]:
            half = dict(blksize=o["blk"] // 2, overlap=o["overlap"] // 2, search=o["search"], searchparam=1, thsad=140)
            bv, fv = core.mv.Recalculate(sup, bv, **half), core.mv.Recalculate(sup, fv, **half)
        # (a cut: the frame before it is shown again, never two scenes mixed)
        if o["flow"]:
            out = core.mv.FlowFPS(src, sup, bv, fv, num=on, den=od, mask=2, blend=False)
        else:
            out = core.mv.BlockFPS(src, sup, bv, fv, num=on, den=od, mode=3, blend=False)
    elif backend == "rife":
        # NOT TESTED (see the top). The plugin works on RGB floats and doubles the rate; other rates through
        # its own fps arguments.
        rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")
        mid = core.rife.RIFE(rgb, model=int(preset) if preset.isdigit() else 0, fps_num=on, fps_den=od, gpu_id=gpu,
                             gpu_thread=2, sc=True)
        out = core.resize.Bicubic(mid, format=fmt, matrix_s="709", dither_type="ordered")
    else:  # blend
        out = core.std.Merge(src, src.std.Trim(first=1), 0.5)
        out = core.std.Interleave([src, out]).std.AssumeFPS(fpsnum=fn * 2, fpsden=fd)
        out = core.resize.Point(out).std.AssumeFPS(fpsnum=fn * 2, fpsden=fd)
        on, od = fn * 2, fd

    dst = sys.stdout.buffer
    tag = "420" if bits == 8 else "420p10"
    try:
        dst.write(f"YUV4MPEG2 W{w} H{h} F{on}:{od} Ip A1:1 C{tag}\n".encode())
        t0, last, shown = time.monotonic(), time.monotonic(), 0
        for j, frame in enumerate(out.frames(prefetch=threads, backlog=threads * 2, close=True)):
            if reader.end is not None and j * od * fn // (on * fd) >= reader.end:
                break  # (past the film's last frame)
            dst.write(b"FRAME\n")
            for p in range(3):
                dst.write(np.ascontiguousarray(np.asarray(frame[p])).data)
            now = time.monotonic()
            if now - last >= 2.0:  # (how fast it goes: the player reads it)
                print(f"fps {(j + 1 - shown) / (now - last):.1f} avg {(j + 1) / (now - t0):.1f}", file=sys.stderr,
                      flush=True)
                last, shown = now, j + 1
        dst.flush()
    except BrokenPipeError:
        pass
    return 0


# ---------------------------------------------------------------------------------------------------------
# subtitles from the web
# ---------------------------------------------------------------------------------------------------------

SUB_PROVIDERS = ["podnapisi", "bsplayer", "opensubtitles", "gestdown", "tvsubtitles"]  # (those needing no account)


def sub_langs(cfg: dict) -> list[str]:
    """The languages subtitles are looked for in, best first: the settings' "sub_langs", else the system's
    language, Czech or Slovak where the clock is set to there (a Czech's system is often in English), English."""
    if cfg.get("sub_langs"):
        return list(cfg["sub_langs"])
    out = []
    lang = (os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or "")[:2].lower()
    if lang.isalpha() and lang != "c":
        out.append(lang)
    zone = os.path.realpath("/etc/localtime")
    for place, code in (("Europe/Prague", "cs"), ("Europe/Bratislava", "sk")):
        if zone.endswith(place) and code not in out:
            out.insert(0, code)
    if "en" not in out:
        out.append("en")
    return out


def subs_dir(film: str) -> str:
    return os.path.join(SUBS_CACHE, re.sub(r"[^\w.-]+", "_", os.path.splitext(os.path.basename(film))[0])[:120])


def fetch_subs(args: list[str]) -> int:
    """fvp.py subs FILM LANG[,LANG...]: looks the film up at the subtitle sites (by its name and size) and
    saves the best subtitles of each language into the cache; prints 'saved LANG PATH' for each."""
    film, langs = args[0], args[1].split(",")
    from babelfish import Language
    from subliminal import download_best_subtitles, region, save_subtitles, scan_video
    region.configure("dogpile.cache.memory")
    out = subs_dir(film)
    os.makedirs(out, exist_ok=True)
    video = scan_video(film)
    want = set()
    for code in langs:
        try:
            want.add(Language.fromietf(code))
        except Exception:
            pass
    found = download_best_subtitles([video], want, providers=SUB_PROVIDERS).get(video, [])
    for saved in save_subtitles(video, found, directory=out):
        path = os.path.join(out, os.path.basename(saved.get_path(video)))
        print(f"saved {saved.language} {path}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------
# the probe: what this computer has
# ---------------------------------------------------------------------------------------------------------

VENDORS = {"0x1002": "AMD", "0x10de": "Nvidia", "0x8086": "Intel"}
# How strong a graphics card is for this work, by its name (a guess until the benchmark has run on it):
# 0 weak integrated (light shaders; no neural frames), 1 newer integrated (medium shaders; neural frames at
# 720p), 2 a mid card (good shaders; neural frames at 1080p), 3 a strong card (the best of both at 4K).
GPU_TIERS = (
    (r"RTX (40[89]0|50[789]0|30[89]0)|RX 7900|RX 9070|RX 6[89]00", 3),
    (r"RTX|Radeon VII|TITAN|8060S|8050S|RX [5-9]\d{3}|\bArc\b.* [AB]\d{3}|GTX 1[06][678]0", 2),
    (r"780M|880M|890M|8\d0S|\bArc\b|\bXe2?\b|140V|140T|680M|760M|860M", 1),
    (r"", 0),
)


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _vulkan_names() -> dict:
    """{vendor id: [device names]} from vulkaninfo, when it's there."""
    out = {}
    try:
        text = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=8).stdout
    except (OSError, subprocess.TimeoutExpired):
        return out
    for block in text.split("GPU")[1:]:
        vid = re.search(r"vendorID\s*=\s*(0x[0-9a-fA-F]+)", block)
        name = re.search(r"deviceName\s*=\s*(.+)", block)
        kind = re.search(r"deviceType\s*=\s*(\S+)", block)
        if vid and name and (not kind or "CPU" not in kind.group(1)):
            out.setdefault(vid.group(1).lower(), []).append(name.group(1).strip())
    return out


def _pci_name(dev: str) -> str:
    """The card's name from lspci (when vulkaninfo isn't installed): what's in the last square brackets."""
    try:
        text = subprocess.run(["lspci", "-s", os.path.basename(os.path.realpath(dev))], capture_output=True,
                              text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    text = re.sub(r"\(rev [0-9a-f]+\)\s*$", "", text.strip())
    text = text.split(": ", 1)[-1]
    text = re.sub(r"^.*?\[(AMD/ATI|AMD)\]\s*", "AMD ", text)  # "Advanced Micro Devices, Inc. [AMD/ATI] Barcelo"
    text = re.sub(r"^(NVIDIA|Intel) Corporation\s*", r"\1 ", text)
    names = re.findall(r"\[([^\]]+)\]", text)  # "... [GeForce RTX 5060]"
    return names[-1] if names else text.strip()


def probe() -> dict:
    cpu = {"name": "", "threads": os.cpu_count() or 1, "flags": set()}
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name") and not cpu["name"]:
            cpu["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("flags") and not cpu["flags"]:
            cpu["flags"] = set(line.split(":", 1)[1].split()) & {"avx2", "avx512f", "avx512bw"}
    names = _vulkan_names()
    gpus = []
    drm = "/sys/class/drm"
    for card in sorted(n for n in (os.listdir(drm) if os.path.isdir(drm) else []) if re.fullmatch(r"card\d+", n)):
        dev = f"{drm}/{card}/device"
        vid = _read(f"{dev}/vendor").lower()
        if vid not in VENDORS:
            continue
        pool = names.get(vid, [])
        name = pool.pop(0) if pool else _pci_name(dev) or f"{VENDORS[vid]} {_read(f'{dev}/device')}"
        try:
            driver = os.path.basename(os.readlink(f"{dev}/driver"))
        except OSError:
            driver = ""
        screens = [c for c in os.listdir(f"{drm}/{card}") if c.startswith(card + "-")
                   and _read(f"{drm}/{card}/{c}/status") == "connected"] if os.path.isdir(f"{drm}/{card}") else []
        render = next((os.path.basename(p) for p in sorted(os.listdir(f"{dev}/drm")) if p.startswith("renderD")), "") \
            if os.path.isdir(f"{dev}/drm") else ""
        tier = next(t for rx, t in GPU_TIERS if re.search(rx, name, re.I))
        gpus.append({"card": card, "vendor": VENDORS[vid], "name": name, "driver": driver, "render": render,
                     "screen": bool(screens), "tier": tier, "index": len(gpus)})
    return {"cpu": cpu, "gpus": gpus, "plan": plan(cpu, gpus)}


def worker_python() -> str | None:
    """The Python that has VapourSynth: the project's own (made by install.sh), else the system's."""
    for py in (os.path.join(HERE, ".venv", "bin", "python"), sys.executable, shutil.which("python3") or ""):
        if py and os.path.exists(py):
            r = subprocess.run([py, "-c", "import vapoursynth, numpy; vapoursynth.core.mv"], capture_output=True)
            if r.returncode == 0:
                return py
    return None


def rife_ready(py: str | None) -> bool:
    if not py:
        return False
    return subprocess.run([py, "-c", "import vapoursynth; vapoursynth.core.rife"], capture_output=True).returncode == 0


def plan(cpu: dict, gpus: list) -> dict:
    """What does what. The screen's card shows and upscales; the frames in between go to the strongest card
    that isn't weak (another one than the screen's when there are two alike: they don't get in each other's
    way), else to the processor; decoding to the screen's card (no copying between cards)."""
    show = next((g for g in gpus if g["screen"]), gpus[0] if gpus else None)
    best = max(gpus, key=lambda g: (g["tier"], g is not show), default=None)
    py = worker_python()
    neural = bool(best and best["tier"] >= 1 and rife_ready(py))
    hw = {"Nvidia": "cuda", "AMD": "vaapi", "Intel": "vaapi"}.get(show["vendor"] if show else "", "none")
    tier = show["tier"] if show else 0
    return {
        "decode": hw,
        "decode_on": show["name"] if show else "the processor",
        "interp": "rife" if neural else "mvtools" if py else "off",
        "interp_on": best["name"] if neural else cpu["name"],
        "interp_gpu": best["index"] if neural else 0,
        "rife_possible": bool(best and best["tier"] >= 1),
        "upscale": UPSCALE_BY_TIER[min(3, tier)],
        "upscale_on": show["name"] if show else "",
        "python": py,
    }


def probe_text(p: dict) -> str:
    c, pl = p["cpu"], p["plan"]
    lines = [f"Processor      {c['name']} ({c['threads']} threads{', ' + ' '.join(sorted(c['flags'])) if c['flags'] else ''})"]
    for g in p["gpus"]:
        lines.append(f"Graphics card  {g['name']} ({g['vendor']}, driver {g['driver'] or '?'}, strength {g['tier']} of 3"
                     f"{', has the screen' if g['screen'] else ''})")
    lines += ["", f"Decoding       {pl['decode']} on {pl['decode_on']}" if pl["decode"] != "none" else "Decoding       on the processor",
              f"Frames between {pl['interp']} on {pl['interp_on']}"
              + ("  (a neural net could run on a card here once the RIFE plugin is installed)"
                 if pl["interp"] == "mvtools" and pl["rife_possible"] else "")
              + ("  (VapourSynth is missing: run install.sh)" if pl["interp"] == "off" else ""),
              f"Upscaling      {pl['upscale']} shaders on {pl['upscale_on']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# the film and the chain
# ---------------------------------------------------------------------------------------------------------

def film_info(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_space,color_transfer,"
                        "color_primaries,color_range,field_order:format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    if not s.get("width"):
        raise SystemExit(f"{APP}: no video in {path}")

    def rate(text):
        a, _, b = (text or "0/1").partition("/")
        return int(a or 0), int(b or 1)
    fn, fd = rate(s.get("avg_frame_rate"))
    if not fn or fn / fd > 240:
        fn, fd = rate(s.get("r_frame_rate"))
    # the usual rates, exactly
    fps = fn / fd if fd else 0
    for num, den in ((24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (50, 1), (60000, 1001), (60, 1)):
        if abs(fps - num / den) < 0.005:
            fn, fd = num, den
    bits = 10 if re.search(r"p1[0246]", s.get("pix_fmt", "")) else 8
    return {"w": int(s["width"]), "h": int(s["height"]), "fn": fn, "fd": fd, "bits": bits,
            "duration": float(d.get("format", {}).get("duration") or 0),
            "matrix": s.get("color_space", ""), "trc": s.get("color_transfer", ""),
            "prim": s.get("color_primaries", ""), "range": s.get("color_range", "")}


def out_rate(info: dict, screen_hz: float) -> tuple[int, int] | None:
    """The rate to make: the screen's (60 for 24, 25 and 30 a second; 59.94 for 23.976 and 29.97, which
    mpv then fits to the screen); None when the film has that many frames already."""
    fps = info["fn"] / info["fd"]
    hz = round(screen_hz) if screen_hz else 60
    hz = 60 if hz < 50 else hz
    if fps >= hz * 0.8:
        return None
    hz = min(hz, 120)
    return (hz * 1000, 1001) if info["fd"] == 1001 else (hz, 1)


# what the film's colours are called by ffmpeg -> by mpv (told to mpv: raw frames carry no such notes)
MPV_MATRIX = {"bt709": "bt.709", "bt2020nc": "bt.2020-ncl", "bt2020c": "bt.2020-cl", "smpte170m": "bt.601",
              "bt470bg": "bt.601"}
MPV_TRC = {"bt709": "bt.1886", "smpte2084": "pq", "arib-std-b67": "hlg"}
MPV_PRIM = {"bt709": "bt.709", "bt2020": "bt.2020", "smpte170m": "bt.601-525", "bt470bg": "bt.601-625"}


def colour_filter(info: dict) -> str:
    parts = []
    for key, table, name in (("matrix", MPV_MATRIX, "colormatrix"), ("trc", MPV_TRC, "gamma"),
                             ("prim", MPV_PRIM, "primaries")):
        if info[key] in table:
            parts.append(f"{name}={table[info[key]]}")
    if info["range"] in ("tv", "pc"):
        parts.append("colorlevels=" + ("limited" if info["range"] == "tv" else "full"))
    return "format=" + ":".join(parts) if parts else ""


class Chain:
    """ffmpeg -> worker -> ffmpeg -> a pipe mpv reads, from one place in the film on."""

    count = 0

    def __init__(self, path: str, info: dict, out: tuple, start: float, pl: dict, preset: str, on_fps=None):
        Chain.count += 1
        os.makedirs(RUNTIME, exist_ok=True)
        self.fifo = os.path.join(RUNTIME, f"{os.getpid()}-{Chain.count}.nut")
        os.mkfifo(self.fifo)
        pix = "yuv420p" if info["bits"] == 8 else "yuv420p10le"
        dec = ["ffmpeg", "-v", "error", "-nostdin"]
        if pl["decode"] != "none":
            dec += ["-hwaccel", pl["decode"]]
        dec += ["-ss", f"{start:.3f}", "-i", path, "-map", "0:v:0", "-an", "-sn",
                "-vf", f"fps={info['fn']}/{info['fd']},format={pix}", "-f", "rawvideo", "-"]
        work = [pl["python"], os.path.abspath(__file__), "worker", str(info["w"]), str(info["h"]), str(info["bits"]),
                str(info["fn"]), str(info["fd"]), str(out[0]), str(out[1]), pl["interp"],
                preset, str(pl.get("interp_gpu", 0))]
        stamp = ["ffmpeg", "-v", "error", "-nostdin", "-f", "yuv4mpegpipe", "-i", "pipe:0", "-c", "copy",
                 "-output_ts_offset", f"{start:.3f}", "-f", "nut", "-y", self.fifo]
        self.p1 = subprocess.Popen(dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.p2 = subprocess.Popen(work, stdin=self.p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.p1.stdout.close()
        self.p3 = subprocess.Popen(stamp, stdin=self.p2.stdout, stderr=subprocess.DEVNULL)
        self.p2.stdout.close()
        self.fps, self.errors = 0.0, []
        threading.Thread(target=self._watch, args=(on_fps,), daemon=True).start()

    def _watch(self, on_fps) -> None:
        for raw in self.p2.stderr:
            line = raw.decode(errors="replace").strip()
            m = re.match(r"fps ([\d.]+)", line)
            if m:
                self.fps = float(m.group(1))
                if on_fps:
                    on_fps(self.fps)
            elif line:
                self.errors.append(line)

    def stop(self) -> None:
        for p in (self.p3, self.p2, self.p1):
            try:
                p.kill()
            except OSError:
                pass
        for p in (self.p3, self.p2, self.p1):
            try:
                p.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                pass
        try:
            os.unlink(self.fifo)
        except OSError:
            pass


def bench(path: str, seconds: float = 6.0, start: float = None, verbose: bool = True) -> dict:
    """How fast each preset makes frames from this film on this computer: {preset: times real time}."""
    info, pr = film_info(path), probe()
    pl = pr["plan"]
    out = out_rate(info, 60) or (60, 1)
    if pl["interp"] == "off":
        raise SystemExit(f"{APP}: VapourSynth is missing: run install.sh")
    start = info["duration"] * 0.4 if start is None else start
    need = out[0] / out[1]
    results = {}
    presets = PRESET_ORDER if pl["interp"] == "mvtools" else ["0"]
    for preset in presets:
        got = []
        chain = Chain(path, info, out, start, pl, preset, got.append)
        sink = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", chain.fifo, "-f", "null", "-"],
                                stderr=subprocess.DEVNULL)
        time.sleep(seconds + 2.5)
        chain.stop()
        sink.kill()
        sink.wait()
        speed = (sum(got[1:]) / len(got[1:]) if len(got) > 1 else got[0] if got else 0.0)
        results[preset] = speed / need
        if verbose:
            print(f"  {preset:8s} {speed:6.1f} frames/s  = {speed / need:4.2f}x real time"
                  + (f"   ({chain.errors[-1]})" if not got and chain.errors else ""), flush=True)
        if results[preset] < 0.9:
            break  # (the next ones are slower still)
    return results


def pick_preset(results: dict, room: float = 1.2) -> str | None:
    """The best preset with room to spare (films have harder scenes than the tested one)."""
    ok = [p for p in results if results[p] >= room]
    return ok[-1] if ok else None


def load_config() -> dict:
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG + ".part", "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(CONFIG + ".part", CONFIG)


def size_class(info: dict) -> str:
    w = info["w"]  # (by width: a wide film is 1920x800)
    return "4k" if w > 2600 else "1080p" if w > 1400 else "720p" if w > 1000 else "sd"


# ---------------------------------------------------------------------------------------------------------
# upscaling: mpv's shaders
# ---------------------------------------------------------------------------------------------------------

# The steps, from cheapest to best: (mpv's own scaler, debanding, shaders from shaders/). A missing shader is
# left out and mpv's scaler does the work. Every frame shown is a new one once frames are made in between,
# so a step must draw a frame well inside 1/60 s.
# Measured on the Ryzen 7 7730U's graphics (Vega 8), 1920x800 to a 4K screen, a frame's drawing: catmull_rom
# 5.6 ms, lanczos 6.7 ms, mpv's ewa_lanczossharp 21 ms, FSRCNNX 47 ms, ArtCNN C4F16 100 ms: the neural nets
# are for stronger cards.
#   plain  a clean ordinary scaler
#   sharp  a sharper one (lanczos)
#   fsr    AMD FSR 1 (edge-directed upscaling and sharpening; runs on any maker's card)
#   light / good / best   small neural nets doubling the picture's sharp part: FSRCNNX, ArtCNN C4F16, C4F32
UPSCALE = {
    "plain": ("catmull_rom", False, []),
    "sharp": ("lanczos", False, []),
    "fsr": ("lanczos", False, ["FSR.glsl"]),
    "light": ("ewa_lanczossharp", True, ["FSRCNNX_x2_8-0-4-1.glsl"]),
    "good": ("ewa_lanczossharp", True, ["ArtCNN_C4F16.glsl"]),
    "best": ("ewa_lanczossharp", True, ["ArtCNN_C4F32.glsl"]),
}
UPSCALE_ORDER = ["plain", "sharp", "fsr", "light", "good", "best"]
UPSCALE_BY_TIER = ("fsr", "light", "good", "best")  # where a card of each strength starts (it steps down by
# itself when frames come late)


def shader_list(step: str) -> list[str]:
    return [p for p in (os.path.join(SHADERS, n) for n in UPSCALE.get(step, ("", False, []))[2]) if os.path.exists(p)]


# ---------------------------------------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------------------------------------

class Mpv:
    """mpv, talked to over its socket."""

    def __init__(self, extra: list[str]):
        os.makedirs(RUNTIME, exist_ok=True)
        self.sock_path = os.path.join(RUNTIME, f"{os.getpid()}.sock")
        args = ["mpv", "--idle=yes", "--force-window=yes", f"--input-ipc-server={self.sock_path}",
                "--no-config", "--vo=gpu-next", "--hwdec=no", "--video-sync=display-resample",
                "--demuxer-max-bytes=600MiB", "--demuxer-readahead-secs=1.5", "--cache=yes", "--cache-pause=yes",
                "--cache-pause-wait=0.3", "--rebase-start-time=no", "--cscale=bilinear", "--dscale=mitchell",
                "--osd-font-size=30", "--keep-open=no", "--sub-auto=no", "--audio-file-auto=no",
                "--title=FVP", "--input-default-bindings=yes", "--osc=no", "--osd-bar=no", "--osd-on-seek=no",
                "--volume-max=150"] + ([f"--script={UI_SCRIPT}"] if os.path.exists(UI_SCRIPT) else []) + extra
        self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.sock = None
        for _ in range(100):
            try:
                s = socket.socket(socket.AF_UNIX)
                s.connect(self.sock_path)
                self.sock = s
                break
            except OSError:
                time.sleep(0.05)
        if self.sock is None:
            raise SystemExit(f"{APP}: mpv did not start")
        self.lock, self.waiting, self.rid, self.handlers = threading.Lock(), {}, 0, []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        while True:
            try:
                data = self.sock.recv(65536)
            except OSError:
                data = b""
            if not data:
                for h in self.handlers:
                    h({"event": "gone"})
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "request_id" in msg and msg["request_id"] in self.waiting:
                    ev, box = self.waiting.pop(msg["request_id"])
                    box.append(msg)
                    ev.set()
                elif "event" in msg:
                    for h in self.handlers:
                        h(msg)

    def cmd(self, *command, wait: float = 2.0):
        with self.lock:
            self.rid += 1
            rid = self.rid
            ev, box = threading.Event(), []
            self.waiting[rid] = (ev, box)
            try:
                self.sock.sendall(json.dumps({"command": list(command), "request_id": rid}).encode() + b"\n")
            except OSError:
                return None
        ev.wait(wait)
        return box[0].get("data") if box and box[0].get("error") == "success" else None

    def get(self, prop: str):
        return self.cmd("get_property", prop)

    def set(self, prop: str, value) -> None:
        self.cmd("set_property", prop, value)

    def osd(self, text: str, secs: float = 2.5) -> None:
        self.cmd("show-text", text, int(secs * 1000))


def clock(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


class Player:
    KEYS = {  # key -> what the player is told
        "RIGHT": "seek 5", "LEFT": "seek -5", "UP": "seek 60", "DOWN": "seek -60", "PGUP": "seek 600",
        "PGDWN": "seek -600", "Shift+RIGHT": "seek 1", "Shift+LEFT": "seek -1", "HOME": "seekto 0",
        "i": "interp", "u": "upscale", "p": "preset 1", "P": "preset -1", "o": "status", "TAB": "status",
    }
    MPV_KEYS = {  # key -> what mpv does by itself
        "WHEEL_UP": "add volume 5", "WHEEL_DOWN": "add volume -5", "+": "add volume 5", "=": "add volume 5",
        "-": "add volume -5", "KP_ADD": "add volume 5", "KP_SUBTRACT": "add volume -5",
        # the menu: mpv's own menu script draws it on Linux ("context-menu", the command, opens only the
        # system's native menu, which exists on Windows: the right button did nothing with it)
        "MBTN_RIGHT": "script-message-to context_menu open", "MENU": "script-message-to context_menu open",
    }

    def __init__(self, path: str, opts: dict):
        self.path, self.opts = os.path.abspath(path), opts
        self.info = film_info(self.path)
        self.probe = probe()
        self.plan = self.probe["plan"]
        self.cfg = load_config()
        self.interp = self.plan["interp"] != "off" and not opts.get("no_interp")
        self.upscale = opts.get("upscale") or self.cfg.get("upscale") or self.plan["upscale"]
        self.preset = opts.get("preset") or self.cfg.get("presets", {}).get(size_class(self.info)) or "fast"
        self.chain, self.base, self.pending = None, 0.0, None
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.underruns = []
        # the sound and subtitle tracks picked (mpv's ids; None: the first sound track): kept by the player and
        # set again on every new stretch. (Read back from mpv at each seek before: a seek while the stretch
        # before it was still opening read "no sound track" and kept it, so the sound was gone after seeking.)
        self.tracks = {"aid": None, "sid": "no"}
        self.extra_subs, self.ready = [], False  # subtitle files added (the web's, a file's); the stretch is open
        self.mpv = Mpv(opts.get("mpv", []))
        self.mpv.handlers.append(self._event)
        for key, what in self.KEYS.items():
            self.mpv.cmd("keybind", key, f"script-message sv {what}")
        for key, what in self.MPV_KEYS.items():
            self.mpv.cmd("keybind", key, what)
        for n, prop in enumerate(("paused-for-cache", "aid", "sid", "pause", "mute", "fullscreen"), 1):
            self.mpv.cmd("observe_property", n, prop)
        hz = self.mpv.get("display-fps") or 0
        for _ in range(20):  # (known once the window is up)
            if hz:
                break
            time.sleep(0.1)
            hz = self.mpv.get("display-fps") or 0
        self.out = out_rate(self.info, hz) if self.interp else None
        if self.upscale not in UPSCALE:
            self.upscale = self.plan["upscale"]
        self._upscale_set()
        if os.environ.get("FVP_UPDATED_FROM"):
            self.mpv.osd(f"FVP updated: {os.environ['FVP_UPDATED_FROM']} → {APP_VERSION}", 5)
        self.start(float(opts.get("start") or 0))
        threading.Thread(target=self._watch_drawing, daemon=True).start()

    def _upscale_set(self) -> None:
        scale, deband, _files = UPSCALE[self.upscale]
        self.mpv.set("scale", scale)
        self.mpv.set("deband", deband)
        self.mpv.set("glsl-shaders", shader_list(self.upscale))
        self._drawn_since = time.monotonic()

    def draw_ms(self) -> float:
        """How long mpv takes to draw a frame now (its passes' averages together), ms."""
        passes = self.mpv.get("vo-passes") or {}
        return sum(p.get("avg", 0) for p in passes.get("fresh", [])) / 1e6

    def _watch_drawing(self) -> None:
        """The upscaling steps down by itself while a frame's drawing takes more than 60 % of its time on the
        screen (the rest is for the desktop and whatever else draws): frames would come late."""
        while not self.done.wait(4.0):
            if self.mpv.get("pause") or time.monotonic() - getattr(self, "_drawn_since", 0) < 6:
                continue
            fps = (self.out[0] / self.out[1]) if (self.chain and self.out) else self.info["fn"] / self.info["fd"]
            ms, room = self.draw_ms(), 600.0 / max(1.0, fps)
            i = UPSCALE_ORDER.index(self.upscale)
            if ms > room and i > 0:
                self.upscale = UPSCALE_ORDER[max(0, i - (2 if ms > 3 * room else 1))]  # (far off: two steps)
                self._upscale_set()
                self.cfg["upscale"] = self.upscale
                save_config(self.cfg)
                self.mpv.osd(f"Upscaling took {ms:.0f} ms a frame (room for {room:.0f}): now '{self.upscale}'", 4)

    # --- one stretch of the film -------------------------------------------------------------------------
    def start(self, pos: float) -> None:
        with self.lock:
            if self.chain:
                self.chain.stop()
                self.chain = None
            pos = max(0.0, min(pos, max(0.0, self.info["duration"] - 1)))
            self.base, self.ready = pos, False
            if self.interp and self.out:
                self.chain = Chain(self.path, self.info, self.out, pos, self.plan, self.preset)
                colour = colour_filter(self.info)
                self.mpv.set("vf", colour)
                self.mpv.set("hwdec", "no")
                self.mpv.cmd("loadfile", self.chain.fifo, "replace")
            else:  # as it is: mpv plays the file itself (and seeks in it by itself)
                self.mpv.set("vf", "")
                self.mpv.set("hwdec", "auto-safe")
                self.mpv.cmd("loadfile", self.path, "replace", -1, f"start={pos:.3f}")

    def _tracks(self, kind: str) -> list:
        return [t for t in (self.mpv.get("track-list") or []) if t.get("type") == kind]

    def _loaded(self) -> None:
        """The stretch is open: the film's own sound and subtitles beside it (mpv seeks them to its time), the
        subtitle files added, and the tracks picked before."""
        with self.lock:
            if self.chain:
                self.mpv.cmd("audio-add", self.path, "auto")
                if not self._tracks("sub"):
                    self.mpv.cmd("sub-add", self.path, "auto")
            for path in self.extra_subs:
                self.mpv.cmd("sub-add", path, "auto")
            audio = self._tracks("audio")
            aid = self.tracks["aid"]
            if audio and aid not in [t["id"] for t in audio]:
                aid = audio[0]["id"]
            self.mpv.set("aid", aid if audio else "no")
            sid = self.tracks["sid"]
            self.mpv.set("sid", sid if sid in [t["id"] for t in self._tracks("sub")] else "no")
            self.mpv.set("force-media-title", os.path.basename(self.path))
            self.mpv.cmd("script-message", "fvp-ui", "duration", f"{self.info['duration']:.2f}")
            self.mpv.cmd("script-message", "fvp-ui", "arrived")
            self.ready = True
        self.menu()

    def pos(self) -> float:
        t = self.mpv.get("time-pos")
        return float(t) if isinstance(t, (int, float)) else self.base

    # --- what mpv says -----------------------------------------------------------------------------------
    def _event(self, msg: dict) -> None:
        ev = msg.get("event")
        if ev == "gone":
            self.done.set()
        elif ev == "file-loaded":
            threading.Thread(target=self._loaded, daemon=True).start()
        elif ev == "end-file":
            if msg.get("reason") == "eof" and self.pending is None:
                with self.lock:
                    far = self.chain is None or self.pos() >= self.info["duration"] - 3 or \
                        (self.chain.p2.poll() is not None and self.chain.p2.returncode == 0)
                if far:
                    self.mpv.cmd("quit")
        elif ev == "property-change" and msg.get("name") == "paused-for-cache" and msg.get("data") is True:
            self._underrun()
        elif ev == "property-change" and msg.get("name") in ("aid", "sid") and self.ready:
            # (picked with mpv's own keys: remembered like the menu's picks)
            v = msg.get("data")
            self.tracks[msg["name"]] = v if isinstance(v, int) and not isinstance(v, bool) else \
                ("no" if msg["name"] == "sid" else self.tracks["aid"])
            threading.Thread(target=self.menu, daemon=True).start()
        elif ev == "property-change" and msg.get("name") in ("pause", "mute", "fullscreen"):
            threading.Thread(target=self.menu, daemon=True).start()
        elif ev == "client-message" and (msg.get("args") or [None])[0] == "sv":
            threading.Thread(target=self._key, args=(msg["args"][1:],), daemon=True).start()

    def _underrun(self) -> None:
        """mpv ran out of frames: the frames aren't made fast enough. Twice within 20 s (not right after a
        seek): a quicker preset."""
        now = time.monotonic()
        if not self.chain or now - getattr(self, "_started", 0) < 4:
            return
        self.underruns = [t for t in self.underruns if now - t < 20] + [now]
        if len(self.underruns) >= 2 and self.plan["interp"] == "mvtools":
            self.underruns = []
            i = PRESET_ORDER.index(self.preset) if self.preset in PRESET_ORDER else 1
            if i > 0:
                self.preset = PRESET_ORDER[i - 1]
                self.mpv.osd(f"Too slow: frames in between now '{self.preset}'", 4)
                self._restart()
            else:
                self.interp = False
                self.mpv.osd("Too slow even at 'fastest': frames in between off", 4)
                self._restart()

    def _restart(self) -> None:
        self._seek_to(self.pos())

    def _seek_to(self, pos: float) -> None:
        """Seeks pile up while a key is held: the last one wins, a moment after it."""
        with self.lock:
            self.pending = pos
            self.mpv.cmd("script-message", "fvp-ui", "seeking", f"{pos:.2f}")  # (the bar shows where it goes)
            mine = self._seek_id = getattr(self, "_seek_id", 0) + 1

        def go():
            time.sleep(0.25)
            with self.lock:
                if self._seek_id != mine:
                    return
                self._started = time.monotonic()
                self.start(pos)
                self.pending = None
        threading.Thread(target=go, daemon=True).start()

    def _key(self, args: list) -> None:
        what = args[0] if args else ""
        if what in ("seek", "seekto"):
            if self.chain is None and self.pending is None:  # mpv plays the file itself
                self.mpv.cmd("seek", float(args[1]), "relative" if what == "seek" else "absolute")
                return
            at = self.pending if self.pending is not None else self.pos()
            self._seek_to(at + float(args[1]) if what == "seek" else float(args[1]))
        elif what == "interp":
            if self.plan["interp"] == "off":
                self.mpv.osd("Frames in between: not available (VapourSynth is missing: run install.sh)", 4)
                return
            self.interp = not self.interp
            if self.interp and not self.out:
                self.out = out_rate(self.info, self.mpv.get("display-fps") or 60)
            if self.interp and not self.out:
                self.mpv.osd("Frames in between: not needed, the film has as many as the screen shows", 4)
                self.interp = False
                return
            self.mpv.osd("Frames in between: " + (self._interp_text() if self.interp else "off"), 3)
            self._restart()
            self.menu()
        elif what in ("preset", "preset-set"):
            if self.plan["interp"] != "mvtools":
                return
            i = PRESET_ORDER.index(self.preset) + int(args[1]) if what == "preset" else PRESET_ORDER.index(args[1])
            if 0 <= i < len(PRESET_ORDER):
                self.preset = PRESET_ORDER[i]
                self.cfg.setdefault("presets", {})[size_class(self.info)] = self.preset
                save_config(self.cfg)
                self.mpv.osd(f"Frames in between: '{self.preset}'", 3)
                if self.interp:
                    self._restart()
                self.menu()
        elif what in ("upscale", "upscale-set"):
            i = (UPSCALE_ORDER.index(self.upscale) + 1) % len(UPSCALE_ORDER) if what == "upscale" else \
                UPSCALE_ORDER.index(args[1])
            self.upscale = UPSCALE_ORDER[i]
            files = shader_list(self.upscale)
            self._upscale_set()
            self.cfg["upscale"] = self.upscale
            save_config(self.cfg)
            missing = UPSCALE[self.upscale][2] and not files
            self.mpv.osd(f"Upscaling: {self.upscale}" + (" (its shader isn't installed: run install.sh)" if missing else ""), 3)
            self.menu()
        elif what == "status":
            self.mpv.osd(self.status(), 6)
            self.mpv.cmd("script-message", "fvp-ui", "show", "6")
        elif what in ("aid", "sid"):
            v = int(args[1]) if args[1].isdigit() else "no"
            self.tracks[what] = v
            if self.ready:
                self.mpv.set(what, v)
            self.menu()
        elif what == "subfile":
            files = self.sub_files()
            if args[1].isdigit() and int(args[1]) < len(files):
                self._add_sub(files[int(args[1])])
        elif what == "websubs":
            self._web_subs()

    # --- subtitles ---------------------------------------------------------------------------------------
    def sub_files(self) -> list[str]:
        """Subtitle files to offer: those beside the film and those fetched from the web for it."""
        out = []
        for d in (os.path.dirname(self.path), subs_dir(self.path)):
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            out += [os.path.join(d, n) for n in names if n.lower().endswith(SUB_EXTS)]
        return out

    def _add_sub(self, path: str) -> None:
        with self.lock:
            if path not in self.extra_subs:
                self.extra_subs.append(path)
                if self.ready:
                    self.mpv.cmd("sub-add", path, "auto")
            sid = next((t["id"] for t in self._tracks("sub") if t.get("external-filename") == path), None)
            if sid is not None:
                self.tracks["sid"] = sid
                self.mpv.set("sid", sid)
        self.mpv.osd(f"Subtitles: {os.path.basename(path)}", 3)
        self.menu()

    def _web_subs(self) -> None:
        if getattr(self, "_searching", False):
            return
        py = self.plan["python"]
        if not py or subprocess.run([py, "-c", "import subliminal"], capture_output=True).returncode != 0:
            self.mpv.osd("Subtitles from the web: not installed (run install.sh)", 4)
            return
        langs = sub_langs(self.cfg)
        self._searching = True
        self.mpv.osd(f"Looking for subtitles on the web ({', '.join(langs)})…", 60)
        try:
            r = subprocess.run([py, os.path.abspath(__file__), "subs", self.path, ",".join(langs)],
                               capture_output=True, text=True, timeout=90)
            saved = [line.split(" ", 2) for line in r.stdout.splitlines() if line.startswith("saved ")]
        except (OSError, subprocess.TimeoutExpired):
            saved = []
        finally:
            self._searching = False
        if not saved:
            self.mpv.osd(f"No subtitles found on the web ({', '.join(langs)})", 5)
            return
        by_lang = {lang: path for _s, lang, path in reversed(saved)}
        best = next((by_lang[k] for want in langs for k in by_lang if k.split("-")[0] == want), saved[0][2])
        for _s, _lang, path in saved:
            if path != best and path not in self.extra_subs:
                self.extra_subs.append(path)
                if self.ready:
                    self.mpv.cmd("sub-add", path, "auto")
        self._add_sub(best)

    # --- the menu (right click) --------------------------------------------------------------------------
    def menu(self) -> None:
        """mpv's own context menu, filled with everything the player can do, as it is now."""
        def item(title, cmd, key="", on=None, off=False):
            # (what's on is marked with a dot in its name: the menu's own check mark is a sign the usual
            # fonts don't have, drawn as an empty box)
            return {"title": ("●  " if on else "") + title, "cmd": cmd, "shortcut": key,
                    "state": ["disabled"] if off else []}

        def sv(what):
            return f"script-message sv {what}"
        sep = {"type": "separator"}

        def label(t):
            own = not t.get("external") or t.get("external-filename") == self.path  # (the film's own track)
            title = t.get("title") or ""
            if title in (os.path.basename(self.path), os.path.basename(t.get("external-filename") or "")):
                title = ""  # (mpv's stand-in for a track with no title: the file's name)
            bits = [t.get("lang") or ("" if not own else "unknown language"), title]
            if not own and "/" + APP + "/subs/" in (t.get("external-filename") or ""):
                bits.insert(0, "from the web:")
            if t["type"] == "audio":
                bits.append(" ".join(str(x) for x in (t.get("codec"), f"{t.get('demux-channel-count')} ch"
                                                      if t.get("demux-channel-count") else "") if x))
            elif t.get("external") and t.get("external-filename") != self.path:
                bits.append(os.path.basename(t.get("external-filename") or "file"))
            if t.get("forced"):
                bits.append("forced")
            return "  ".join(b for b in bits if b) or f"track {t['id']}"
        audio, subs = self._tracks("audio"), self._tracks("sub")
        aid, sid = self.mpv.get("aid"), self.mpv.get("sid")
        no_frames = self.plan["interp"] == "off"
        files = self.sub_files()
        known = {t.get("external-filename") for t in subs}
        data = [
            item("Pause / play", "cycle pause", "Space", on=bool(self.mpv.get("pause"))),
            {"title": "Seek", "type": "submenu", "submenu": [
                item("5 s forward", sv("seek 5"), "→"), item("5 s back", sv("seek -5"), "←"),
                item("1 min forward", sv("seek 60"), "↑"), item("1 min back", sv("seek -60"), "↓"),
                item("10 min forward", sv("seek 600"), "PgUp"), item("10 min back", sv("seek -600"), "PgDn"),
                item("To the start", sv("seekto 0"), "Home")]},
            sep,
            {"title": "Frames in between", "type": "submenu", "submenu": [
                item("On" if not no_frames else "On (VapourSynth is missing: run install.sh)", sv("interp"), "i",
                     on=self.interp and bool(self.out), off=no_frames), sep]
                + [item(f"{p}" + ("  (quickest)" if p == PRESET_ORDER[0] else "  (slowest)"
                                  if p == PRESET_ORDER[-1] else ""), sv(f"preset-set {p}"),
                        on=p == self.preset, off=self.plan["interp"] != "mvtools") for p in PRESET_ORDER]},
            {"title": "Upscaling", "type": "submenu", "submenu": [
                item({"plain": "plain  (ordinary scaler)", "sharp": "sharp  (lanczos)", "fsr": "fsr  (AMD FSR 1)",
                      "light": "light  (neural net FSRCNNX)", "good": "good  (neural net ArtCNN C4F16)",
                      "best": "best  (neural net ArtCNN C4F32)"}[u], sv(f"upscale-set {u}"), on=u == self.upscale)
                for u in UPSCALE_ORDER]},
            sep,
            {"title": "Sound", "type": "submenu", "submenu": [
                item(label(t), sv(f"aid {t['id']}"), on=t["id"] == aid) for t in audio]
                + [sep, item("Louder", "add volume 5", "+ / wheel"), item("Quieter", "add volume -5", "- / wheel"),
                   item("Mute", "cycle mute", "m", on=bool(self.mpv.get("mute")))]},
            {"title": "Subtitles", "type": "submenu", "submenu": [
                item("Off", sv("sid no"), on=sid in (False, None, "no"))]
                + [item(label(t), sv(f"sid {t['id']}"), on=t["id"] == sid) for t in subs]
                + [sep, item(f"Look on the web ({', '.join(sub_langs(self.cfg))})", sv("websubs"))]
                + [item("File: " + os.path.basename(p), sv(f"subfile {i}")) for i, p in enumerate(files)
                   if p not in known]},
            sep,
            item("Full screen", "cycle fullscreen", "f", on=bool(self.mpv.get("fullscreen"))),
            item("Status", sv("status"), "o"),
            sep,
            item("Quit FVP", "quit", "q"),
        ]
        self.mpv.set("menu-data", data)

    def _interp_text(self) -> str:
        return (f"{self.plan['interp']} '{self.preset}' on {self.plan['interp_on']}, "
                f"{self.info['fn'] / self.info['fd']:.3f} -> {self.out[0] / self.out[1]:.3f} frames/s")

    def status(self) -> str:
        made = f", making {self.chain.fps:.0f}/s" if self.chain and self.chain.fps else ""
        drops = self.mpv.get("frame-drop-count")
        late = self.mpv.get("vo-delayed-frame-count")
        return "\n".join([
            f"{clock(self.pos())} / {clock(self.info['duration'])}   {self.info['w']}x{self.info['h']} "
            f"{self.info['bits']}-bit",
            "Frames in between: " + (self._interp_text() + made if self.interp and self.out else "off") + "   [i, p/P]",
            f"Upscaling: {self.upscale} on {self.plan['upscale_on']}, {self.draw_ms():.1f} ms a frame   [u]",
            f"Decoding: {self.plan['decode']}   dropped {drops}, late {late}",
        ])

    def run(self) -> int:
        try:
            while not self.done.wait(0.5):
                if self.mpv.proc.poll() is not None:
                    break
        except KeyboardInterrupt:
            pass
        with self.lock:
            if self.chain:
                self.chain.stop()
        try:
            self.mpv.proc.terminate()
        except OSError:
            pass
        try:
            os.unlink(self.mpv.sock_path)
        except OSError:
            pass
        return 0


# ---------------------------------------------------------------------------------------------------------
# updating itself
# ---------------------------------------------------------------------------------------------------------

REPO = "kupecjan1980-pixel/fvp"
INSTALLED = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), APP)


def _ver(text: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:3])


def auto_update(argv: list[str]) -> None:
    """At the start of playing: when GitHub has a newer release, it's downloaded, installed (install.sh) and
    started in this one's place. Only the installed copy does it (not one run from its source folder), not
    with --no-update or "auto_update": false in the settings; no network or a slow one: skipped (2.5 s)."""
    import tarfile
    import tempfile
    import urllib.request
    if os.environ.get("FVP_UPDATED_FROM") or "--no-update" in argv or load_config().get("auto_update") is False \
            or os.path.realpath(HERE) != os.path.realpath(INSTALLED):
        return
    agent = {"User-Agent": "fvp-updater"}
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest", headers=agent)
        with urllib.request.urlopen(req, timeout=2.5) as r:
            tag = json.load(r)["tag_name"]
    except Exception:
        return
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag) or _ver(tag) <= _ver(APP_VERSION):
        return
    print(f"FVP {APP_VERSION}: updating to {tag[1:]}…", flush=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            req = urllib.request.Request(f"https://github.com/{REPO}/archive/refs/tags/{tag}.tar.gz", headers=agent)
            with urllib.request.urlopen(req, timeout=20) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
                tar.extractall(tmp, filter="data")
            script = os.path.join(tmp, f"fvp-{tag[1:]}", "install.sh")
            done = subprocess.run(["bash", script], capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            print(f"FVP: the update did not install ({done.stderr.strip()[-200:]}); playing with {APP_VERSION}")
            return
    except Exception as e:
        print(f"FVP: the update did not download ({e}); playing with {APP_VERSION}")
        return
    os.environ["FVP_UPDATED_FROM"] = APP_VERSION
    os.execv(sys.executable, [sys.executable, os.path.join(INSTALLED, "fvp.py")] + argv)


USAGE = f"""usage: {APP} FILM [--start SECONDS] [--preset {'|'.join(PRESET_ORDER)}] [--upscale {'|'.join(UPSCALE_ORDER)}]
                    [--no-interp] [--no-update] [-- mpv options]
       {APP} probe            what this computer has and what will be used for what
       {APP} bench FILM       how fast each preset runs here; the best one that keeps up is remembered

keys: arrows / PgUp / PgDn seek, + / - or the mouse wheel volume, i frames in between on/off,
      p / P better / quicker preset, u upscaling step, o or Tab status, space pause, f full screen, q quit
mouse: right click the menu (sound and subtitle tracks, subtitles from the web, everything else);
       move it for the bar along the bottom (position / length), click or drag the bar to seek"""


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "worker":
        return worker(argv[1:])
    if len(argv) >= 3 and argv[0] == "subs":
        return fetch_subs(argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"FVP {APP_VERSION}")
        return 0
    if argv[0] == "probe":
        print(probe_text(probe()))
        return 0
    if argv[0] == "bench":
        if len(argv) < 2:
            print(USAGE)
            return 2
        info = film_info(argv[1])
        print(f"{os.path.basename(argv[1])}: {info['w']}x{info['h']} {info['bits']}-bit, {info['fn'] / info['fd']:.3f} frames/s")
        results = bench(argv[1])
        best = pick_preset(results)
        cfg = load_config()
        if best:
            cfg.setdefault("presets", {})[size_class(info)] = best
            save_config(cfg)
            print(f"-> '{best}' for {size_class(info)} films (remembered)")
        else:
            print("-> none keeps up with room to spare: frames in between would stutter at this size here")
        return 0
    opts, files, i = {"mpv": []}, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            opts["mpv"] = argv[i + 1:]
            break
        if a in ("--start", "--preset", "--upscale") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
            i += 2
            continue
        if a == "--no-interp":
            opts["no_interp"] = True
        elif a == "--no-update":
            pass
        else:
            files.append(a)
        i += 1
    if not files or not os.path.exists(files[0]):
        print(USAGE)
        return 2
    for tool in ("mpv", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{APP}: {tool} is missing")
    auto_update(argv)  # (a newer release: installed and started in this one's place)
    return Player(files[0], opts).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
