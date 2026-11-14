#!/usr/bin/env bash
# FVP (Fluid Video Player): puts it in ~/.local/share/fvp with a Python of its own that has VapourSynth and
# mvtools (and subliminal, for subtitles from the web; from PyPI: the same on Arch, Bazzite, CachyOS, Fedora, Ubuntu; nothing needs root), fetches the
# upscaling shaders, and adds the `fvp` command and a launcher entry ("Open with" for films).
# ./install.sh --remove takes it away again.
set -euo pipefail
cd "$(dirname "$0")"
DEST="$HOME/.local/share/fvp"
BIN="$HOME/.local/bin/fvp"
DESKTOP="$HOME/.local/share/applications/local.hanis.fvp.desktop"
if [ "${1:-}" = "--remove" ]; then
    rm -rf "$DEST" "$BIN" "$DESKTOP"
    echo "FVP removed (its settings stay in ~/.config/fvp)"
    exit 0
fi
for tool in python3 mpv ffmpeg ffprobe curl; do
    command -v "$tool" >/dev/null || { echo "FVP needs $tool: install it first" >&2; exit 1; }
done
mkdir -p "$DEST/shaders" "$(dirname "$BIN")" "$(dirname "$DESKTOP")"
cp fvp.py fvp_ui.lua CHANGELOG.md README.md "$DEST/"
[ -d shaders ] && cp -n shaders/*.glsl "$DEST/shaders/" 2>/dev/null || true
if [ ! -x "$DEST/.venv/bin/python" ]; then
    python3 -m venv "$DEST/.venv"
fi
"$DEST/.venv/bin/pip" install -q --upgrade vapoursynth vapoursynth-mvtools numpy subliminal
# the shaders (text files): AMD FSR 1, FSRCNNX, ArtCNN
fetch() { [ -s "$DEST/shaders/$1" ] || curl -sfL -m 60 -o "$DEST/shaders/$1" "$2" || echo "could not fetch $1 (upscaling step without it falls back to mpv's scaler)" >&2; }
fetch FSR.glsl "https://gist.githubusercontent.com/agyild/82219c545228d70c5604f865ce0b0ce5/raw/FSR.glsl"
fetch FSRCNNX_x2_8-0-4-1.glsl "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_8-0-4-1.glsl"
fetch ArtCNN_C4F16.glsl "https://raw.githubusercontent.com/Artoriuz/ArtCNN/main/GLSL/ArtCNN_C4F16.glsl"
fetch ArtCNN_C4F32.glsl "https://raw.githubusercontent.com/Artoriuz/ArtCNN/main/GLSL/ArtCNN_C4F32.glsl"
printf '#!/bin/sh\nexec python3 "%s/fvp.py" "$@"\n' "$DEST" > "$BIN"
chmod +x "$BIN"
cat > "$DESKTOP" <<D
[Desktop Entry]
Type=Application
Name=FVP
GenericName=Fluid Video Player
Comment=Films with frames made in between and upscaled as they play
Exec=$BIN %f
Icon=video-x-generic
Categories=AudioVideo;Video;Player;
MimeType=video/x-matroska;video/mp4;video/x-msvideo;video/webm;video/quicktime;video/mpeg;video/mp2t;
D
command -v update-desktop-database >/dev/null && update-desktop-database "$(dirname "$DESKTOP")" 2>/dev/null || true
echo "FVP $("$BIN" --version | cut -d' ' -f2) installed: fvp FILM   (fvp probe: what this computer will use)"
