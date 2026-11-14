# FVP versions

Numbering as in Handheld Control and Rádio Beat: 0.MINOR.PATCH. Every change (commit) is a version: PATCH one
more; a bigger novelty raises MINOR (PATCH from 0 again).

## 0.2 — Menu, seek bar, subtitles

- **0.2.0** (2026-11-14) A menu on the right mouse button with everything the player does: pause, seeking, frames in between and their preset, the upscaling step, the sound track, volume and mute, subtitles, full screen, status, quit. A bar along the bottom shows position / length (h:mm:ss) and the volume, and seeks on a click or a drag; it shows while the mouse moves and during a seek. Volume on the mouse wheel and the + and - keys. Subtitles: the film's own tracks, files beside the film, or looked up on the web (podnapisi.net, BSPlayer, OpenSubtitles and others, by the film's name and size, in Czech / the system's language and English). Sound tracks can be picked. Fixed: the sound was gone after seeking with the keys (a seek that came while the one before it was still opening read "no sound track" from mpv and kept it; the picked tracks are now kept by the player). One-line download and install: get.sh

## 0.1 — First prototype

- **0.1.0** (2026-11-14) The first prototype, made and measured on the Ryzen 7 7730U mini PC: a film is decoded by ffmpeg (on the graphics card), frames are made in between by mvtools on the processor (24 → 60 a second, five presets, a quicker one picked by itself when frames run out), and mpv upscales and shows them (steps from ordinary scalers through AMD FSR 1 to small neural nets, stepping down by itself when drawing is too slow). Seeking, sound and subtitles from the film's file, 8- and 10-bit films. `fvp probe` tells what a computer has (AMD, Intel, Nvidia; several cards) and what is used for what; `fvp bench` measures the presets. A neural net for the frames in between (RIFE) is wired in but not tested
