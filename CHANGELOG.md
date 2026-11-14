# FVP versions

Numbering as in Handheld Control and Rádio Beat: 0.MINOR.PATCH. Every change (commit) is a version: PATCH one
more; a bigger novelty raises MINOR (PATCH from 0 again).

## 0.1 — First prototype

- **0.1.0** (2026-11-14) The first prototype, made and measured on the Ryzen 7 7730U mini PC: a film is decoded by ffmpeg (on the graphics card), frames are made in between by mvtools on the processor (24 → 60 a second, five presets, a quicker one picked by itself when frames run out), and mpv upscales and shows them (steps from ordinary scalers through AMD FSR 1 to small neural nets, stepping down by itself when drawing is too slow). Seeking, sound and subtitles from the film's file, 8- and 10-bit films. `fvp probe` tells what a computer has (AMD, Intel, Nvidia; several cards) and what is used for what; `fvp bench` measures the presets. A neural net for the frames in between (RIFE) is wired in but not tested
