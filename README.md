# FVP — Fluid Video Player

Plays films with frames made in between (24 → 60 a second) and upscaled, as they play. Linux; AMD, Intel and
Nvidia processors and graphics cards, one or several of them.

    git clone https://github.com/kupecjan1980-pixel/fvp && cd fvp
    ./install.sh          # into ~/.local/share/fvp, no root needed
    fvp FILM              # play
    fvp probe             # what this computer has and what FVP will use it for
    fvp bench FILM        # how fast each preset runs here; the best one that keeps up is remembered

Keys: arrows / PgUp / PgDn seek · `i` frames in between on/off · `p` / `P` better / quicker preset ·
`u` next upscaling step · `o` or Tab status · space pause · `f` full screen · `q` quit.

## How it works

    ffmpeg (decodes; on the graphics card: VA-API on AMD and Intel, NVDEC on Nvidia)
      → worker (VapourSynth: makes the frames in between)
      → ffmpeg (stamps the film's own times on them) → mpv (upscales with shaders, shows)

mpv takes the sound and the subtitles from the film's file. The work is done outside mpv because the mpv most
systems ship is built without VapourSynth. A seek starts the chain anew at the new place.

**Frames in between**: `mvtools` (motion vectors on the processor, what Smooth Video Project does) in five
presets, `fastest` to `best`. When mpv runs out of frames twice in 20 s, FVP steps to a quicker preset by itself.
A neural net (`rife`, on any Vulkan card) is wired in but NOT TESTED: it needs the VapourSynth RIFE plugin and a
card that runs it at speed.

**Upscaling** steps, cheapest first: `plain`, `sharp` (ordinary scalers), `fsr` (AMD FSR 1, any card),
`light` / `good` / `best` (small neural nets: FSRCNNX, ArtCNN). A card starts at the step its strength suggests
and FVP steps down by itself while drawing a frame takes more than 60 % of its time on the screen.

## What runs where

Measured only on the first row; the rest is the plan (`fvp probe` shows what a computer gets, the automatic
stepping corrects wrong guesses).

| Computer | Decoding | Frames in between | Upscaling starts at |
|---|---|---|---|
| Ryzen 7 7730U, Vega 8 (measured) | VA-API | mvtools `fast` at 1080p (1.5–2.7× real time) | `fsr` (8.5 ms a frame at 4K) |
| Ryzen 7 8700G, Radeon 780M | VA-API | mvtools; RIFE at 720p | `light` |
| OneXFly, Ryzen AI 9 HX 370, 890M | VA-API | mvtools; RIFE at 720p–1080p | `light` |
| ONEXPLAYER 3, Intel Arc | VA-API | mvtools; RIFE at 720p–1080p | `light` |
| NEXT 2, Ryzen AI Max+ 395, 8060S | VA-API | RIFE at 1080p | `good` |
| i9-9900T + RTX 5060 | NVDEC | RIFE at 1080p (TensorRT later) | `good` |
| Xeon E5-2699A v4 + Radeon VII + Titan V | the screen's card | mvtools on 44 threads, or RIFE on the card without the screen | `good` |
| Ryzen 9 7950X3D + RX 7900 XTX | VA-API | RIFE up to 1440p, mvtools `best` | `best` |
| Ryzen 9 9950X3D + RTX 5090 | NVDEC | RIFE at 4K (TensorRT later) | `best` |

With two cards the screen's card decodes, upscales and shows, and the frames in between go to the strongest
other card, so the two jobs don't get in each other's way.

## Not there yet

- RIFE: untested (above); TensorRT on Nvidia and OpenVINO on Intel not started.
- No seek bar or mouse seeking (mpv can't seek in a pipe; the keys do it).
- HDR films: their colours are passed on to mpv, not checked on an HDR screen.
- 4K films on weak processors: mvtools at 4K needs a strong processor; FVP turns the frames off when even
  `fastest` can't keep up.

## Licence

GPL-2.0 (see LICENSE). The upscaling shaders are fetched by install.sh from their authors and keep their own
licences: AMD FSR 1 (MIT), FSRCNNX (igv), ArtCNN (MIT).
