# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
# ]
# ///
"""The storm fence, hexified: MRMS radar rain as H3 res 7 cell means, HRRR's claim
of the same storm as a hex-edge line on top. The long-window build of storm-fence.py.

Same instrument, different field carrier. The raster notebook ships one byte per MRMS
pixel-hour (2.43M px at the default box) and hits V8's string cap at 6 days (docs/11).
Here the kernel folds each hour to res 7 cell means (~17x smaller) and the browser
fills the cells as one SolidPolygonLayer with per-vertex colours: closed rings, ring
starts, uint64 cells, the hex-waves binary-attribute carrier. The window cap moves to
about a month and is COMPUTED from the box's cell count, not advertised. Res 7 holds
~5 MRMS px per cell, so the fill has no holes; cell means smooth convective cores,
which is the thesis's cost, judged on screen against the raster look.

THE FENCE is unchanged in meaning: HRRR analysis >= threshold, decided per res 6 cell
(any / majority / all of the cell's ~4 pixels), dissolved to hex edges in the browser,
drawn silver. To keep threshold and rule live without shipping HRRR pixel-hours, the
kernel ships per-cell ORDER STATISTICS per hour: the max (any), the floor(n/2)+1-th
largest (majority), the min (all), plus the cell mean for the pick chart. Each rule is
then exact: a cell is a member iff its rule's statistic clears the threshold. No
smoothing anywhere; raw membership, hex-edge geometry by design (docs/11, settled).

THE MATCH (right panel): the fence scored against radar, per res 6 cell per hour.
Radar membership is the SAME any / majority / all rule over the cell's res 7 means,
px-weighted; cells where radar reports nothing are not scored. Hit / miss / false
alarm, CSI, POD, FAR, this hour and as a window series. Both HUD panels collapse
to a small discrete button (H, M). This is the season instrument: load Helene to
Milton, scrub, and watch the line.

Everything else is storm-fence.py: the stores, the mirrors, the box, the labels, the
edge table, the transport, the panel language. The raster stays the short-window look.

Run: uv run marimo edit storm-fence-hex.py   (or: uv run python fly_fence.py storm-fence-hex.py)
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import pathlib
    import sys

    import anywidget
    import marimo as mo
    import numpy as np
    import pyarrow as pa
    import traitlets
    import zarr
    from h3ronpy import change_resolution
    from h3ronpy.vector import cells_to_wkb_polygons, coordinates_to_cells
    from obstore.store import HTTPStore
    from zarr.storage import ObjectStore

    ROOT = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
    sys.path.insert(0, str(ROOT / "join"))
    import hrrr_mirror

    return (
        HTTPStore,
        ObjectStore,
        ROOT,
        anywidget,
        cells_to_wkb_polygons,
        change_resolution,
        coordinates_to_cells,
        hrrr_mirror,
        json,
        mo,
        np,
        pa,
        traitlets,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The storm fence, hexified

    Hourly MRMS radar precipitation for a region, folded to H3 res 7 cell means: one
    hex per ~5 km² (~5 radar px), blues, dry ground transparent. On top, **the fence**: the hours'
    HRRR analysis precipitation, thresholded and decided per H3 res 6 cell, dissolved
    to hex-edge boundaries in the browser. Agreement is the fence hugging the colour;
    disagreement is rain spilling out of the pen, or an empty pen on dark ground.

    **Where the numbers come from.** [dynamical.org](https://dynamical.org/)'s Zarr
    builds on source.coop, read anonymously: MRMS CONUS analysis hourly (0.01°,
    `precipitation_surface`, the QC'd blend) and NOAA HRRR analysis (3 km Lambert,
    `precipitation_surface`). Both are live to roughly the hour; the newest one to
    two MRMS hours arrive partially filled.

    **The join.** HRRR pixels carry res 7 H3 labels, parented to res 6. MRMS pixels
    carry res 9 labels, parented to the same res 6 cells. The fence and the click
    chart live on those shared parents; nothing is resampled anywhere.
    """)
    return


@app.cell
def _(ROOT):
    # ------------------------------------------------------------------ the weather
    MRMS_URL = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/dynamical/noaa-mrms-conus-analysis-hourly/v0.3.0.zarr"
    MRMS_VAR = "precipitation_surface"  # the QC'd blend; kg m-2 s-1, x3600 = mm/h
    MRMS_CHUNK_H = 648  # 27 days per time chunk / shard
    HRRR_VAR = "precipitation_surface"
    # Opening window: an int is the last DAYS UTC days through the newest common hour
    # (LIVE); a ("YYYY-MM-DD", "YYYY-MM-DD") tuple is a fixed window.
    DAYS = ("2024-09-19", "2024-10-16")  # the bad stretch: Helene forms -> Milton exits
    # No HOURLY_MAX_DAYS here: the cap is computed downstream from the box's cell
    # counts against V8's string cap (the raster notebook's 6-day wall, docs/11).
    # `frames` ships as FOUR part-traits (4x headroom, ~90 days at this box); READ
    # time is the real cost of a month, minutes cold.
    # The region (W, S, E, N). Required: full-CONUS MRMS is 24.5M px per frame; the
    # mesh and the frames are built for the box's rectangle of the grid.
    BOX = (-100.0, 20.0, -60.0, 41.0)  # the widest SE stage the data allows: MRMS ends at lat 20 / lon -60 (HRRR cone at 21.1 / -60.9; beyond it the fill has no fence, which is honest)
    # ------------------------------------------------------------------ the labels
    # RES_L: MRMS label res (res 9: 0.105 km2 against the ~1 km2 pixel; res 8 is
    # borderline, docs/03). RES_F: the fence / join res, res 6 parents on both sides.
    # RES_C: the FILL res, res 7 MRMS cell means (~5 px per cell, no holes; HRRR at
    # res 7 WOULD have holes, ~9 km2 px vs ~5 km2 cells, so the fence stays res 6).
    RES_L, RES_F, RES_C = 9, 6, 7
    READ_THREADS = 8
    # ------------------------------------------------------------------ the fence
    THRESHOLD_MM = 1.0  # mm/h; the HUD slider is the live version
    FENCE_RULE = "majority"  # any | majority | all of the cell's HRRR pixels wet
    # ------------------------------------------------------------------ the film
    # Field ramp: dark blue to white (protan-safe, single hue). The low end fades to
    # transparent in the shader-side alpha, so dry ground stays the basemap's dark.
    FIELD_STOPS = ["#0b1d33", "#123a63", "#1d5a96", "#3b83bf", "#74b2d9", "#b8d9ec", "#ffffff"]
    RAMP_HI_MM = None  # mm/h at the last stop, or None for the window's p99 of wet pixels
    FPS = 8
    MAP_HEIGHT = 640
    BASE_STYLE = "https://tiles.openfreemap.org/styles/dark"  # OpenFreeMap Dark: keyless, no registration, no limits
    import tempfile as _tempfile

    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"
    MRMS_MIRROR_DIR = CACHE_DIR + "/mrms-mirror/v0.3.0.zarr"
    PROTO_CACHE = ROOT / "proto" / "cache"  # label7.npy: full-grid HRRR res 7 labels (heat domes)
    return (
        BASE_STYLE,
        BOX,
        CACHE_DIR,
        DAYS,
        FENCE_RULE,
        FIELD_STOPS,
        FPS,
        HRRR_VAR,
        MAP_HEIGHT,
        MRMS_CHUNK_H,
        MRMS_MIRROR_DIR,
        MRMS_URL,
        MRMS_VAR,
        PROTO_CACHE,
        RAMP_HI_MM,
        READ_THREADS,
        RES_C,
        RES_F,
        RES_L,
        THRESHOLD_MM,
    )


@app.cell
def _(anywidget, traitlets):
    class StormFenceHex(anywidget.AnyWidget):
        """deck.gl SolidPolygonLayer over H3 res 7 cells; an H3 res 6 fence on top.

        Kernel -> browser, once (the geometry): `verts` (float32 V x 2 web mercator
        world coords, closed rings), `starts` (uint32 K7+1 ring starts), `c7p`
        (uint32 K7: fence-cell row per res 7 cell, 0xFFFFFFFF none), `cnt7` (uint16
        K7: MRMS px per res 7 cell, the weights), `cells` (uint64 K res 6 fence
        cells, sorted), `ea`/`eb` (uint32 E: cell row each side of a hex edge, eb
        0xFFFFFFFF outside), `exy` (float32 E x 4 edge endpoints, world coords),
        `geom` (JSON). Per window: `frames0..frames3` (uint8, row-split parts of
        the F x K7 MRMS res 7 cell means, q = 16*sqrt(mm/h), 255 = no data; four
        parts = 4x the V8 string cap), `hany`/`hmaj`/`hall` (uint8 F x K HRRR
        per-res-6-cell order statistics: max, floor(n/2)+1-th largest, min) and
        `hmean` (uint8 F x K cell mean), same coding, `config` (JSON).
        Browser -> kernel: `window` only ({"d0","d1"} from the load button).

        THE FENCE runs here, per frame, and is EXACT per rule: a cell is a member
        iff its rule's order statistic clears the threshold (any = max, majority =
        the c*2>n statistic, all = min), then every edge whose two sides differ in
        membership is fence, one LineLayer from typed arrays. The edge table is
        static; threshold and rule never touch the kernel. No smoothing.
        """

        _esm = r"""
        import {COORDINATE_SYSTEM} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {LineLayer, SolidPolygonLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import maplibregl from "https://esm.sh/maplibre-gl@4.7.1";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .sf { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .sf * { box-sizing: border-box; }
          .sf .sf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .sf .sf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .sf .sf-ml { position: absolute; inset: 0; }
          .sf .sf-map:fullscreen { height: 100vh !important; width: 100vw; }
          .sf .sf-hud { position: absolute; z-index: 5; }
          .sf .sf-hud.sf-tl { top: .6rem; left: .6rem; width: 23rem; max-width: calc(100% - 1.2rem); }
          .sf .sf-hud.sf-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .sf .sf-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .sf .sf-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }
          .sf .sf-ttl { font-weight: 600; }
          .sf .sf-sub { color: var(--dim); display: block; margin-top: .1rem; }
          .sf .sf-fields { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
          .sf .sf-fields button.sf-b { flex: 0 0 auto; font-size: 11px; padding: .12rem .4rem; min-width: 0; }
          .sf .sf-fields button.sf-on { background: #3a3f2a; border-color: var(--accent); color: #fff; }
          .sf .sf-fields select { font-size: 11px; padding: .12rem .3rem; }
          .sf .sf-legend { display: flex; align-items: center; gap: .45rem; margin-top: .45rem; }
          .sf .sf-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .sf .sf-row { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin-top: .4rem; }
          .sf .sf-row .sf-v { font-size: 15px; }
          .sf .sf-row .sf-k { color: var(--dim); }
          .sf .sf-cell { margin-top: .35rem; display: none; }
          .sf.sf-picked .sf-cell { display: block; }
          .sf .sf-how { color: var(--dim); font-size: 11px; }
          .sf .sf-chart { display: block; width: 100%; height: 108px; margin-top: .3rem; cursor: crosshair; }
          .sf .sf-key { font-size: 11px; margin-top: .15rem; }
          .sf .sf-key .sf-mr { color: #cfe6f7; }
          .sf .sf-key .sf-hr { color: var(--accent); }
          .sf .sf-toggle, .sf .sf-clear { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .sf .sf-toggle:hover, .sf .sf-clear:hover { color: var(--ink); }
          .sf .sf-params { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .sf .sf-p { display: grid; grid-template-columns: 6.2rem 1fr 3.6rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .sf .sf-p label { color: var(--dim); }
          .sf .sf-transport { display: flex; align-items: center; gap: .55rem; }
          .sf .sf-stamp { font-size: 15px; min-width: 11.5rem; }
          .sf .sf-stamp small { display: block; font-size: 10px; color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
          .sf .sf-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .sf .sf-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .sf .sf-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .sf input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .sf button.sf-b, .sf select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .sf button.sf-b:hover, .sf select:hover { background: #2b323b; }
          .sf button:focus-visible, .sf select:focus-visible, .sf input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .sf .sf-dim { color: var(--dim); }
          .sf .sf-win { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .4rem; margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .sf .sf-win input[type=date] { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .15rem .3rem; font: inherit; color-scheme: dark; min-width: 0; }
          .sf .sf-win .sf-note { flex-basis: 100%; }
          .sf .sf-win .sf-note.sf-bad { color: var(--accent); }
          .sf .sf-win button.sf-load:disabled { opacity: .55; cursor: default; }
          .sf .sf-ruler { display: none; position: absolute; right: .6rem; bottom: 3.9rem; color: var(--dim); z-index: 5; text-align: right; white-space: pre; background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .35rem .5rem; pointer-events: none; }
          .sf .sf-hud.sf-tr { top: .6rem; right: .6rem; width: 20rem; max-width: calc(100% - 1.2rem); }
          .sf .sf-hud .sf-mini { display: none; background: var(--panel); border: 1px solid rgba(255,255,255,.14); color: var(--dim); cursor: pointer; font: inherit; padding: .2rem .5rem; backdrop-filter: blur(6px); }
          .sf .sf-hud .sf-mini:hover { color: var(--ink); }
          .sf.sf-showr .sf-ruler { display: block; }
          .sf .sf-hud.sf-min { width: auto; }
          .sf .sf-hud.sf-min > .sf-card { display: none; }
          .sf .sf-hud.sf-min .sf-mini { display: inline-block; }
          .sf .sf-mchart { height: 120px; }
          .sf .sf-tr .sf-how { margin-top: .25rem; }
          @media (max-width: 720px) { .sf .sf-stamp { min-width: 0; } .sf .sf-hud.sf-tl { width: calc(100% - 1.2rem); } }
        `;

        function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
        function buildLut(stops) {
          const rgb = stops.map(hexToRgb), lut = new Uint8Array(256 * 3);
          for (let i = 0; i < 256; i++) {
            const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
            for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
          }
          return lut;
        }
        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset ?? 0, v.byteLength);
          return null;
        }
        const typed = (u8, T) => u8 && u8.length ? new T(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)) : null;
        const NONE = 4294967295;
        const MM_OF = q => (q / 16) * (q / 16);        // uint8 -> mm/h (q = 16 sqrt(mm))
        const QT = mm => 16 * Math.sqrt(Math.max(mm, 0));

        function render({model, el}) {
          el.innerHTML = "";
          const root = document.createElement("div"); root.className = "sf";
          root.innerHTML = `<style>${CSS}</style>
            <div class="sf-map">
              <div class="sf-hud sf-tl"><button class="sf-mini sf-mini-l" title="controls (H)">☰</button><div class="sf-card sf-panel">
                <div class="sf-head"><span><span class="sf-ttl"></span><span class="sf-sub"></span></span><button class="sf-toggle" title="collapse to a button (H)">hide</button></div>
                <div class="sf-fields">
                  <button class="sf-b sf-fn sf-on" title="the fence: HRRR &ge; threshold per res 6 cell, dissolved to hex edges (B)">fence</button>
                  <button class="sf-b sf-fill sf-on" title="the fill: fence cells orange, opacity = the share of the cell's reporting radar that backs the fence this hour (V)">fill</button>
                  <select class="sf-rule" title="fence membership: how much of a res 6 cell the model must wet">
                    <option value="any">rule: any HRRR pixel</option>
                    <option value="majority" selected>rule: majority</option>
                    <option value="all">rule: all pixels</option>
                  </select>
                </div>
                <div class="sf-dim sf-fnote"></div>
                <div class="sf-legend"><span class="sf-num sf-lo">0</span><div class="sf-grad"></div><span class="sf-num sf-hi"></span></div>
                <div class="sf-dim sf-cap">colour: MRMS radar rain this hour, res 7 cell mean, sqrt-scaled, dry transparent</div>
                <div class="sf-body">
                  <div class="sf-row"><span class="sf-k">wet &ge; 0.1 mm/h, of covered ground</span><span class="sf-num sf-v sf-wet">–</span></div>
                  <div class="sf-cell">
                    <div class="sf-row"><span class="sf-k sf-cname">–</span><span><span class="sf-num sf-v sf-cval">–</span> <button class="sf-clear" title="clear">×</button></span></div>
                    <div class="sf-how sf-num"></div>
                    <canvas class="sf-chart" height="108"></canvas>
                    <div class="sf-key"><span class="sf-mr">— radar (MRMS, cell mean)</span> &nbsp; <span class="sf-hr">— model (HRRR, cell mean)</span></div>
                  </div>
                  <div class="sf-params">
                    <div class="sf-p"><label>threshold</label><input type="range" class="sf-thr" min="0.1" max="10" step="0.1"><span class="sf-num sf-thrv"></span></div>
                    <div class="sf-p"><label>opacity</label><input type="range" class="sf-opac" min="0" max="1" step="0.05" value="0.9"><span class="sf-num sf-opacv">0.90</span></div>
                  </div>
                  <div class="sf-win">
                    <input type="date" class="sf-d0" title="first UTC day, inclusive" aria-label="window start (UTC day)"><span class="sf-dim">to</span><input type="date" class="sf-d1" title="last UTC day, inclusive" aria-label="window end (UTC day)">
                    <button class="sf-b sf-load" title="read this window">load</button>
                    <button class="sf-b sf-live" title="the last two UTC days through the newest common hour">live</button>
                    <span class="sf-dim sf-note"></span>
                  </div>
                  <div class="sf-dim sf-hint">click for both lines at that res 6 cell · space plays · ← → step · B fence · V fill · H panel · M match · R stats · F fullscreen</div>
                </div>
              </div></div>
              <span class="sf-ruler sf-num"></span>
              <div class="sf-hud sf-tr">
                <button class="sf-mini sf-mini-r" title="the match (M)">‹</button>
                <div class="sf-card">
                  <div class="sf-head"><span><span class="sf-ttl">the match</span><span class="sf-sub">model fence vs radar, same rule, per res 6 cell</span></span><button class="sf-toggle sf-mtoggle" title="collapse to a button (M)">hide</button></div>
                  <div class="sf-row"><span class="sf-k">CSI this hour</span><span class="sf-num sf-v sf-csi">–</span></div>
                  <div class="sf-row"><span class="sf-k">POD / FAR</span><span class="sf-num sf-podfar">–</span></div>
                  <div class="sf-row"><span class="sf-k">hit · miss · false alarm</span><span class="sf-num sf-hmf">–</span></div>
                  <canvas class="sf-chart sf-mchart" height="120"></canvas>
                  <div class="sf-key"><span style="color:#ffffff">— CSI</span> &nbsp; <span style="color:#6db1f2">— POD</span> &nbsp; <span style="color:#e6c14a">— FAR</span> &nbsp; <span class="sf-dim">6 h smoothed · raw faint</span></div>
                  <div class="sf-how">scored where radar reports: radar side is the same rule over the cell's res 7 means, px-weighted · CSI = hit / (hit + miss + false)</div>
                </div>
              </div>
              <div class="sf-hud sf-bl"><div class="sf-card sf-transport">
                <button class="sf-b sf-prev" title="step back (←)">‹</button>
                <button class="sf-b sf-play" title="play / pause (space)">▶</button>
                <button class="sf-b sf-next" title="step forward (→)">›</button>
                <div class="sf-track"><div class="sf-ticks"></div><input class="sf-frame" type="range" min="0" max="0" value="0" step="1" aria-label="frame"></div>
                <div class="sf-stamp sf-num"><small class="sf-stampk">hour (UTC)</small><span class="sf-stampv">–</span></div>
                <select class="sf-fps" title="frames per second"><option>2</option><option>4</option><option>6</option><option>8</option><option>12</option><option>24</option></select>
                <button class="sf-b sf-full" title="fullscreen (F)">⛶</button>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".sf-map"), playBtn = q(".sf-play"), slider = q(".sf-frame"), ticks = q(".sf-ticks"),
                stampV = q(".sf-stampv"), fpsSel = q(".sf-fps"), grad = q(".sf-grad"),
                loEl = q(".sf-lo"), hiEl = q(".sf-hi"), chart = q(".sf-chart"), ruler = q(".sf-ruler"),
                ttl = q(".sf-ttl"), sub = q(".sf-sub"), wetEl = q(".sf-wet"),
                cname = q(".sf-cname"), cval = q(".sf-cval"), howEl = q(".sf-how"),
                d0In = q(".sf-d0"), d1In = q(".sf-d1"), loadBtn = q(".sf-load"), liveBtn = q(".sf-live"), noteEl = q(".sf-note"),
                thrIn = q(".sf-thr"), thrV = q(".sf-thrv"), opacIn = q(".sf-opac"), opacV = q(".sf-opacv"),
                ruleSel = q(".sf-rule"), fnBtn = q(".sf-fn"), fNote = q(".sf-fnote"), toggle = q(".sf-toggle"),
                tlHud = q(".sf-tl"), trHud = q(".sf-tr"), mToggle = q(".sf-mtoggle"),
                csiEl = q(".sf-csi"), podfarEl = q(".sf-podfar"), hmfEl = q(".sf-hmf"), mchart = q(".sf-mchart");

          let geo = {}, cfg = {}, K7 = 0, V = 0, K = 0, E = 0, F = 0;
          let verts = null, starts = null, c7p = null, cnt7 = null, cells = null, ea = null, eb = null, exy = null;
          let frames = null, hplanes = {}, meansW = null;
          let data = null, color = [null, null], cbuf = 0, lut = null;
          let chIdx = null, chStart = null, mkey = "", mcsi = null, mpod = null, mfar = null;
          let cellIndex = new Map();
          let frame = 0, playing = false, timer = null, deck = null, map = null, labelBase = null, selected = -1;
          let showFence = true, buildMs = 0, fenceInfo = "", fencedCells = 0, paintMs = 0, renderMs = 0, renderT0 = 0;
          let showFill = true, fillc = [null, null], fbuf = 0, fdata = null;
          let mser = null, hser = null;   // the picked cell's two series
          const FENCE_RGB = [205, 210, 216];   // silver; the accent stays for the HUD and the pick
          const PICK_RGB = [230, 193, 74];
          let HOME = {longitude: -91, latitude: 33.5, zoom: 5, minZoom: 2, maxZoom: 12};

          const fmt = v => Number.isFinite(v) ? v.toFixed(2) + " mm/h" : "no data";
          const thr = () => parseFloat(thrIn.value) || 1;

          // THE GEOMETRY, once: the cell index, the rings, the colour buffers.
          function loadStatic() {
            try { geo = JSON.parse(model.get("geom") || "{}"); } catch (e) { geo = {}; }
            K7 = geo.k7 | 0; V = geo.v | 0; K = geo.k | 0; E = geo.e | 0;
            verts = typed(bytesOf(model.get("verts")), Float32Array);
            starts = typed(bytesOf(model.get("starts")), Uint32Array);
            c7p = typed(bytesOf(model.get("c7p")), Uint32Array);
            cnt7 = typed(bytesOf(model.get("cnt7")), Uint16Array);
            cells = typed(bytesOf(model.get("cells")), BigUint64Array);
            ea = typed(bytesOf(model.get("ea")), Uint32Array);
            eb = typed(bytesOf(model.get("eb")), Uint32Array);
            exy = typed(bytesOf(model.get("exy")), Float32Array);
            if (geo.home) HOME = Object.assign({}, HOME, geo.home);
            if (!verts || !starts || !cells || !K7) return;
            const t0 = performance.now();
            cellIndex = new Map();
            for (let i = 0; i < K; i++) cellIndex.set(cells[i].toString(16), i);
            color = [new Uint8Array(V * 4), new Uint8Array(V * 4)];
            data = {length: K7, startIndices: starts, attributes: {getPolygon: {value: verts, size: 2}}};
            fillc = [new Uint8Array(V * 4), new Uint8Array(V * 4)];
            for (const c2 of fillc) for (let v2 = 0; v2 < V; v2++) { const o = v2 * 4; c2[o] = 235; c2[o + 1] = 140; c2[o + 2] = 60; }
            fdata = {length: K7, startIndices: starts, attributes: {getPolygon: {value: verts, size: 2}}};
            chStart = new Uint32Array(K + 1);
            for (let i = 0; i < K7; i++) if (c7p[i] !== NONE) chStart[c7p[i] + 1]++;
            for (let p = 0; p < K; p++) chStart[p + 1] += chStart[p];
            chIdx = new Uint32Array(chStart[K]);
            const cur = Uint32Array.from(chStart.subarray(0, K));
            for (let i = 0; i < K7; i++) { const p = c7p[i]; if (p !== NONE) chIdx[cur[p]++] = i; }
            buildMs = Math.round(performance.now() - t0);
          }

          // THE FRAMES, per window. Gzipped traits stream-decode straight into ONE
          // preallocated array: no concatenation copy, and the model retains only
          // the compressed bytes. The clen guard makes the decode run exactly once,
          // when the last part trait lands.
          async function gunzipInto(u8, out, off) {
            const rd = new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
            for (;;) { const {done, value} = await rd.read(); if (done) return off; out.set(value, off); off += value.length; }
          }
          let loadSeq = 0;
          async function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px";
            lut = buildLut(cfg.field_stops || ["#0b1d33", "#ffffff"]);
            ttl.textContent = cfg.title || ""; sub.textContent = cfg.subtitle || "";
            syncWindow();
            const nf = cfg.nf | 0, clen = cfg.clen || [];
            const parts = [];
            for (const nm of ["frames0", "frames1", "frames2", "frames3"]) {
              const b = bytesOf(model.get(nm));
              if (b && b.length) parts.push(b);
            }
            if (!K7 || !nf || !clen.length || parts.length !== clen.length || parts.some((b, i) => b.length !== clen[i])) { frames = null; hplanes = {}; F = 0; legend(); return; }
            const seq = ++loadSeq;
            let buf, planes;
            try {
              buf = new Uint8Array(nf * K7);
              let off = 0;
              for (const p of parts) off = await gunzipInto(p, buf, off);
              if (seq !== loadSeq) return;
              if (off !== nf * K7) { frames = null; hplanes = {}; F = 0; legend(); return; }
              planes = {};
              for (const k2 of ["hany", "hmaj", "hall", "hmean"]) {
                const b8 = bytesOf(model.get(k2));
                if (!b8 || !b8.length || !K) { planes[k2] = null; continue; }
                const o2 = new Uint8Array(nf * K);
                await gunzipInto(b8, o2, 0);
                planes[k2] = o2;
              }
              if (seq !== loadSeq) return;
            } catch (e) { ruler.textContent = "decode: " + e.message; console.error(e); return; }
            frames = buf; F = nf;
            hplanes = planes;
            const wq = QT(0.1);
            meansW = new Float32Array(F);
            for (let f = 0; f < F; f++) { let w = 0, n = 0; for (let i = 0; i < K7; i++) { const qv = frames[f * K7 + i]; if (qv !== 255) { const c2 = cnt7[i]; n += c2; if (qv >= wq) w += c2; } } meansW[f] = n ? w / n : NaN; }
            mkey = ""; mcsi = null;
            slider.max = String(Math.max(0, F - 1));
            if (frame >= F) frame = 0;
            fpsSel.value = String(cfg.fps || 8);
            if (!thrIn.dataset.set) { thrIn.value = cfg.threshold ?? 1; ruleSel.value = cfg.rule || "majority"; thrIn.dataset.set = "1"; }
            if (selected >= 0) buildSeries(selected);
            legend(); fenceLabel();
            const labels = cfg.labels || [];
            let html = "";
            for (let f = 1; f < labels.length; f++) {
              const d0 = labels[f - 1].slice(0, 10), d1 = labels[f].slice(0, 10);
              if (d0 !== d1) html += `<i style="left:${(f / (F - 1) * 100).toFixed(2)}%"></i>`;
            }
            ticks.innerHTML = F > 1 ? html : "";
          }

          // THE FILL: per res 7 cell from the field, q sqrt-coded, written to every
          // vertex of the cell's ring; dry alpha 0 (the dark basemap is the dry
          // ground); no data this hour: faint grey.
          function paint() {
            const tp = performance.now();
            cbuf ^= 1;
            const c = color[cbuf], base = frame * K7, qhi = QT(cfg.hi || 20);
            for (let i = 0; i < K7; i++) {
              const qv = frames ? frames[base + i] : 255;
              let r = 0, g2 = 0, b = 0, a = 0;
              if (qv === 255) { r = 40; g2 = 44; b = 50; a = 55; }
              else if (qv !== 0) {
                let t = qv / qhi; if (t > 1) t = 1;
                const j = Math.round(t * 255) * 3;
                r = lut[j]; g2 = lut[j + 1]; b = lut[j + 2];
                a = Math.round(255 * Math.min(1, 0.3 + 0.7 * t));
              }
              for (let v = starts[i]; v < starts[i + 1]; v++) { const o = v * 4; c[o] = r; c[o + 1] = g2; c[o + 2] = b; c[o + 3] = a; }
            }
            paintMs = Math.round(performance.now() - tp);
            return c;
          }

          // THE FILL (the match made spatial): each fence cell's area, orange (the
          // protan-safe complement of the blues ramp), its
          // opacity the px-weighted share of the cell's reporting radar that backs
          // the fence this hour. A full pen is solid-ish silver, a marginal one a
          // faint wash, a false alarm stays empty; the storm reads through beneath.
          function paintFill() {
            fbuf ^= 1;
            const c = fillc[fbuf], plane = planeOf(), thrq = QT(thr()), base7 = frame * K7, base = frame * K;
            const fr = new Uint8Array(K);
            if (frames && plane && chStart) {
              for (let p = 0; p < K; p++) {
                const v = plane[base + p];
                if (v === 255 || v < thrq) continue;
                const a2 = chStart[p], b2 = chStart[p + 1];
                let wet = 0, tot = 0;
                for (let j = a2; j < b2; j++) { const i = chIdx[j], w = cnt7[i], qv = frames[base7 + i]; if (qv === 255) continue; tot += w; if (qv >= thrq) wet += w; }
                if (tot) fr[p] = Math.round(180 * wet / tot);
              }
            }
            for (let i = 0; i < K7; i++) {
              const p = c7p[i], a = p === NONE ? 0 : fr[p];
              for (let v2 = starts[i]; v2 < starts[i + 1]; v2++) c[v2 * 4 + 3] = a;
            }
            return c;
          }

          // THE FENCE, per frame: membership per res 6 cell by the rule, then every
          // edge whose two sides differ. Raw membership; no smoothing (docs/11).
          function fenceMembers() {
            const key = ruleSel.value === "any" ? "hany" : ruleSel.value === "all" ? "hall" : "hmaj";
            const plane = hplanes[key], base = frame * K, thrq = QT(thr());
            const member = new Uint8Array(K);
            let m = 0;
            for (let p = 0; p < K; p++) { const v = plane[base + p]; if (v !== 255 && v >= thrq) { member[p] = 1; m++; } }
            fencedCells = m;
            return member;
          }
          function fenceLayers() {
            fenceInfo = "";
            if (!showFence || !hplanes.hmaj || !F || !E) return [];
            const t0 = performance.now();
            const member = fenceMembers();
            const act = [];
            for (let e = 0; e < E; e++) {
              const ma = member[ea[e]], mb = eb[e] === NONE ? 0 : member[eb[e]];
              if (ma !== mb) act.push(e);
            }
            const n = act.length, src = new Float32Array(n * 2), dst = new Float32Array(n * 2);
            for (let j = 0; j < n; j++) {
              const e = act[j];
              src[2 * j] = exy[4 * e]; src[2 * j + 1] = exy[4 * e + 1];
              dst[2 * j] = exy[4 * e + 2]; dst[2 * j + 1] = exy[4 * e + 3];
            }
            fenceInfo = `fence (${ruleSel.value} ≥ ${thr().toFixed(1)} mm/h): ${fencedCells.toLocaleString()} cells · ${n.toLocaleString()} edges · ${Math.round(performance.now() - t0)} ms`;
            if (!n) return [];
            return [new LineLayer({
              id: "fence", beforeId: labelBase || undefined,
              data: {length: n, attributes: {getSourcePosition: {value: src, size: 2}, getTargetPosition: {value: dst, size: 2}}},
              getColor: [FENCE_RGB[0], FENCE_RGB[1], FENCE_RGB[2], 235], getWidth: 1.6,
              coordinateSystem: COORDINATE_SYSTEM.CARTESIAN, widthUnits: "pixels", widthMinPixels: 1, pickable: false,
            })];
          }
          function pickedLayers() {
            if (selected < 0) return [];
            const segs = [];
            for (let e = 0; e < E; e++) if (ea[e] === selected || eb[e] === selected) segs.push(e);
            const n = segs.length, src = new Float32Array(n * 2), dst = new Float32Array(n * 2);
            for (let j = 0; j < n; j++) {
              const e = segs[j];
              src[2 * j] = exy[4 * e]; src[2 * j + 1] = exy[4 * e + 1];
              dst[2 * j] = exy[4 * e + 2]; dst[2 * j + 1] = exy[4 * e + 3];
            }
            return [new LineLayer({
              id: "picked", beforeId: labelBase || undefined,
              data: {length: n, attributes: {getSourcePosition: {value: src, size: 2}, getTargetPosition: {value: dst, size: 2}}},
              getColor: [PICK_RGB[0], PICK_RGB[1], PICK_RGB[2], 235], getWidth: 1.4,
              coordinateSystem: COORDINATE_SYSTEM.CARTESIAN, widthUnits: "pixels", widthMinPixels: 1, pickable: false,
            })];
          }
          function fenceLabel() {
            thrV.textContent = thr().toFixed(1);
            fNote.textContent = showFence ? `fence: HRRR analysis ≥ ${thr().toFixed(1)} mm/h, ${ruleSel.value} of the res 6 cell's ~4 pixels, hex edges, no smoothing` : "";
          }
          fnBtn.onclick = () => { showFence = !showFence; fnBtn.classList.toggle("sf-on", showFence); fenceLabel(); update(); };
          const flBtn = q(".sf-fill");
          flBtn.onclick = () => { showFill = !showFill; flBtn.classList.toggle("sf-on", showFill); update(); };
          let ttimer = null;
          thrIn.oninput = () => { fenceLabel(); if (ttimer) clearTimeout(ttimer); ttimer = setTimeout(update, 60); };
          ruleSel.onchange = () => { fenceLabel(); update(); };
          opacIn.oninput = () => { opacV.textContent = parseFloat(opacIn.value).toFixed(2); update(); };

          function layers() {
            const out = [];
            if (data) {
              data.attributes.getFillColor = {value: paint(), size: 4};
              out.push(new SolidPolygonLayer({
                id: "hex", data, _normalize: false, _windingOrder: "CW", filled: true, extruded: false,
                beforeId: labelBase || undefined,
                coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                material: false, pickable: false, opacity: parseFloat(opacIn.value),
                parameters: {depthTest: false},
                updateTriggers: {getFillColor: cbuf},
              }));
              if (showFill && frames && hplanes.hmaj && chStart) {
                fdata.attributes.getFillColor = {value: paintFill(), size: 4};
                out.push(new SolidPolygonLayer({
                  id: "hexfill", data: fdata, _normalize: false, _windingOrder: "CW", filled: true, extruded: false,
                  beforeId: labelBase || undefined,
                  coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                  material: false, pickable: false,
                  parameters: {depthTest: false},
                  updateTriggers: {getFillColor: fbuf},
                }));
              }
              out.push(...fenceLayers());
              out.push(...pickedLayers());
            }
            return out;
          }
          function legend() {
            if (!lut) return;
            const stops = [];
            for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            loEl.textContent = "0";
            hiEl.textContent = (cfg.hi || 20).toFixed(0) + "+ mm/h";
          }

          function stats() {
            wetEl.textContent = meansW && F ? (meansW[frame] * 100).toFixed(1) + " %" : "–";
            if (selected >= 0 && F && mser) cval.textContent = fmt(mser[frame]) + " / " + fmt(hser[frame]);
          }
          // THE SERIES: the one place the two datasets share a frame. Radar: the res 6
          // cell's res 7 children, px-weighted mean. Model: the shipped hmean plane.
          function buildSeries(s) {
            mser = new Float32Array(F).fill(NaN); hser = new Float32Array(F).fill(NaN);
            const ch = [];
            let wpx = 0;
            for (let i = 0; i < K7; i++) if (c7p[i] === s) { ch.push(i); wpx += cnt7[i]; }
            const hm = hplanes.hmean;
            for (let f = 0; f < F; f++) {
              let sm = 0, n = 0;
              for (const i of ch) { const qv = frames[f * K7 + i]; if (qv !== 255) { sm += MM_OF(qv) * cnt7[i]; n += cnt7[i]; } }
              if (n) mser[f] = sm / n;
              if (hm) { const hv = hm[f * K + s]; if (hv !== 255) hser[f] = MM_OF(hv); }
            }
            return [ch.length, wpx];
          }
          function drawChart() {
            if (selected < 0 || !frames || F < 2 || !mser) return;
            const w = chart.clientWidth || 300, h = chart.height;
            if (chart.width !== w) chart.width = w;
            const g = chart.getContext("2d");
            g.clearRect(0, 0, w, h);
            const L = 44, R = 4, T = 6, B = 14;
            const X = f => L + (w - L - R) * f / (F - 1);
            let hi = 1;
            for (let f = 0; f < F; f++) { if (Number.isFinite(mser[f])) hi = Math.max(hi, mser[f]); if (Number.isFinite(hser[f])) hi = Math.max(hi, hser[f]); }
            const Y = v => T + (h - T - B) * (1 - v / hi);
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - R, Y(0)); g.moveTo(L, Y(hi)); g.lineTo(w - R, Y(hi)); g.stroke();
            const tv = thr();
            if (tv < hi) { g.setLineDash([3, 3]); g.strokeStyle = "#8b929c"; g.beginPath(); g.moveTo(L, Y(tv)); g.lineTo(w - R, Y(tv)); g.stroke(); g.setLineDash([]); }
            g.fillStyle = "#8b929c"; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "right";
            g.fillText(hi.toFixed(1), L - 4, Y(hi) + 4); g.fillText("0", L - 4, Y(0) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, h - 3);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - R, h - 3);
            const line = (ser, colr) => {
              g.strokeStyle = colr; g.lineWidth = 1.5; g.beginPath();
              let pen = false;
              for (let f = 0; f < F; f++) { const v = ser[f]; if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
              g.stroke();
            };
            line(hser, "#e6c14a"); line(mser, "#cfe6f7");
            g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), h - B); g.stroke();
            for (const [ser, colr] of [[mser, "#ffffff"], [hser, "#e6c14a"]]) {
              const cv = ser[frame];
              if (Number.isFinite(cv)) { g.fillStyle = colr; g.beginPath(); g.arc(X(frame), Y(cv), 2.5, 0, 6.283); g.fill(); }
            }
          }
          // THE MATCH: the fence scored against radar, per res 6 cell, per hour.
          // Model membership: the shipped order-stat plane. Radar membership: the
          // SAME rule over the cell's res 7 children, px-weighted; cells where the
          // radar reports nothing are not scored. CSI = hit / (hit + miss + false).
          // The hour's numbers are computed live; the window series is recomputed
          // 250 ms after the threshold or rule settles.
          let matchMs = 0, mtimer = null;
          function tallyFrame(f, plane, thrq, rule) {
            const base7 = f * K7, base = f * K;
            let h = 0, ms = 0, fa = 0;
            for (let p = 0; p < K; p++) {
              const a = chStart[p], b = chStart[p + 1];
              if (a === b) continue;
              let wet = 0, tot = 0, seen = 0, allwet = 1, anywet = 0;
              for (let j = a; j < b; j++) {
                const i = chIdx[j], w = cnt7[i], qv = frames[base7 + i];
                tot += w;
                if (qv === 255) { allwet = 0; continue; }
                seen = 1;
                if (qv >= thrq) { wet += w; anywet = 1; } else allwet = 0;
              }
              if (!seen) continue;
              const r = rule === "any" ? anywet : rule === "all" ? allwet : (wet * 2 > tot ? 1 : 0);
              const v = plane[base + p], m = v !== 255 && v >= thrq ? 1 : 0;
              if (m && r) h++; else if (r) ms++; else if (m) fa++;
            }
            return [h, ms, fa];
          }
          const planeOf = () => hplanes[ruleSel.value === "any" ? "hany" : ruleSel.value === "all" ? "hall" : "hmaj"];
          function matchSeries() {
            if (!frames || !planeOf() || !chStart || !F) return;
            const t0 = performance.now();
            const plane = planeOf(), thrq = QT(thr()), rule = ruleSel.value;
            mcsi = new Float32Array(F).fill(NaN); mpod = new Float32Array(F).fill(NaN); mfar = new Float32Array(F).fill(NaN);
            for (let f = 0; f < F; f++) {
              const [h, ms, fa] = tallyFrame(f, plane, thrq, rule);
              if (h + ms + fa) mcsi[f] = h / (h + ms + fa);
              if (h + ms) mpod[f] = h / (h + ms);
              if (h + fa) mfar[f] = fa / (h + fa);
            }
            mkey = rule + "|" + thr().toFixed(2);
            matchMs = Math.round(performance.now() - t0);
          }
          function drawMatch() {
            if (!frames || !planeOf() || !chStart || !F) { csiEl.textContent = "–"; podfarEl.textContent = "–"; hmfEl.textContent = "–"; return; }
            const [h, ms, fa] = tallyFrame(frame, planeOf(), QT(thr()), ruleSel.value);
            const csi = h + ms + fa ? h / (h + ms + fa) : NaN;
            const pod = h + ms ? h / (h + ms) : NaN, far = h + fa ? fa / (h + fa) : NaN;
            const fm = v => Number.isFinite(v) ? v.toFixed(2) : "–";
            csiEl.textContent = fm(csi);
            podfarEl.textContent = fm(pod) + " / " + fm(far);
            hmfEl.textContent = `${h} · ${ms} · ${fa}`;
            const key = ruleSel.value + "|" + thr().toFixed(2);
            if (mkey !== key) { if (mtimer) clearTimeout(mtimer); mtimer = setTimeout(() => { matchSeries(); drawMatch(); }, 250); }
            if (trHud.classList.contains("sf-min") || F < 2 || !mcsi) return;
            const w = mchart.clientWidth || 280, hh = mchart.height;
            if (mchart.width !== w) mchart.width = w;
            const g = mchart.getContext("2d");
            g.clearRect(0, 0, w, hh);
            const L = 26, Rm = 4, T = 5, B = 13;
            const X = f => L + (w - L - Rm) * f / (F - 1), Y = v => T + (hh - T - B) * (1 - v);
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - Rm, Y(0)); g.moveTo(L, Y(1)); g.lineTo(w - Rm, Y(1)); g.stroke();
            g.fillStyle = "#8b929c"; g.font = "10px ui-monospace, Menlo, monospace"; g.textAlign = "right";
            g.fillText("1", L - 4, Y(1) + 4); g.fillText("0", L - 4, Y(0) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, hh - 2);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - Rm, hh - 2);
            const line = (ser, colr, lw) => {
              g.strokeStyle = colr; g.lineWidth = lw; g.beginPath();
              let pen = false;
              for (let f = 0; f < F; f++) { const v = ser[f]; if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
              g.stroke();
            };
            const smooth = ser => { const o2 = new Float32Array(F); for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let d = -3; d <= 3; d++) { const v = ser[f + d]; if (Number.isFinite(v)) { s += v; n++; } } o2[f] = n ? s / n : NaN; } return o2; };
            g.globalAlpha = 0.2; line(mpod, "#6db1f2", 1); line(mfar, "#e6c14a", 1); line(mcsi, "#ffffff", 1);
            g.globalAlpha = 1; line(smooth(mpod), "#6db1f2", 1.4); line(smooth(mfar), "#e6c14a", 1.4); line(smooth(mcsi), "#ffffff", 2);
            g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), hh - B); g.stroke();
          }
          mchart.addEventListener("click", ev => {
            if (F < 2) return;
            const r = mchart.getBoundingClientRect(), L = 26, Rm = 4;
            const t = ((ev.clientX - r.left) - L) / (r.width - L - Rm);
            frame = Math.max(0, Math.min(F - 1, Math.round(t * (F - 1)))); update();
          });

          chart.addEventListener("click", ev => {
            if (F < 2) return;
            const r = chart.getBoundingClientRect(), L = 44, R = 4;
            const t = ((ev.clientX - r.left) - L) / (r.width - L - R);
            frame = Math.max(0, Math.min(F - 1, Math.round(t * (F - 1)))); update();
          });

          // THE WINDOW CONTROL: the one thing that crosses back.
          let loading = false;
          const dayCount = () => {
            const a = Date.parse(d0In.value), b = Date.parse(d1In.value);
            return Number.isFinite(a) && Number.isFinite(b) ? Math.round(Math.abs(b - a) / 864e5) + 1 : 0;
          };
          function checkWindow() {
            const w = cfg.win || {};
            const n = dayCount(), lim = w.hourly_max || 7;
            let bad = "";
            if (!n) bad = "pick both days";
            else if (n > lim) bad = `${n} days is over the ${lim}-day limit`;
            noteEl.classList.toggle("sf-bad", !!bad);
            const mb = w.n_px ? ` · ${(n * 24 * w.n_px / 1e6).toFixed(0)} MB to the browser` : "";
            if (loading) noteEl.textContent = `loading ${n} days…`;
            else noteEl.textContent = bad || `${n} UTC days · ${n * 24} hourly frames${mb} · ${w.cost || ""} · limit ${lim} d`;
            loadBtn.disabled = loading || !!bad;
            return !bad;
          }
          function syncWindow() {
            const w = cfg.win;
            if (!w) return;
            d0In.min = d1In.min = w.first || ""; d0In.max = d1In.max = w.last || "";
            if (w.d0) d0In.value = w.d0;
            if (w.d1) d1In.value = w.d1;
            loading = false; loadBtn.textContent = "load";
            checkWindow();
          }
          const doLoad = () => {
            if (!checkWindow()) return;
            let d0 = d0In.value, d1 = d1In.value;
            if (d1 < d0) [d0, d1] = [d1, d0];
            loading = true; loadBtn.textContent = "loading"; frame = 0; checkWindow();
            model.set("window", JSON.stringify({d0, d1}));
            model.save_changes();
          };
          d0In.onchange = d1In.onchange = checkWindow;
          loadBtn.onclick = doLoad;
          liveBtn.onclick = () => {
            const w = cfg.win || {};
            if (!w.last) return;
            const last = new Date(w.last + "T00:00:00Z"), prev = new Date(last.getTime() - 864e5);
            d0In.value = prev.toISOString().slice(0, 10); d1In.value = w.last;
            doLoad();
          };
          checkWindow();

          // THE PICK: latLngToCell at the fence res into the cell index; the readout is
          // the res 6 cell and its two lines. No pixel-level pick: the chart is the join.
          function pick(lng, lat) {
            let s = -1;
            try { s = cellIndex.get(latLngToCell(lat, lng, geo.res_f || 6)) ?? -1; } catch (e) { s = -1; }
            if (s >= 0 && s !== selected && F) {
              selected = s;
              root.classList.add("sf-picked");
              if (tlHud.classList.contains("sf-min")) minTL(false);
              const [nm, nh] = buildSeries(s);
              cname.textContent = `cell ${cells[s].toString(16)}`;
              howEl.textContent = `res ${geo.res_f || 6} · ${nm} res ${geo.res_c || 7} cells · ${nh} MRMS px · radar / model this hour:`;
            } else { selected = -1; root.classList.remove("sf-picked"); }
            update();
          }
          q(".sf-clear").onclick = () => { selected = -1; root.classList.remove("sf-picked"); update(); };

          const rulerText = () => `${K7.toLocaleString()} res ${geo.res_c || 7} cells · ${V.toLocaleString()} verts · ${K.toLocaleString()} res ${geo.res_f || 6} fence cells · ${E.toLocaleString()} edges\nindex ${buildMs} ms · ${F} frames · paint ${paintMs} ms · render ${renderMs} ms` + (fenceInfo ? "\n" + fenceInfo : "");
          function update() {
            if (!deck) return;
            const ls = layers(); renderT0 = performance.now();
            deck.setProps({layers: ls});
            slider.value = String(frame);
            stampV.textContent = (cfg.labels && cfg.labels[frame]) ? cfg.labels[frame] : `frame ${frame}`;
            stats(); drawChart(); drawMatch();
            ruler.textContent = rulerText();
          }
          function setPlaying(p) {
            playing = p; playBtn.textContent = p ? "❚❚" : "▶";
            if (timer) { clearInterval(timer); timer = null; }
            if (p && F > 1) timer = setInterval(() => { frame = (frame + 1) % F; update(); }, 1000 / (parseFloat(fpsSel.value) || 8));
          }
          const step = d => { if (F) { frame = (frame + d + F) % F; update(); } };
          playBtn.onclick = () => setPlaying(!playing);
          q(".sf-prev").onclick = () => step(-1);
          q(".sf-next").onclick = () => step(1);
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          fpsSel.onchange = () => { if (playing) setPlaying(true); };
          const minTL = m => tlHud.classList.toggle("sf-min", m);
          const minTR = m => { trHud.classList.toggle("sf-min", m); if (!m) drawMatch(); };
          toggle.onclick = () => minTL(true);
          q(".sf-mini-l").onclick = () => minTL(false);
          mToggle.onclick = () => minTR(true);
          q(".sf-mini-r").onclick = () => minTR(false);
          q(".sf-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen?.(); };
          mapEl.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px"; });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "BUTTON") return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".sf-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { minTL(!tlHud.classList.contains("sf-min")); }
            else if (ev.key === "m" || ev.key === "M") { minTR(!trHud.classList.contains("sf-min")); }
            else if (ev.key === "b" || ev.key === "B") { fnBtn.click(); }
            else if (ev.key === "v" || ev.key === "V") { flBtn.click(); }
            else if (ev.key === "r" || ev.key === "R") { root.classList.toggle("sf-showr"); }
          });

          function boot() {
            loadStatic(); loadFrames().then(update);
            if (!document.querySelector("link[data-sf-ml]")) {
              const l = document.createElement("link"); l.rel = "stylesheet";
              l.href = "https://esm.sh/maplibre-gl@4.7.1/dist/maplibre-gl.css"; l.dataset.sfMl = "1"; document.head.appendChild(l);
            }
            const mlEl = document.createElement("div"); mlEl.className = "sf-ml"; mapEl.prepend(mlEl);
            map = new maplibregl.Map({
              container: mlEl, style: cfg.style || "https://tiles.openfreemap.org/styles/dark",
              center: [HOME.longitude, HOME.latitude], zoom: HOME.zoom, minZoom: HOME.minZoom, maxZoom: HOME.maxZoom,
              attributionControl: {compact: true}, dragRotate: false, pitchWithRotate: false,
            });
            map.touchZoomRotate.disableRotation();
            map.on("error", e => { if (e && e.error) console.error(e.error); });
            map.on("style.load", () => {
              const sym = map.getStyle().layers.find(l2 => l2.type === "symbol");
              labelBase = sym ? sym.id : null;
              update();
            });
            deck = new MapboxOverlay({
              interleaved: true, layers: layers(),
              onAfterRender: () => { if (renderT0) { renderMs = Math.round(performance.now() - renderT0); renderT0 = 0; ruler.textContent = rulerText(); } },
            });
            map.addControl(deck);
            new ResizeObserver(() => map.resize()).observe(mapEl);
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".sf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
            mapEl.addEventListener("pointerup", ev => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !map) return;
              const r = mapEl.getBoundingClientRect();
              try { const ll = map.unproject([ev.clientX - r.left, ev.clientY - r.top]); pick(ll.lng, ll.lat); }
              catch (e) { ruler.textContent = "unproject: " + e.message; }
            }, true);
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:verts", () => { loadStatic(); update(); loadFrames().then(update); });
          for (const nm of ["frames0", "frames1", "frames2", "frames3"]) model.on("change:" + nm, () => { loadFrames().then(update); });
          model.on("change:config", () => { loadFrames().then(update); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (map) map.remove(); };
        }
        export default {render};
        """
        verts = traitlets.Bytes(b"").tag(sync=True)
        starts = traitlets.Bytes(b"").tag(sync=True)
        c7p = traitlets.Bytes(b"").tag(sync=True)
        cnt7 = traitlets.Bytes(b"").tag(sync=True)
        cells = traitlets.Bytes(b"").tag(sync=True)
        ea = traitlets.Bytes(b"").tag(sync=True)
        eb = traitlets.Bytes(b"").tag(sync=True)
        exy = traitlets.Bytes(b"").tag(sync=True)
        geom = traitlets.Unicode("{}").tag(sync=True)
        frames0 = traitlets.Bytes(b"").tag(sync=True)
        frames1 = traitlets.Bytes(b"").tag(sync=True)
        frames2 = traitlets.Bytes(b"").tag(sync=True)
        frames3 = traitlets.Bytes(b"").tag(sync=True)
        hany = traitlets.Bytes(b"").tag(sync=True)
        hmaj = traitlets.Bytes(b"").tag(sync=True)
        hall = traitlets.Bytes(b"").tag(sync=True)
        hmean = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        window = traitlets.Unicode("").tag(sync=True)

    return (StormFenceHex,)


@app.cell
def _(HRRR_VAR, HTTPStore, MRMS_CHUNK_H, MRMS_MIRROR_DIR, MRMS_URL, MRMS_VAR, ObjectStore, hrrr_mirror, np, zarr):
    # THE STORES, opened once; only metadata, time and the 1-D / 2-D coords are read.
    # MRMS is plain zarr v3 over HTTP behind the same MirrorStore as HRRR (closed time
    # shards to disk, the youngest always fetched). HRRR comes through join/hrrr_mirror.
    import time as _stime

    _st0 = _stime.perf_counter()

    def _dtimes(arr, units):
        _base = np.datetime64(units.split("since")[1].strip().split(".")[0].replace(" ", "T"))
        _u = {"seconds": "s", "minutes": "m", "hours": "h", "days": "D"}[units.split()[0]]
        return (_base + arr.astype(f"timedelta64[{_u}]")).astype("datetime64[m]")

    _inner = ObjectStore(HTTPStore.from_url(MRMS_URL), read_only=True)
    _Tm = zarr.open_group(_inner, mode="r")["time"].shape[0]
    _young_m = (_Tm - 1) // MRMS_CHUNK_H

    def _m_mirrorable(key, _young=_young_m):
        p = key.split("/")
        return len(p) == 5 and p[0] == MRMS_VAR and p[1] == "c" and p[2].isdigit() and int(p[2]) < _young

    mstore = hrrr_mirror.MirrorStore(_inner, MRMS_MIRROR_DIR, _m_mirrorable)
    mg = zarr.open_group(mstore, mode="r")
    mtimes = _dtimes(mg["time"][:], mg["time"].attrs["units"])
    mlat = mg["latitude"][:].astype(np.float64)  # 54.995 -> 20.005, row 0 is north
    mlon = mg["longitude"][:].astype(np.float64)  # -129.995 -> -60.005

    hg, hmirror = hrrr_mirror.open_hrrr([HRRR_VAR])
    htimes = _dtimes(hg["time"][:], hg["time"].attrs["units"])
    hlat = hg["latitude"][:].astype(np.float64)
    hlon = hg["longitude"][:].astype(np.float64)
    store_stats = (
        f"MRMS {mlat.size}x{mlon.size} px, hourly to {np.datetime_as_string(mtimes[-1], unit='m')}Z · "
        f"HRRR {hlat.shape[0]}x{hlat.shape[1]} px, hourly to {np.datetime_as_string(htimes[-1], unit='m')}Z · "
        f"open {_stime.perf_counter() - _st0:.1f}s"
    )
    return hg, hlat, hlon, hmirror, htimes, mg, mlat, mlon, mstore, mtimes, store_stats


@app.cell
def _(BOX, PROTO_CACHE, RES_C, RES_F, RES_L, cells_to_wkb_polygons, change_resolution, coordinates_to_cells, hlat, hlon, mg, mlat, mlon, mo, mtimes, np, pa):
    # THE GEOMETRY, ONCE. The MRMS box rectangle, the radar-coverage mask (a mature
    # hour's finite pixels), res 9 labels -> res 7 FILL cells (rings to the browser)
    # and the px -> cell map; the HRRR box rectangle, its
    # res 6 parents from the heat-domes res 7 label cache; the shared fence cells; the
    # STATIC EDGE TABLE (each hex edge once, the cell row on each side, from the ring
    # WKB by midpoint pairing); the HRRR store blocks the read fetches.
    import time as _gtime

    mo.stop(BOX is None, mo.md("**BOX is required**: full-CONUS MRMS is 24.5M px per frame."))
    mo.stop(not (PROTO_CACHE / "label7.npy").exists(), mo.md("**proto/cache/label7.npy missing**: run hrrr-heat-domes.py once to build the HRRR label cache."))
    _gt0 = _gtime.perf_counter()
    _W, _S, _E, _N = BOX

    # ---- MRMS box, mesh, coverage, labels ------------------------------------------
    _rr = np.flatnonzero((mlat >= _S) & (mlat <= _N))
    _cc = np.flatnonzero((mlon >= _W) & (mlon <= _E))
    r0, r1, c0, c1 = int(_rr[0]), int(_rr[-1]) + 1, int(_cc[0]), int(_cc[-1]) + 1
    ny, nx = r1 - r0, c1 - c0

    def _wm(lo, la):
        _x = (lo + 180.0) / 360.0 * 512.0
        _y = (1.0 - np.log(np.tan(np.radians(la)) + 1.0 / np.cos(np.radians(la))) / np.pi) / 2.0 * 512.0
        return _x, 512.0 - _y

    # coverage: the static radar mask, from a mature hour (docs/11: bit-identical
    # across hours; the newest 1-2 hours are partially filled, so read T-6)
    _cov = np.isfinite(mg["precipitation_surface"][mtimes.size - 6, r0:r1, c0:c1])
    lidx = np.flatnonzero(_cov.ravel()).astype(np.uint32)
    N = int(lidx.size)
    _plat = np.repeat(mlat[r0:r1], nx)[lidx]
    _plon = np.tile(mlon[c0:c1], ny)[lidx]
    _lab9 = np.asarray(coordinates_to_cells(_plat, _plon, int(RES_L))).astype(np.uint64)
    _uniq9 = int(np.unique(_lab9).size)
    _par7 = np.asarray(change_resolution(pa.array(_lab9), int(RES_C))).astype(np.uint64)

    # ---- the FILL cells: res 7, from the covered pixels; every cell has px, no holes
    cells7 = np.unique(_par7)
    K7 = int(cells7.size)
    m7idx = np.searchsorted(cells7, _par7).astype(np.uint32)
    cnt7 = np.bincount(m7idx, minlength=K7).astype(np.uint16)

    # ---- HRRR box, res 6 parents ----------------------------------------------------
    _hin = (hlon >= _W) & (hlon <= _E) & (hlat >= _S) & (hlat <= _N)
    _hr = np.flatnonzero(_hin.any(axis=1))
    _hc = np.flatnonzero(_hin.any(axis=0))
    hr0, hr1, hc0, hc1 = int(_hr[0]), int(_hr[-1]) + 1, int(_hc[0]), int(_hc[-1]) + 1
    hny, hnx = hr1 - hr0, hc1 - hc0
    Nh = hny * hnx
    _lab7 = np.load(PROTO_CACHE / "label7.npy").reshape(hlat.shape)[hr0:hr1, hc0:hc1].ravel()
    _par6h = np.asarray(change_resolution(pa.array(_lab7.astype(np.uint64)), int(RES_F))).astype(np.uint64)

    # ---- the fence cells and the label maps -----------------------------------------
    fcells = np.unique(_par6h)
    K = int(fcells.size)
    hpidx = np.searchsorted(fcells, _par6h).astype(np.uint32)
    _par6c = np.asarray(change_resolution(pa.array(cells7), int(RES_F))).astype(np.uint64)
    _cp = np.searchsorted(fcells, _par6c)
    _cp[_cp >= K] = 0
    c7p = np.where(fcells[_cp] == _par6c, _cp, np.uint32(0xFFFFFFFF)).astype(np.uint32)

    # ---- the edge table: every hex edge once, by ring-midpoint pairing --------------
    # WKB rings in float64 (h3's own vertices; shared edges pair exactly at 1e-6 deg).
    _wkb = cells_to_wkb_polygons(pa.array(fcells))
    _raw = b"".join(_wkb.to_pylist())
    _buf = np.frombuffer(_raw, dtype=np.uint8)
    _e_cell, _e_p0, _e_p1 = [], [], []
    _off = 0
    for _i in range(K):
        _npts = int.from_bytes(_raw[_off + 9 : _off + 13], "little")
        _pts = np.frombuffer(_buf[_off + 13 : _off + 13 + _npts * 16].tobytes(), dtype="<f8").reshape(-1, 2)
        _e_cell.append(np.full(_npts - 1, _i, np.uint32))
        _e_p0.append(_pts[:-1])
        _e_p1.append(_pts[1:])
        _off += 13 + _npts * 16
    assert _off == len(_raw)
    _ec = np.concatenate(_e_cell)
    _p0 = np.concatenate(_e_p0)
    _p1 = np.concatenate(_e_p1)
    _mx = np.rint(((_p0[:, 0] + _p1[:, 0]) / 2 + 180.0) * 1e6).astype(np.int64)
    _my = np.rint(((_p0[:, 1] + _p1[:, 1]) / 2 + 90.0) * 1e6).astype(np.int64)
    _key = _mx * 400_000_000 + _my
    _ord = np.argsort(_key, kind="stable")
    _ks = _key[_ord]
    _first = np.r_[True, _ks[1:] != _ks[:-1]]
    _gstart = np.flatnonzero(_first)
    _gsize = np.diff(np.r_[_gstart, _ks.size])
    _pairs = _gstart[_gsize == 2]
    _singles = _gstart[_gsize == 1]
    _odd = int((_gsize > 2).sum())
    ea = np.concatenate([_ec[_ord[_pairs]], _ec[_ord[_singles]]]).astype(np.uint32)
    eb = np.concatenate([_ec[_ord[_pairs + 1]], np.full(_singles.size, 0xFFFFFFFF, np.uint32)])
    _eidx = np.concatenate([_ord[_pairs], _ord[_singles]])
    _sx, _sy = _wm(_p0[_eidx, 0], _p0[_eidx, 1])
    _tx, _ty = _wm(_p1[_eidx, 0], _p1[_eidx, 1])
    exy = np.stack([_sx, _sy, _tx, _ty], axis=-1).astype(np.float32)
    E = int(ea.size)

    # ---- the FILL rings: every res 7 cell's closed ring, web mercator float32 ------
    # (the hex-waves carrier: verts + ring starts + per-vertex colour; the fast path
    # assumes uniform hexagons, the loop is the pentagon-safe fallback)
    _wkb7 = cells_to_wkb_polygons(pa.array(cells7))
    _raw7 = b"".join(_wkb7.to_pylist())
    _rec = len(_raw7) // K7
    if len(_raw7) == _rec * K7 and (_rec - 13) % 16 == 0 and int.from_bytes(_raw7[9:13], "little") == (_rec - 13) // 16:
        _np7 = (_rec - 13) // 16
        _pts7 = np.frombuffer(_raw7, np.uint8).reshape(K7, _rec)[:, 13:].copy().view("<f8").reshape(K7, _np7, 2)
        starts = (np.arange(K7 + 1, dtype=np.uint32) * _np7).astype(np.uint32)
        _fx, _fy = _wm(_pts7[:, :, 0].ravel(), _pts7[:, :, 1].ravel())
    else:
        _b7 = np.frombuffer(_raw7, dtype=np.uint8)
        _lens, _px7, _py7, _o7 = [], [], [], 0
        for _i7 in range(K7):
            _n7 = int.from_bytes(_raw7[_o7 + 9 : _o7 + 13], "little")
            _p7 = np.frombuffer(_b7[_o7 + 13 : _o7 + 13 + _n7 * 16].tobytes(), dtype="<f8").reshape(-1, 2)
            _lens.append(_n7)
            _px7.append(_p7[:, 0])
            _py7.append(_p7[:, 1])
            _o7 += 13 + _n7 * 16
        starts = np.r_[0, np.cumsum(_lens)].astype(np.uint32)
        _fx, _fy = _wm(np.concatenate(_px7), np.concatenate(_py7))
    verts = np.stack([_fx, _fy], axis=-1).astype(np.float32).reshape(-1, 2)
    V = int(verts.shape[0])

    # ---- the HRRR store blocks (45x45, full-grid aligned) that touch the box --------
    _B = 45
    _hprow = np.arange(Nh, dtype=np.int64).reshape(hny, hnx)
    hblocks = []
    for _j in range(hr0 // _B, -(-hr1 // _B)):
        for _i2 in range(hc0 // _B, -(-hc1 // _B)):
            _ys = slice(max(_j * _B, hr0), min((_j + 1) * _B, hr1))
            _xs = slice(max(_i2 * _B, hc0), min((_i2 + 1) * _B, hc1))
            _sub = _hprow[_ys.start - hr0 : _ys.stop - hr0, _xs.start - hc0 : _xs.stop - hc0].ravel()
            hblocks.append((_ys, _xs, _sub))

    _clon0, _clat0 = (_W + _E) / 2, (_S + _N) / 2
    geom = {
        "k7": K7, "v": V, "k": K, "e": E, "res_f": int(RES_F), "res_c": int(RES_C),
        "home": {"longitude": _clon0, "latitude": _clat0, "zoom": float(np.clip(np.log2(1000 * 360 / (256 * (_E - _W))), 3, 8))},
    }
    pix_stats = (
        f"box rows {r0}:{r1} cols {c0}:{c1} ({ny}x{nx} = {ny * nx:,} MRMS px, {N:,} covered) · "
        f"{_uniq9:,} res {RES_L} labels for {N:,} px ({'unique' if _uniq9 == N else 'NOT unique'}) · "
        f"{K7:,} res {RES_C} fill cells ({V:,} verts, ~{N / K7:.1f} px each) · "
        f"HRRR {hny}x{hnx} = {Nh:,} px in {len(hblocks)} blocks · {K:,} res {RES_F} fence cells · "
        f"{E:,} edges ({_singles.size:,} outer, {_odd} odd groups) · "
        f"{_gtime.perf_counter() - _gt0:.1f}s · {(verts.nbytes + starts.nbytes + exy.nbytes + c7p.nbytes + cnt7.nbytes) / 1e6:.0f} MB static to the browser"
    )
    return E, K, K7, N, Nh, c0, c1, c7p, cnt7, ea, eb, exy, fcells, geom, hblocks, hpidx, lidx, m7idx, pix_stats, r0, r1, starts, verts


@app.cell
def _():
    # Kernel-side memo across window loads.
    HOLD = {"key": None, "mq": None, "ha": None, "hj": None, "hl": None, "hm": None, "times": None, "stats": "", "hi": 20.0}
    return (HOLD,)


@app.cell
def _(StormFenceHex, c7p, cnt7, ea, eb, exy, fcells, geom, json, mo, starts, verts):
    # THE WIDGET, BUILT ONCE with the geometry; frames and config are set by the wiring
    # cell below, so a window change never rebuilds the rings or the edge table.
    film = mo.ui.anywidget(
        StormFenceHex(
            verts=verts.astype("<f4").tobytes(),
            starts=starts.astype("<u4").tobytes(),
            c7p=c7p.astype("<u4").tobytes(),
            cnt7=cnt7.astype("<u2").tobytes(),
            cells=fcells.astype("<u8").tobytes(),
            ea=ea.astype("<u4").tobytes(),
            eb=eb.astype("<u4").tobytes(),
            exy=exy.astype("<f4").tobytes(),
            geom=json.dumps(geom),
        )
    )
    film
    return (film,)


@app.cell
def _(DAYS, K, K7, N, film, htimes, json, mo, mtimes, np):
    # THE WINDOW: the HUD's `window` trait once "load" has been pressed, else the
    # opening default. LIVE = an int DAYS: the last N days through the newest hour
    # both stores have filled.
    import datetime as _wdt

    _last_t = min(mtimes[-1], htimes[-1])
    _first_t = max(mtimes[0], htimes[0])
    _last = _last_t.astype("datetime64[D]").astype(_wdt.date)
    _first = _first_t.astype("datetime64[D]").astype(_wdt.date)
    _req = {}
    try:
        _req = json.loads(film.widget.window or "{}")
    except ValueError:
        _req = {}
    if _req.get("d0") and _req.get("d1"):
        _d0 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d0"])))
        _d1 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d1"])))
    elif isinstance(DAYS, tuple):
        _d0, _d1 = (max(_first, min(_last, _wdt.date.fromisoformat(d))) for d in DAYS)
    else:
        _d0, _d1 = _last - _wdt.timedelta(days=DAYS - 1), _last
    if _d1 < _d0:
        _d0, _d1 = _d1, _d0
    n_days = (_d1 - _d0).days + 1
    # Traits ship gzipped now, so the V8 string cap binds on COMPRESSED bytes and
    # stops being the wall (sparse rain compresses 10-50x). The ceiling is the
    # browser's DECODED arrays: frames (K7 B/frame) + the four planes (K B/frame
    # each); budget 3 GB decoded.
    _decb = 3_000_000_000
    hourly_max = min(90, int(_decb / ((K7 + 4 * K) * 24)))
    win_cfg = {
        "first": _first.isoformat(),
        "last": _last.isoformat(),
        "d0": _d0.isoformat(),
        "d1": _d1.isoformat(),
        "hourly_max": hourly_max,
        "cost": f"MRMS ~{max(2, int(n_days * 24 * N / 20e6))}s + HRRR chunk (seconds mirrored, minutes from S3)",
        "n_px": int(K7 + 4 * K),
    }
    mo.stop(n_days > hourly_max, mo.md(f"**{n_days} days is over the {hourly_max}-day limit.** Shorten the window."))
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[m]")
    t1 = min((np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[m]"), _last_t)
    window_note = f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to {np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    return n_days, t0, t1, win_cfg, window_note


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the wait in the next cell comes from

    The MRMS read is the box slice straight off the sharded Zarr, 24 hours at a time,
    averaged per res 7 cell as it lands and quantised to a byte per cell-hour
    (q = 16·√(mm/h), so 0.1 mm/h is still 5 steps up and 250 mm/h fits). The HRRR
    read is the heat-domes block read: each 45 × 45 store block's window hours out of
    its 2,160-hour chunk, in threads, through the disk mirror, then folded per res 6
    cell to four planes an hour: max (the `any` rule), the majority-order statistic,
    min (`all`), and the cell mean (the pick chart). Both stores mirror closed time
    chunks to disk; the youngest chunk always comes from the wire.
    """)
    return


@app.cell
def _(HOLD, HRRR_VAR, K, K7, MRMS_VAR, N, Nh, READ_THREADS, RAMP_HI_MM, c0, c1, hblocks, hg, hmirror, hpidx, htimes, lidx, m7idx, mg, mstore, mtimes, np, r0, r1, t0, t1):
    # THE READ. MRMS: 24-hour slabs of the box rectangle, covered pixels kept, folded
    # to res 7 cell means per hour. HRRR: block-wise into (F, Nh), threaded, then
    # folded to the four res 6 planes (max / majority statistic / min / mean) that
    # keep threshold and rule live and exact in the browser. Memoised.
    import time as _rtime
    from concurrent.futures import ThreadPoolExecutor as _Pool

    _key = (str(t0), str(t1), int(K7), int(K))
    if HOLD["key"] == _key and HOLD["mq"] is not None:
        mq7, h_any, h_maj, h_all, h_mean = HOLD["mq"], HOLD["ha"], HOLD["hj"], HOLD["hl"], HOLD["hm"]
        frame_times, ramp_hi = HOLD["times"], HOLD["hi"]
        read_stats = HOLD["stats"] + " (memo)"
    else:
        _rt0 = _rtime.perf_counter()
        _mi0 = int(np.searchsorted(mtimes, t0))
        _mi1 = int(np.searchsorted(mtimes, t1, side="right"))
        _hi0 = int(np.searchsorted(htimes, t0))
        _hi1 = int(np.searchsorted(htimes, t1, side="right"))
        frame_times = mtimes[_mi0:_mi1]
        F = int(frame_times.size)
        assert F == _hi1 - _hi0, f"hour mismatch: MRMS {F} vs HRRR {_hi1 - _hi0}"

        def _quant(mmh):
            _q = np.clip(np.rint(16.0 * np.sqrt(np.maximum(mmh, 0.0))), 0, 254)
            return np.where(np.isnan(mmh), 255, _q).astype(np.uint8)

        _mv = mg[MRMS_VAR]
        mq7 = np.empty((F, K7), np.uint8)
        _wet = []
        _mh0, _mm0 = mstore.hits, mstore.misses
        for _s0 in range(0, F, 24):
            _s1 = min(_s0 + 24, F)
            _blk = _mv[_mi0 + _s0 : _mi0 + _s1, r0:r1, c0:c1].reshape(_s1 - _s0, -1)[:, lidx] * 3600.0
            for _fi in range(_s1 - _s0):
                _row = _blk[_fi]
                _ok = np.isfinite(_row)
                _cnt = np.bincount(m7idx[_ok], minlength=K7)
                _sum = np.bincount(m7idx[_ok], weights=_row[_ok].astype(np.float64), minlength=K7)
                _mean = np.divide(_sum, _cnt, out=np.zeros(K7), where=_cnt > 0)
                mq7[_s0 + _fi] = np.where(_cnt > 0, _quant(_mean), np.uint8(255))
                _w = _mean[(_cnt > 0) & (_mean > 0.1)]
                if _w.size:
                    _wet.append(_w[:: max(1, _w.size // 2000)])
        _t_mrms = _rtime.perf_counter() - _rt0

        _hv = hg[HRRR_VAR]
        _hraw = np.full((F, Nh), np.nan, np.float32)
        _hh0, _hm0 = hmirror.hits, hmirror.misses

        def _rd(job):
            _ys, _xs, _sub = job
            _hraw[:, _sub] = _hv[_hi0:_hi1, _ys, _xs].reshape(F, -1)

        with _Pool(READ_THREADS) as _ex:
            list(_ex.map(_rd, hblocks))
        hq = _quant(_hraw * 3600.0)

        # THE PLANES: per res 6 cell per hour, the k-th largest of the cell's pixel
        # values for each rule (any: k=1, majority: k=n//2+1, all: k=n, with n the
        # cell's pixel count and no-data counting as a pixel that is never wet,
        # exactly the raster notebook's browser rule; q order = mm order, so the
        # statistics commute with the quantisation), plus the valid-pixel mean.
        _ord = np.argsort(hpidx, kind="stable")
        _gn = np.bincount(hpidx, minlength=K)
        _G = int(_gn.max())
        _gs = np.r_[0, np.cumsum(_gn)[:-1]].astype(np.int64)
        _colp = (np.arange(Nh, dtype=np.int64) - np.repeat(_gs, _gn)).astype(np.int64)
        _rowp = hpidx[_ord].astype(np.int64)
        _c_all = (_G - _gn).astype(np.int64)
        _c_maj = (_G - (_gn // 2 + 1)).astype(np.int64)
        _hm8 = np.where(hq == 255, np.uint8(0), hq)
        _mmv = (hq.astype(np.float64) / 16.0) ** 2
        h_any = np.empty((F, K), np.uint8)
        h_maj = np.empty((F, K), np.uint8)
        h_all = np.empty((F, K), np.uint8)
        h_mean = np.empty((F, K), np.uint8)
        _pad = np.zeros((K, _G), np.uint8)
        for _f in range(F):
            _pad[:] = 0
            _pad[_rowp, _colp] = _hm8[_f, _ord]
            _srt = np.sort(_pad, axis=1)
            h_any[_f] = _srt[:, _G - 1]
            h_all[_f] = np.take_along_axis(_srt, _c_all[:, None], 1)[:, 0]
            h_maj[_f] = np.take_along_axis(_srt, _c_maj[:, None], 1)[:, 0]
            _okh = hq[_f] != 255
            _cnh = np.bincount(hpidx[_okh], minlength=K)
            _smh = np.bincount(hpidx[_okh], weights=_mmv[_f][_okh], minlength=K)
            h_mean[_f] = np.where(_cnh > 0, _quant(np.divide(_smh, _cnh, out=np.zeros(K), where=_cnh > 0)), np.uint8(255))
        _vals = np.concatenate(_wet) if _wet else np.array([1.0])
        ramp_hi = float(RAMP_HI_MM) if RAMP_HI_MM else max(2.0, float(np.percentile(_vals, 99)))
        read_stats = (
            f"{F} hours · MRMS {F * N:,} px-hours -> {F * K7:,} cell-hours in {_t_mrms:.1f}s "
            f"(mirror {mstore.hits - _mh0} ranges from disk, {mstore.misses - _mm0} fetched) · "
            f"HRRR {F * Nh:,} px-hours -> 4 x {F * K:,} plane-hours in {_rtime.perf_counter() - _rt0 - _t_mrms:.1f}s "
            f"(mirror {hmirror.hits - _hh0} / {hmirror.misses - _hm0}) · ramp top p99 wet cell {ramp_hi:.1f} mm/h"
        )
        HOLD["key"], HOLD["mq"], HOLD["ha"], HOLD["hj"], HOLD["hl"], HOLD["hm"] = _key, mq7, h_any, h_maj, h_all, h_mean
        HOLD["times"], HOLD["stats"], HOLD["hi"] = frame_times, read_stats, ramp_hi
    return frame_times, h_all, h_any, h_maj, h_mean, mq7, ramp_hi, read_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using the map.** Space plays, arrows step, drag the slider to scrub, `B` toggles
    the fence, `H` folds the panel, `F` goes fullscreen. The **threshold** slider and
    the **rule** select move the fence with no kernel round-trip: membership is decided
    per res 6 cell from the HRRR pixel-hours already in the browser, and the fence is
    the set of hex edges whose two sides disagree. Click anywhere for the res 6 cell's
    two lines: radar (the cell's res 7 means, px-weighted) against model (HRRR pixels
    in the cell, mean), the one place the two datasets share a frame. **live** loads the last
    two UTC days; the newest one to two MRMS hours are partially filled.
    """)
    return


@app.cell
def _(
    BASE_STYLE,
    FENCE_RULE,
    FIELD_STOPS,
    FPS,
    MAP_HEIGHT,
    THRESHOLD_MM,
    film,
    frame_times,
    h_all,
    h_any,
    h_maj,
    h_mean,
    json,
    mq7,
    n_days,
    np,
    ramp_hi,
    read_stats,
    win_cfg,
    window_note,
):
    # THE WIRING: re-runs on every window change and only pushes JSON + bytes at the
    # existing widget. Every binary trait ships GZIPPED (the model then holds tens of
    # MB, not a GB); config goes first carrying the compressed part lengths, planes
    # next, frame parts last: the JS decodes exactly once, when the last part lands
    # and every part's compressed length matches config.
    import gzip as _gzip
    _fp = np.array_split(mq7, 4, axis=0)
    _gz = [_gzip.compress(_p.tobytes(), 2) for _p in _fp]
    _pz = [_gzip.compress(_v.tobytes(), 2) for _v in (h_any, h_maj, h_all, h_mean)]
    film.config = json.dumps(
        {
            "labels": [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in frame_times],
            "nf": int(frame_times.size),
            "clen": [len(_z) for _z in _gz],
            "hi": ramp_hi,
            "field_stops": FIELD_STOPS,
            "threshold": THRESHOLD_MM,
            "rule": FENCE_RULE,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "style": BASE_STYLE,
            "title": "the storm fence, hexified · MRMS radar vs HRRR analysis",
            "subtitle": f"{window_note} · {n_days} days · field: MRMS res 7 cell mean · fence: HRRR ≥ threshold per H3 res 6 cell",
            "meta": read_stats,
            "win": win_cfg,
            "autoplay": False,
        }
    )
    film.hany = _pz[0]
    film.hmaj = _pz[1]
    film.hall = _pz[2]
    film.hmean = _pz[3]
    film.frames3 = _gz[3]
    film.frames2 = _gz[2]
    film.frames1 = _gz[1]
    film.frames0 = _gz[0]
    return


@app.cell
def _(mo, pix_stats, read_stats, store_stats):
    mo.md(
        "<br>".join(
            f"<span style='color:#8b929c;font-size:.85em'>{s}</span>"
            for s in (store_stats, pix_stats, read_stats)
        )
    )
    return


if __name__ == "__main__":
    app.run()
