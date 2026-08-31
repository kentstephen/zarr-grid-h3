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
"""The storm fence: MRMS radar rain on its native 0.01 deg grid, HRRR's claim of the
same storm as a hex-edge line on top.

One univariate field (MRMS precipitation, blues, dark blue to white, dry ground
transparent) drawn as one textured quad per pixel on the store's own lonlat mesh.
On top, THE FENCE: the HRRR analysis precipitation at the same hour, thresholded,
decided per H3 res 6 cell (any / majority / all of the cell's ~4 HRRR pixels wet),
dissolved to boundary edges in the browser and drawn as one yellow line. The two
datasets never share a ramp; the comparison happens in the viewer's eye. The join
never appears as colour: it appears as the fact that the fence can be drawn at all,
because both grids' pixels carry H3 labels.

The fence pipeline is docs/11: HRRR pixels carry res 7 labels (the heat-domes cache),
parented to res 6; MRMS pixels get res 9 labels parented to the same res 6 cells (the
pick chart's join). The kernel ships the HRRR pixel-hours and a STATIC edge table (one
row per hex edge, with the cell on each side); membership, the rule, and the edge
finding run per frame in the browser, so the threshold slider and the rule select move
the fence with zero kernel round-trips. No smoothing anywhere: the fence is raw
membership, judged on screen (docs/11, settled). No "pixel" rule either: the fence is
hex-edge geometry by design, a per-pixel fence would be a different instrument.

Chassis: hrrr-heat-domes.py (constants, MirrorStore, mesh, window loader, transport,
panel language). No DuckDB and no counties here: the browser does the dissolve, and
MRMS's own radar-coverage mask replaces the land mask.

Run: uv run marimo edit storm-fence.py   (or: uv run python fly_fence.py)
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
    # The storm fence

    Hourly MRMS radar precipitation for a region, drawn as the raster itself: one quad
    per 0.01° pixel, blues, dry ground transparent. On top, **the fence**: the hours'
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
    DAYS = 2
    HOURLY_MAX_DAYS = 7
    # The region (W, S, E, N). Required: full-CONUS MRMS is 24.5M px per frame; the
    # mesh and the frames are built for the box's rectangle of the grid.
    BOX = (-100.0, 27.0, -82.0, 40.5)  # gulf coast to Ohio valley, the storm alley
    # ------------------------------------------------------------------ the labels
    # RES_L: MRMS label res (res 9: 0.105 km2 against the ~1 km2 pixel; res 8 is
    # borderline, docs/03). RES_F: the fence / join res, res 6 parents on both sides.
    RES_L, RES_F = 9, 6
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
    BASE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    LABEL_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
    import tempfile as _tempfile

    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"
    MRMS_MIRROR_DIR = CACHE_DIR + "/mrms-mirror/v0.3.0.zarr"
    PROTO_CACHE = ROOT / "proto" / "cache"  # label7.npy: full-grid HRRR res 7 labels (heat domes)
    return (
        BASE_TILES,
        BOX,
        CACHE_DIR,
        DAYS,
        FENCE_RULE,
        FIELD_STOPS,
        FPS,
        HOURLY_MAX_DAYS,
        HRRR_VAR,
        LABEL_TILES,
        MAP_HEIGHT,
        MRMS_CHUNK_H,
        MRMS_MIRROR_DIR,
        MRMS_URL,
        MRMS_VAR,
        PROTO_CACHE,
        RAMP_HI_MM,
        READ_THREADS,
        RES_F,
        RES_L,
        THRESHOLD_MM,
    )


@app.cell
def _(anywidget, traitlets):
    class StormFence(anywidget.AnyWidget):
        """deck.gl SimpleMeshLayer over the MRMS lonlat grid; an H3 res 6 fence on top.

        Kernel -> browser, once (the geometry): `corners` (float32 (ny+1)(nx+1)x2 web
        mercator world coords), `lidx` (uint32 flat grid index of each covered MRMS
        pixel, N), `mpidx` (uint32 N: fence-cell row per MRMS pixel, 0xFFFFFFFF none),
        `cells` (uint64 K res 6 fence cells, sorted), `hpidx` (uint32 Nh: fence-cell
        row per HRRR box pixel), `ea`/`eb` (uint32 E: cell row each side of a hex edge,
        eb 0xFFFFFFFF outside), `exy` (float32 E x 4 edge endpoints, world coords),
        `geom` (JSON). Per window: `frames` (uint8 F x N MRMS, q = 16*sqrt(mm/h),
        255 = no data), `hframes` (uint8 F x Nh HRRR, same coding), `config` (JSON).
        Browser -> kernel: `window` only ({"d0","d1"} from the load button).

        THE FENCE runs here, per frame: count each cell's wet HRRR pixels against the
        threshold, decide by the rule (any / majority / all), then every edge whose two
        sides differ in membership is fence, one LineLayer from typed arrays. The edge
        table is static; threshold and rule never touch the kernel. No smoothing.
        """

        _esm = r"""
        import {Deck, COORDINATE_SYSTEM} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer, LineLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {SimpleMeshLayer} from "https://esm.sh/@deck.gl/mesh-layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .sf { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .sf * { box-sizing: border-box; }
          .sf .sf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .sf .sf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
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
          .sf.sf-collapsed .sf-body, .sf.sf-collapsed .sf-sub { display: none; }
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
          .sf .sf-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; text-align: right; white-space: pre; }
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
              <div class="sf-hud sf-tl"><div class="sf-card sf-panel">
                <div class="sf-head"><span><span class="sf-ttl"></span><span class="sf-sub"></span></span><button class="sf-toggle" title="hide / show (H)">hide</button></div>
                <div class="sf-fields">
                  <button class="sf-b sf-fn sf-on" title="the fence: HRRR &ge; threshold per res 6 cell, dissolved to hex edges (B)">fence</button>
                  <select class="sf-rule" title="fence membership: how much of a res 6 cell the model must wet">
                    <option value="any">rule: any HRRR pixel</option>
                    <option value="majority" selected>rule: majority</option>
                    <option value="all">rule: all pixels</option>
                  </select>
                </div>
                <div class="sf-dim sf-fnote"></div>
                <div class="sf-legend"><span class="sf-num sf-lo">0</span><div class="sf-grad"></div><span class="sf-num sf-hi"></span></div>
                <div class="sf-dim sf-cap">colour: MRMS radar rain this hour, sqrt-scaled, dry transparent</div>
                <div class="sf-body">
                  <div class="sf-row"><span class="sf-k">wet &ge; 0.1 mm/h, covered px</span><span class="sf-num sf-v sf-wet">–</span></div>
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
                  <div class="sf-dim sf-hint">click for both lines at that res 6 cell · space plays · ← → step · B fence · H hide · F fullscreen</div>
                </div>
              </div></div>
              <span class="sf-ruler sf-num"></span>
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
                ruleSel = q(".sf-rule"), fnBtn = q(".sf-fn"), fNote = q(".sf-fnote"), toggle = q(".sf-toggle");

          let geo = {}, cfg = {}, ny = 0, nx = 0, N = 0, Nh = 0, K = 0, E = 0, F = 0;
          let corners = null, lidx = null, mpidx = null, cells = null, hpidx = null, ea = null, eb = null, exy = null, hcount = null;
          let frames = null, hframes = null, meansW = null;
          let mesh = null, tex = [null, null], texData = [null, null], texK = 0, lut = null;
          let cellIndex = new Map();
          let frame = 0, playing = false, timer = null, deck = null, selected = -1;
          let showFence = true, buildMs = 0, fenceInfo = "", fencedCells = 0, paintMs = 0, renderMs = 0, renderT0 = 0;
          let mser = null, hser = null;   // the picked cell's two series
          const FENCE_RGB = [230, 193, 74];
          let HOME = {longitude: -91, latitude: 33.5, zoom: 5, minZoom: 2, maxZoom: 12};

          const fmt = v => Number.isFinite(v) ? v.toFixed(2) + " mm/h" : "no data";
          const thr = () => parseFloat(thrIn.value) || 1;

          // THE GEOMETRY, once: the cell index, the mesh, the pixel counts per cell.
          function loadStatic() {
            try { geo = JSON.parse(model.get("geom") || "{}"); } catch (e) { geo = {}; }
            ny = geo.ny | 0; nx = geo.nx | 0; N = geo.n | 0; Nh = geo.nh | 0; K = geo.k | 0; E = geo.e | 0;
            corners = typed(bytesOf(model.get("corners")), Float32Array);
            lidx = typed(bytesOf(model.get("lidx")), Uint32Array);
            mpidx = typed(bytesOf(model.get("mpidx")), Uint32Array);
            cells = typed(bytesOf(model.get("cells")), BigUint64Array);
            hpidx = typed(bytesOf(model.get("hpidx")), Uint32Array);
            ea = typed(bytesOf(model.get("ea")), Uint32Array);
            eb = typed(bytesOf(model.get("eb")), Uint32Array);
            exy = typed(bytesOf(model.get("exy")), Float32Array);
            if (geo.home) HOME = Object.assign({}, HOME, geo.home);
            if (!corners || !lidx || !cells || !N) return;
            const t0 = performance.now();
            cellIndex = new Map();
            for (let i = 0; i < K; i++) cellIndex.set(cells[i].toString(16), i);
            hcount = new Uint16Array(K);
            for (let i = 0; i < Nh; i++) hcount[hpidx[i]]++;
            const V = (ny + 1) * (nx + 1);
            const pos = new Float32Array(V * 3), uv = new Float32Array(V * 2);
            for (let r = 0; r <= ny; r++) for (let c = 0; c <= nx; c++) {
              const v = r * (nx + 1) + c;
              pos[3 * v] = corners[2 * v]; pos[3 * v + 1] = corners[2 * v + 1]; pos[3 * v + 2] = 0;
              uv[2 * v] = c / nx; uv[2 * v + 1] = r / ny;
            }
            const idx = new Uint32Array(ny * nx * 6); let k = 0;
            for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
              const a = r * (nx + 1) + c, b = a + 1, d = a + nx + 1, e2 = d + 1;
              idx[k++] = a; idx[k++] = d; idx[k++] = b; idx[k++] = b; idx[k++] = d; idx[k++] = e2;
            }
            mesh = {attributes: {positions: {value: pos, size: 3}, texCoords: {value: uv, size: 2}}, indices: {value: idx, size: 1}};
            for (let k2 = 0; k2 < 2; k2++) {
              tex[k2] = document.createElement("canvas"); tex[k2].width = nx; tex[k2].height = ny;
              texData[k2] = tex[k2].getContext("2d").createImageData(nx, ny);
            }
            buildMs = Math.round(performance.now() - t0);
          }

          // THE FRAMES, per window.
          function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px";
            lut = buildLut(cfg.field_stops || ["#0b1d33", "#ffffff"]);
            ttl.textContent = cfg.title || ""; sub.textContent = cfg.subtitle || "";
            syncWindow();
            const u8 = bytesOf(model.get("frames"));
            if (!u8 || !u8.length || !N) { frames = null; hframes = null; F = 0; legend(); return; }
            frames = new Uint8Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            F = Math.floor(frames.length / N);
            const h8 = bytesOf(model.get("hframes"));
            hframes = h8 && h8.length ? new Uint8Array(h8.buffer.slice(h8.byteOffset, h8.byteOffset + h8.byteLength)) : null;
            const wq = QT(0.1);
            meansW = new Float32Array(F);
            for (let f = 0; f < F; f++) { let w = 0, n = 0; for (let i = 0; i < N; i++) { const qv = frames[f * N + i]; if (qv !== 255) { n++; if (qv >= wq) w++; } } meansW[f] = n ? w / n : NaN; }
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

          // THE TEXTURE: covered pixels from the field, q sqrt-coded; dry alpha 0 (the
          // dark basemap is the dry ground); no radar coverage: faint grey.
          function paintTexture() {
            const tp = performance.now();
            texK ^= 1;
            const cv = tex[texK], td = texData[texK], d = td.data; d.fill(0);
            const base = frame * N, qhi = QT(cfg.hi || 20);
            for (let i = 0; i < N; i++) {
              const o = lidx[i] * 4, qv = frames ? frames[base + i] : 255;
              if (qv === 255) { d[o] = 40; d[o + 1] = 44; d[o + 2] = 50; d[o + 3] = 55; continue; }
              if (qv === 0) continue;
              let t = qv / qhi; if (t > 1) t = 1;
              const j = Math.round(t * 255) * 3;
              d[o] = lut[j]; d[o + 1] = lut[j + 1]; d[o + 2] = lut[j + 2];
              d[o + 3] = Math.round(255 * Math.min(1, 0.3 + 0.7 * t));
            }
            cv.getContext("2d").putImageData(td, 0, 0);
            paintMs = Math.round(performance.now() - tp);
            return cv;
          }

          // THE FENCE, per frame: membership per res 6 cell by the rule, then every
          // edge whose two sides differ. Raw membership; no smoothing (docs/11).
          function fenceMembers() {
            const cnt = new Uint16Array(K), base = frame * Nh, thrq = QT(thr());
            for (let i = 0; i < Nh; i++) { const v = hframes[base + i]; if (v !== 255 && v >= thrq) cnt[hpidx[i]]++; }
            const rule = ruleSel.value, member = new Uint8Array(K);
            let m = 0;
            for (let p = 0; p < K; p++) {
              const c = cnt[p], n = hcount[p];
              if (rule === "any" ? c > 0 : rule === "all" ? (n > 0 && c === n) : c * 2 > n) { member[p] = 1; m++; }
            }
            fencedCells = m;
            return member;
          }
          function fenceLayers() {
            fenceInfo = "";
            if (!showFence || !hframes || !F || !E) return [];
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
              id: "fence",
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
              id: "picked",
              data: {length: n, attributes: {getSourcePosition: {value: src, size: 2}, getTargetPosition: {value: dst, size: 2}}},
              getColor: [255, 255, 255, 235], getWidth: 1.4,
              coordinateSystem: COORDINATE_SYSTEM.CARTESIAN, widthUnits: "pixels", widthMinPixels: 1, pickable: false,
            })];
          }
          function fenceLabel() {
            thrV.textContent = thr().toFixed(1);
            fNote.textContent = showFence ? `fence: HRRR analysis ≥ ${thr().toFixed(1)} mm/h, ${ruleSel.value} of the res 6 cell's ~4 pixels, hex edges, no smoothing` : "";
          }
          fnBtn.onclick = () => { showFence = !showFence; fnBtn.classList.toggle("sf-on", showFence); fenceLabel(); update(); };
          let ttimer = null;
          thrIn.oninput = () => { fenceLabel(); if (ttimer) clearTimeout(ttimer); ttimer = setTimeout(update, 60); };
          ruleSel.onchange = () => { fenceLabel(); update(); };
          opacIn.oninput = () => { opacV.textContent = parseFloat(opacIn.value).toFixed(2); update(); };

          const tiles = (id, url, opacity) => new TileLayer({
            id, data: url, tileSize: 256, minZoom: 0, maxZoom: 19, opacity, pickable: false,
            renderSubLayers: p => { const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          function layers() {
            const out = [];
            if (cfg.base_tiles) out.push(tiles("base", cfg.base_tiles, 1.0));
            if (mesh) {
              out.push(new SimpleMeshLayer({
                id: "raster", data: [0], mesh, texture: paintTexture(),
                coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                getPosition: d => [0, 0, 0], getColor: [255, 255, 255, 255],
                material: false, pickable: false, opacity: parseFloat(opacIn.value),
                textureParameters: {minFilter: "nearest", magFilter: "nearest"},
                parameters: {depthTest: false},
              }));
              out.push(...fenceLayers());
              out.push(...pickedLayers());
            }
            if (cfg.label_tiles) out.push(tiles("labels", cfg.label_tiles, 0.6));
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
          // THE SERIES: the one place the two datasets share a frame. Cell means, per
          // hour, radar vs model, straight from the frame matrices via the label maps.
          function buildSeries(s) {
            mser = new Float32Array(F).fill(NaN); hser = new Float32Array(F).fill(NaN);
            const mpx = [], hpx = [];
            for (let i = 0; i < N; i++) if (mpidx[i] === s) mpx.push(i);
            for (let i = 0; i < Nh; i++) if (hpidx[i] === s) hpx.push(i);
            for (let f = 0; f < F; f++) {
              let sm = 0, n = 0;
              for (const i of mpx) { const qv = frames[f * N + i]; if (qv !== 255) { sm += MM_OF(qv); n++; } }
              if (n) mser[f] = sm / n;
              if (hframes) {
                let sh = 0, nh = 0;
                for (const i of hpx) { const qv = hframes[f * Nh + i]; if (qv !== 255) { sh += MM_OF(qv); nh++; } }
                if (nh) hser[f] = sh / nh;
              }
            }
            return [mpx.length, hpx.length];
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
              if (root.classList.contains("sf-collapsed")) { root.classList.remove("sf-collapsed"); toggle.textContent = "hide"; }
              const [nm, nh] = buildSeries(s);
              cname.textContent = `cell ${cells[s].toString(16)}`;
              howEl.textContent = `res ${geo.res_f || 6} · ${nm} MRMS px · ${nh} HRRR px · radar / model this hour:`;
            } else { selected = -1; root.classList.remove("sf-picked"); }
            update();
          }
          q(".sf-clear").onclick = () => { selected = -1; root.classList.remove("sf-picked"); update(); };

          const rulerText = () => `${N.toLocaleString()} MRMS px · ${Nh.toLocaleString()} HRRR px · ${K.toLocaleString()} res 6 cells · ${E.toLocaleString()} edges\nindex + mesh ${buildMs} ms · ${F} frames · paint ${paintMs} ms · render ${renderMs} ms` + (fenceInfo ? "\n" + fenceInfo : "");
          function update() {
            if (!deck) return;
            const ls = layers(); renderT0 = performance.now();
            deck.setProps({layers: ls});
            slider.value = String(frame);
            stampV.textContent = (cfg.labels && cfg.labels[frame]) ? cfg.labels[frame] : `frame ${frame}`;
            stats(); drawChart();
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
          toggle.onclick = () => { root.classList.toggle("sf-collapsed"); toggle.textContent = root.classList.contains("sf-collapsed") ? "show" : "hide"; };
          q(".sf-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen?.(); };
          mapEl.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px"; });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "BUTTON") return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".sf-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { toggle.click(); }
            else if (ev.key === "b" || ev.key === "B") { fnBtn.click(); }
          });

          function boot() {
            loadStatic(); loadFrames();
            deck = new Deck({
              parent: mapEl, initialViewState: HOME, controller: true, layers: layers(),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); console.error(e); },
              onAfterRender: () => { if (renderT0) { renderMs = Math.round(performance.now() - renderT0); renderT0 = 0; ruler.textContent = rulerText(); } },
            });
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".sf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
            mapEl.addEventListener("pointerup", ev => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !deck) return;
              const r = mapEl.getBoundingClientRect();
              try { const ll = deck.getViewports()[0].unproject([ev.clientX - r.left, ev.clientY - r.top]); pick(ll[0], ll[1]); }
              catch (e) { ruler.textContent = "unproject: " + e.message; }
            }, true);
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:corners", () => { loadStatic(); loadFrames(); update(); });
          model.on("change:frames", () => { loadFrames(); update(); });
          model.on("change:config", () => { loadFrames(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (deck) deck.finalize(); };
        }
        export default {render};
        """
        corners = traitlets.Bytes(b"").tag(sync=True)
        lidx = traitlets.Bytes(b"").tag(sync=True)
        mpidx = traitlets.Bytes(b"").tag(sync=True)
        cells = traitlets.Bytes(b"").tag(sync=True)
        hpidx = traitlets.Bytes(b"").tag(sync=True)
        ea = traitlets.Bytes(b"").tag(sync=True)
        eb = traitlets.Bytes(b"").tag(sync=True)
        exy = traitlets.Bytes(b"").tag(sync=True)
        geom = traitlets.Unicode("{}").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        hframes = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        window = traitlets.Unicode("").tag(sync=True)

    return (StormFence,)


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
def _(BOX, PROTO_CACHE, RES_F, RES_L, cells_to_wkb_polygons, change_resolution, coordinates_to_cells, hlat, hlon, mg, mlat, mlon, mo, mtimes, np, pa):
    # THE GEOMETRY, ONCE. The MRMS box rectangle, its lonlat mesh corners in web
    # mercator world coords, the radar-coverage mask (a mature hour's finite pixels),
    # res 9 labels -> res 6 parents per covered MRMS pixel; the HRRR box rectangle, its
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
    _dlat = float(mlat[1] - mlat[0])  # negative
    _dlon = float(mlon[1] - mlon[0])
    _elat = np.concatenate([mlat[r0:r1] - _dlat / 2, [mlat[r1 - 1] + _dlat / 2]])
    _elon = np.concatenate([mlon[c0:c1] - _dlon / 2, [mlon[c1 - 1] + _dlon / 2]])

    def _wm(lo, la):
        _x = (lo + 180.0) / 360.0 * 512.0
        _y = (1.0 - np.log(np.tan(np.radians(la)) + 1.0 / np.cos(np.radians(la))) / np.pi) / 2.0 * 512.0
        return _x, 512.0 - _y

    _EX, _EY = np.meshgrid(_elon, _elat)
    _cwx, _cwy = _wm(_EX.ravel(), _EY.ravel())
    corners = np.stack([_cwx, _cwy], axis=-1).astype(np.float32).reshape(ny + 1, nx + 1, 2)

    # coverage: the static radar mask, from a mature hour (docs/11: bit-identical
    # across hours; the newest 1-2 hours are partially filled, so read T-6)
    _cov = np.isfinite(mg["precipitation_surface"][mtimes.size - 6, r0:r1, c0:c1])
    lidx = np.flatnonzero(_cov.ravel()).astype(np.uint32)
    N = int(lidx.size)
    _plat = np.repeat(mlat[r0:r1], nx)[lidx]
    _plon = np.tile(mlon[c0:c1], ny)[lidx]
    _lab9 = np.asarray(coordinates_to_cells(_plat, _plon, int(RES_L))).astype(np.uint64)
    _uniq9 = int(np.unique(_lab9).size)
    _par6m = np.asarray(change_resolution(pa.array(_lab9), int(RES_F))).astype(np.uint64)

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
    _mp = np.searchsorted(fcells, _par6m)
    _mp[_mp >= K] = 0
    mpidx = np.where(fcells[_mp] == _par6m, _mp, np.uint32(0xFFFFFFFF)).astype(np.uint32)

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
        "ny": ny, "nx": nx, "n": N, "nh": Nh, "k": K, "e": E, "res_f": int(RES_F),
        "home": {"longitude": _clon0, "latitude": _clat0, "zoom": float(np.clip(np.log2(1000 * 360 / (256 * (_E - _W))), 3, 8))},
    }
    pix_stats = (
        f"box rows {r0}:{r1} cols {c0}:{c1} ({ny}x{nx} = {ny * nx:,} MRMS px, {N:,} covered) · "
        f"{_uniq9:,} res {RES_L} labels for {N:,} px ({'unique' if _uniq9 == N else 'NOT unique'}) · "
        f"HRRR {hny}x{hnx} = {Nh:,} px in {len(hblocks)} blocks · {K:,} res {RES_F} fence cells · "
        f"{E:,} edges ({_singles.size:,} outer, {_odd} odd groups) · "
        f"{_gtime.perf_counter() - _gt0:.1f}s · {(corners.nbytes + exy.nbytes + mpidx.nbytes + hpidx.nbytes + lidx.nbytes) / 1e6:.0f} MB static to the browser"
    )
    return E, K, N, Nh, c0, c1, corners, ea, eb, exy, fcells, geom, hblocks, hpidx, lidx, mpidx, ny, nx, pix_stats, r0, r1


@app.cell
def _():
    # Kernel-side memo across window loads.
    HOLD = {"key": None, "mq": None, "hq": None, "times": None, "stats": "", "hi": 20.0}
    return (HOLD,)


@app.cell
def _(StormFence, corners, ea, eb, exy, fcells, geom, hpidx, json, lidx, mo, mpidx):
    # THE WIDGET, BUILT ONCE with the geometry; frames and config are set by the wiring
    # cell below, so a window change never rebuilds the mesh or the edge table.
    film = mo.ui.anywidget(
        StormFence(
            corners=corners.astype("<f4").tobytes(),
            lidx=lidx.astype("<u4").tobytes(),
            mpidx=mpidx.astype("<u4").tobytes(),
            cells=fcells.astype("<u8").tobytes(),
            hpidx=hpidx.astype("<u4").tobytes(),
            ea=ea.astype("<u4").tobytes(),
            eb=eb.astype("<u4").tobytes(),
            exy=exy.astype("<f4").tobytes(),
            geom=json.dumps(geom),
        )
    )
    film
    return (film,)


@app.cell
def _(DAYS, HOURLY_MAX_DAYS, N, Nh, film, htimes, json, mo, mtimes, np):
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
    win_cfg = {
        "first": _first.isoformat(),
        "last": _last.isoformat(),
        "d0": _d0.isoformat(),
        "d1": _d1.isoformat(),
        "hourly_max": HOURLY_MAX_DAYS,
        "cost": f"MRMS ~{max(2, int(n_days * 24 * N / 20e6))}s + HRRR chunk (seconds mirrored, minutes from S3)",
        "n_px": int(N + Nh),
    }
    mo.stop(n_days > HOURLY_MAX_DAYS, mo.md(f"**{n_days} days is over the {HOURLY_MAX_DAYS}-day limit.** Shorten the window."))
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[m]")
    t1 = min((np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[m]"), _last_t)
    window_note = f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to {np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    return n_days, t0, t1, win_cfg, window_note


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the wait in the next cell comes from

    The MRMS read is the box slice straight off the sharded Zarr, 24 hours at a time,
    quantised to a byte per pixel-hour as it lands (q = 16·√(mm/h), so 0.1 mm/h is
    still 5 steps up and 250 mm/h fits). The HRRR read is the heat-domes block read:
    each 45 × 45 store block's window hours out of its 2,160-hour chunk, in threads,
    through the disk mirror. Both stores mirror closed time chunks to disk; the
    youngest chunk always comes from the wire.
    """)
    return


@app.cell
def _(HOLD, HRRR_VAR, MRMS_VAR, N, Nh, READ_THREADS, RAMP_HI_MM, c0, c1, hblocks, hg, hmirror, htimes, lidx, mg, mstore, mtimes, np, r0, r1, t0, t1):
    # THE READ. MRMS: 24-hour slabs of the box rectangle, covered pixels kept,
    # quantised in place. HRRR: block-wise into (F, Nh), threaded. Memoised.
    import time as _rtime
    from concurrent.futures import ThreadPoolExecutor as _Pool

    _key = (str(t0), str(t1), int(N), int(Nh))
    if HOLD["key"] == _key and HOLD["mq"] is not None:
        mq, hq, frame_times, ramp_hi = HOLD["mq"], HOLD["hq"], HOLD["times"], HOLD["hi"]
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
        mq = np.empty((F, N), np.uint8)
        _wet = []
        _mh0, _mm0 = mstore.hits, mstore.misses
        for _s0 in range(0, F, 24):
            _s1 = min(_s0 + 24, F)
            _blk = _mv[_mi0 + _s0 : _mi0 + _s1, r0:r1, c0:c1].reshape(_s1 - _s0, -1)[:, lidx] * 3600.0
            mq[_s0:_s1] = _quant(_blk)
            _w = _blk[_blk > 0.1]
            if _w.size:
                _wet.append(_w[:: max(1, _w.size // 20000)])
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
        _vals = np.concatenate(_wet) if _wet else np.array([1.0])
        ramp_hi = float(RAMP_HI_MM) if RAMP_HI_MM else max(2.0, float(np.percentile(_vals, 99)))
        read_stats = (
            f"{F} hours · MRMS {F * N:,} px-hours in {_t_mrms:.1f}s "
            f"(mirror {mstore.hits - _mh0} ranges from disk, {mstore.misses - _mm0} fetched) · "
            f"HRRR {F * Nh:,} px-hours in {_rtime.perf_counter() - _rt0 - _t_mrms:.1f}s "
            f"(mirror {hmirror.hits - _hh0} / {hmirror.misses - _hm0}) · ramp top p99 wet {ramp_hi:.1f} mm/h"
        )
        HOLD["key"], HOLD["mq"], HOLD["hq"], HOLD["times"], HOLD["stats"], HOLD["hi"] = _key, mq, hq, frame_times, read_stats, ramp_hi
    return frame_times, hq, mq, ramp_hi, read_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using the map.** Space plays, arrows step, drag the slider to scrub, `B` toggles
    the fence, `H` folds the panel, `F` goes fullscreen. The **threshold** slider and
    the **rule** select move the fence with no kernel round-trip: membership is decided
    per res 6 cell from the HRRR pixel-hours already in the browser, and the fence is
    the set of hex edges whose two sides disagree. Click anywhere for the res 6 cell's
    two lines: radar (MRMS pixels in the cell, mean) against model (HRRR pixels in the
    cell, mean), the one place the two datasets share a frame. **live** loads the last
    two UTC days; the newest one to two MRMS hours are partially filled.
    """)
    return


@app.cell
def _(
    BASE_TILES,
    FENCE_RULE,
    FIELD_STOPS,
    FPS,
    LABEL_TILES,
    MAP_HEIGHT,
    THRESHOLD_MM,
    film,
    frame_times,
    hq,
    json,
    mq,
    n_days,
    np,
    ramp_hi,
    read_stats,
    win_cfg,
    window_note,
):
    # THE WIRING: re-runs on every window change and only pushes JSON + bytes at the
    # existing widget. Config, then hframes, then frames: the JS recomputes on frames.
    film.config = json.dumps(
        {
            "labels": [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in frame_times],
            "hi": ramp_hi,
            "field_stops": FIELD_STOPS,
            "threshold": THRESHOLD_MM,
            "rule": FENCE_RULE,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "base_tiles": BASE_TILES or "",
            "label_tiles": LABEL_TILES or "",
            "title": "the storm fence · MRMS radar vs HRRR analysis",
            "subtitle": f"{window_note} · {n_days} days · field: MRMS 0.01° · fence: HRRR ≥ threshold per H3 res 6 cell",
            "meta": read_stats,
            "win": win_cfg,
            "autoplay": False,
        }
    )
    film.hframes = hq.tobytes()
    film.frames = mq.tobytes()
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
