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
#     "duckdb>=1.5.5",
#     "pyproj>=3.7",
# ]
# ///
"""HRRR heat domes on the native 3 km grid. H3 is the transformation, never the picture.

The map draws dynamical.org's HRRR analysis (2 m heat index, hourly) as one textured
quad per pixel on a mesh built from the store's own Lambert grid: pixel corners through
pyproj, nearest-filter texture, no warp, no resample. Every land pixel carries its H3
res 7 label (one cell per pixel, a relabel) and the label's res 6 parent. Nothing
hexagonal is drawn. H3 does four jobs here:

  1. PICK: a click becomes latLngToCell(res 7) and the label index names the pixel. If
     the click lands in one of the ~40% of res 7 cells that hold no pixel centre, the
     grid's own Lambert inverse snaps it to the pixel, and the readout says which hit.
  2. COUNTY: a pixel's county is the res 6 polyfill join (the land mask, too).
  3. DOMES: the sustained-heat levels are applied per pixel; membership is then decided
     per pixel (no H3) or per res 6 parent cell (any / majority / all of its pixels),
     and handed back to the pixels. The boundaries are drawn on PIXEL EDGES at each
     contour level, every frame, following the sliders. The difference between the
     rules is what H3 adds and removes: single-pixel speckle goes, the edge coarsens
     to ~4-pixel granularity.
  4. THE DOME TABLE (DuckDB): the res 6 parents of the member pixels are dissolved with
     h3_cells_to_multi_polygon_wkb, dumped into blobs with an Albers area and a
     centroid, and tracked hour by hour, next to the plain pixel count x 9 km2 so the
     table shows what the cell rule adds.

The chassis is x-sql-marimo/xsql-hrrr-heat-domes.py (constants, DuckDB, the PMTiles
county reader, the disk MirrorStore, the window loader, the accumulator, the dome
table). The fold is gone: there is no GROUP BY when the unit is the pixel, so the
read is block-wise straight off the Zarr (only the 45x45 store blocks that touch land
inside BOX, in threads, through the mirror) into an (hours x land pixels) matrix.

Sustained heat is the same smoothing as before, stated plainly: L[t] = a L[t-1] +
(1 - a) max(0, HI[t] - thr), a = 2^(-1/half_life), rain flushes and wind vents it when
those fields are read. It is not a published index; the sliders are the assumptions.

Run: uv run marimo edit hrrr-heat-domes.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import json
    import math
    import struct

    import anywidget
    import duckdb
    import marimo as mo
    import numpy as np
    import obstore
    import pyarrow as pa
    import pyarrow.parquet as pq
    import traitlets
    import xarray as xr
    import zarr
    from h3ronpy import change_resolution
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store

    return (
        S3Store,
        anywidget,
        asyncio,
        change_resolution,
        coordinates_to_cells,
        duckdb,
        gzip,
        json,
        math,
        mo,
        np,
        obstore,
        pa,
        pq,
        struct,
        traitlets,
        xr,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HRRR heat domes on the native grid

    Hourly heat index for the lower 48, drawn as the HRRR raster itself: one quad per
    3 km pixel on the store's own Lambert grid, no warp, no resample, no hexagons.
    Underneath, every land pixel carries an H3 label, and that label is what does the
    work: the click, the county, the dome membership, the dome table. Switch to
    **sustained heat** (an exponentially weighted mean of the heat index above a
    threshold, decaying with a half-life you set) and the **boundaries** of the domes
    are drawn on pixel edges at fixed levels, moving with the film. The **dome rule**
    select is the experiment: decide membership per pixel, or per res 6 cell by any,
    majority or all of its ~4 pixels, and watch what the cell rule adds and removes.

    **Where the numbers come from.** [dynamical.org](https://dynamical.org/)'s Zarr
    build of NOAA's HRRR analysis (3 km, hourly, CONUS, CC-BY 4.0), read anonymously
    from `s3://dynamical-noaa-hrrr`: 2 m temperature and relative humidity, optionally
    precipitation rate and 10 m wind. Counties are Overture Maps divisions from
    Overture's PMTiles; they are the land mask and the name a clicked pixel reports.

    **What it costs.** The store is chunked for time series (each 45 × 45 pixel column
    is 2,160 hours deep), so a window fetches every filled hour of the chunk it falls
    in, whatever its length. Chunks already on the disk mirror read in seconds; a new
    one is minutes on a home link. The default window (the eastern dome, Jun 29 to
    Jul 5, 2026) is mirrored. A week over CONUS land is ~150 MB per field to the
    browser; `BOX` in the constants cell can cut that down to a region.
    """)
    return


@app.cell
def _():
    # ------------------------------------------------------------------ the weather
    ANALYSIS_BUCKET = "dynamical-noaa-hrrr"
    ANALYSIS_PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
    # Heat index needs temperature_2m and relative_humidity_2m (always read). Rain is
    # the accumulator's flush, wind its vent; each is another whole 90-day chunk layer
    # per block from the wire (seconds from the mirror).
    READ_RAIN = True
    READ_WIND = True
    # Opening window: an int is the last DAYS UTC days; a ("YYYY-MM-DD", "YYYY-MM-DD")
    # tuple is a fixed window. Summer 2026's domes: East Jun 28-Jul 5 (mirrored on
    # this machine), West Jul 6-12, Plains Jul 23-29.
    DAYS = ("2026-06-29", "2026-07-05")
    HOURLY_MAX_DAYS = 14
    # The pixels: a lon/lat box (W, S, E, N), or None for all of CONUS. The mesh, the
    # labels and the frames are built for the box's bounding rectangle of the grid.
    BOX = None  # e.g. (-95.0, 29.0, -69.0, 46.0) for the eastern dome's region
    # ------------------------------------------------------------------ the labels
    # RES_L: the LABEL resolution, one cell per pixel (res 7: 5.2 km2 against the 9 km2
    # pixel; measured 1,905,141 labels for 1,905,141 pixels). RES_T: the TRAVERSAL
    # resolution, cellToParent of the label (res 6: ~4 pixels per cell), where domes
    # are decided and counties joined.
    RES_L, RES_T = 7, 6
    CHUNK_CACHE_GB = 2  # icechunk chunk-bytes cache; the disk mirror does the real holding
    READ_THREADS = 8
    # Boundary levels: degC of sustained excess; the browser outlines the set of member
    # pixels at or above each, every frame, and the dome table dissolves the same sets.
    CONTOURS = [1.0, 3.0, 5.0, 10.0]
    # Dome membership for the kernel's table: "pixel" (no H3), or per res 6 cell by
    # "any" / "majority" / "all" of its pixels. The HUD's select is the live version.
    DOME_RULE = "majority"
    DOME_MIN_KM2 = 500.0  # blobs smaller than this are speckle (a res 6 cell is ~36 km2)
    PIXEL_KM2 = 9.0  # nominal 3 km x 3 km; the true area varies with the map scale factor

    # ------------------------------------------------------------------ the land mask
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    COUNTY_Z = 8
    CONUS_BOX = (-124.8, 24.4, -66.9, 49.5)  # the county fetch box (names the cache file)
    NOT_CONUS = {"AK", "HI"}
    import tempfile as _tempfile

    # Shared with x-sql-marimo: the counties parquet and the shard mirror live here.
    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"
    MIRROR_DIR = CACHE_DIR + "/hrrr-mirror/" + ANALYSIS_PREFIX.split("/")[-1]

    # ------------------------------------------------------------------ the film
    # Heat index: diverging blue <-> yellow/orange (protan-safe: no red leg), pale at
    # the pivot (the film's median). Sustained heat: one-signed, inferno's stops,
    # scaled in the browser to the p98 of what it just computed. Boundaries: gold.
    INDEX_STOPS = ["#08306b", "#2f79b5", "#9ecae1", "#f2f0e6", "#fee391", "#fdb034", "#d94801"]
    LOAD_STOPS = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]
    PIVOT = None  # degC heat index, or None for the film median
    SPAN = None  # degC either side, or None for the p2/p98 rule
    THRESHOLD = 27.0
    HALF_LIFE = 12.0
    RAIN_FLUSH = 0.5
    WIND_VENT = 0.3
    FPS = 8
    MAP_HEIGHT = 640
    # Carto's basemap tiles (both the no-labels ground and the labels layer) carry an
    # "API KEY REQUIRED" watermark since 2026-08-30, and OpenStreetMap's servers refuse
    # apps (403). Esri's Dark Gray Canvas is keyless for light use (attribution: Esri,
    # HERE, Garmin, OpenStreetMap contributors). BASE_DARK desaturates + multiplies by
    # BASE_TINT in the shader, for a light basemap; the canvas is dark already.
    BASE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    BASE_DARK = False
    BASE_TINT = [70, 76, 86]
    LABEL_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
    return (
        ANALYSIS_BUCKET,
        ANALYSIS_PREFIX,
        BASE_DARK,
        BASE_TILES,
        BASE_TINT,
        BOX,
        CACHE_DIR,
        CHUNK_CACHE_GB,
        CONTOURS,
        CONUS_BOX,
        COUNTY_Z,
        DAYS,
        DOME_MIN_KM2,
        DOME_RULE,
        FPS,
        HALF_LIFE,
        HOURLY_MAX_DAYS,
        INDEX_STOPS,
        LABEL_TILES,
        LOAD_STOPS,
        MAP_HEIGHT,
        MIRROR_DIR,
        NOT_CONUS,
        OVERTURE_RELEASE,
        PIVOT,
        PIXEL_KM2,
        PM_BUCKET,
        PM_PATH,
        RAIN_FLUSH,
        READ_RAIN,
        READ_THREADS,
        READ_WIND,
        RES_L,
        RES_T,
        SPAN,
        THRESHOLD,
        WIND_VENT,
    )


@app.cell
def _(duckdb):
    # DuckDB does the geometry: county seam dissolve, polyfill, the dome dissolve.
    con = duckdb.connect()
    con.sql("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class RasterFilm(anywidget.AnyWidget):
        """deck.gl SimpleMeshLayer over the native grid; H3 labels for picking and domes.

        Kernel -> browser, once (the geometry): `corners` (float32 (ny+1)(nx+1)x2 web
        mercator world coords, 512 units per world), `lidx` (uint32 flat grid index of
        each land pixel, N, row-major), `labels` (uint64 res 7 cell per land pixel),
        `pidx` (uint32 res 6 parent row per land pixel), `cty` (uint16 county per land
        pixel, 65535 none), `names` (JSON), `geom` (JSON: ny, nx, n, p, the Lambert
        edge origin x0/y0 and signed dx/dy for the snap, the home view). Per window:
        `frames` (uint8 F x N, frame-major: heat index in 0.5 degC steps offset -40,
        255 = no data), `wx` (uint8 F x N or empty: wind m/s in the high nibble, rain
        in 0.5 mm/h steps in the low), `config` (JSON: labels, ramps, accumulator
        defaults, contours, fps, height, title, subtitle, `win`).
        Browser -> kernel: `window` only ({"d0","d1"} JSON from the load button).

        The ACCUMULATOR runs here over the frame matrix whenever a slider moves. The
        MESH is (ny+1)(nx+1) shared vertices, two triangles per pixel, built once; a
        frame repaints one RGBA canvas (the texture) from the field through a LUT.
        The BOUNDARIES: per pixel a level code (how many contours it clears), the rule
        applied per res 6 parent, then every pixel edge whose two sides differ in
        level is a boundary segment for the levels in between; one PathLayer per level.
        PICK is h3-js latLngToCell(res 7) into the label Map, then the grid's own
        Lambert inverse when the cell holds no pixel centre.

        esm.sh pins: every deck package at 9.3.10 with `?deps` so all resolve to one
        core; h3-js 4.5.0.
        """

        _esm = r"""
        import {Deck, COORDINATE_SYSTEM} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer, LineLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {SimpleMeshLayer} from "https://esm.sh/@deck.gl/mesh-layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .rf { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .rf * { box-sizing: border-box; }
          .rf .rf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .rf .rf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .rf .rf-map:fullscreen { height: 100vh !important; width: 100vw; }
          .rf .rf-hud { position: absolute; z-index: 5; }
          .rf .rf-hud.rf-tl { top: .6rem; left: .6rem; width: 23rem; max-width: calc(100% - 1.2rem); }
          .rf .rf-hud.rf-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .rf .rf-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .rf .rf-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }
          .rf .rf-ttl { font-weight: 600; }
          .rf .rf-sub { color: var(--dim); display: block; margin-top: .1rem; }
          .rf .rf-fields { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
          .rf .rf-fields button.rf-b { flex: 0 0 auto; font-size: 11px; padding: .12rem .4rem; min-width: 0; }
          .rf .rf-fields button.rf-on { background: #3a3f2a; border-color: var(--accent); color: #fff; }
          .rf .rf-fields select { font-size: 11px; padding: .12rem .3rem; }
          .rf .rf-legend { display: flex; align-items: center; gap: .45rem; margin-top: .45rem; }
          .rf .rf-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .rf .rf-row { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin-top: .4rem; }
          .rf .rf-row .rf-v { font-size: 16px; }
          .rf .rf-row .rf-k { color: var(--dim); }
          .rf .rf-cell { margin-top: .35rem; display: none; }
          .rf.rf-picked .rf-cell { display: block; }
          .rf .rf-how { color: var(--dim); font-size: 11px; }
          .rf .rf-chart { display: block; width: 100%; height: 96px; margin-top: .3rem; cursor: crosshair; }
          .rf.rf-collapsed .rf-body, .rf.rf-collapsed .rf-sub { display: none; }
          .rf .rf-toggle, .rf .rf-clear { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .rf .rf-toggle:hover, .rf .rf-clear:hover { color: var(--ink); }
          .rf .rf-params { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .rf .rf-p { display: grid; grid-template-columns: 6.2rem 1fr 3.4rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .rf .rf-p label { color: var(--dim); }
          .rf .rf-p.rf-off { display: none; }
          .rf .rf-transport { display: flex; align-items: center; gap: .55rem; }
          .rf .rf-stamp { font-size: 15px; min-width: 11.5rem; }
          .rf .rf-stamp small { display: block; font-size: 10px; color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
          .rf .rf-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .rf .rf-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .rf .rf-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .rf input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .rf button.rf-b, .rf select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .rf button.rf-b:hover, .rf select:hover { background: #2b323b; }
          .rf button:focus-visible, .rf select:focus-visible, .rf input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .rf .rf-dim { color: var(--dim); }
          .rf .rf-win { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .4rem; margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .rf .rf-win input[type=date] { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .15rem .3rem; font: inherit; color-scheme: dark; min-width: 0; }
          .rf .rf-win .rf-note { flex-basis: 100%; }
          .rf .rf-win .rf-note.rf-bad { color: var(--accent); }
          .rf .rf-win button.rf-load:disabled { opacity: .55; cursor: default; }
          .rf .rf-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; text-align: right; white-space: pre; }
          @media (max-width: 720px) { .rf .rf-stamp { min-width: 0; } .rf .rf-hud.rf-tl { width: calc(100% - 1.2rem); } }
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
        const HI_OF = q => q / 2 - 40;  // uint8 -> degC

        // HRRR's Lambert conformal conic (sphere R=6371229, tangent at 38.5N / 97.5W):
        // the forward, for the SNAP: lon/lat -> grid metres without any H3. The raster's
        // own inverse; checked against pyproj to the metre (2026-08-30).
        const LCC = (() => {
          const R = 6371229, lat0 = 38.5 * Math.PI / 180, lon0 = -97.5 * Math.PI / 180;
          const n = Math.sin(lat0), F = Math.cos(lat0) * Math.pow(Math.tan(Math.PI / 4 + lat0 / 2), n) / n;
          const rho0 = R * F / Math.pow(Math.tan(Math.PI / 4 + lat0 / 2), n);
          return (lon, lat) => {
            const la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
            const rho = R * F / Math.pow(Math.tan(Math.PI / 4 + la / 2), n), th = n * (lo - lon0);
            return [rho * Math.sin(th), rho0 - rho * Math.cos(th)];
          };
        })();

        function render({model, el}) {
          el.innerHTML = "";
          const root = document.createElement("div"); root.className = "rf";
          root.innerHTML = `<style>${CSS}</style>
            <div class="rf-map">
              <div class="rf-hud rf-tl"><div class="rf-card rf-panel">
                <div class="rf-head"><span><span class="rf-ttl"></span><span class="rf-sub"></span></span><button class="rf-toggle" title="hide / show (H)">hide</button></div>
                <div class="rf-fields">
                  <button class="rf-b rf-fi rf-on" data-field="index" title="NWS heat index this hour (I)">heat index</button>
                  <button class="rf-b rf-fi" data-field="load" title="sustained heat: exponentially weighted mean of heat index above the threshold (L)">sustained heat</button>
                  <button class="rf-b rf-bnd rf-on" title="boundaries of the sustained heat at the contour levels, on pixel edges (B)">boundaries</button>
                  <select class="rf-rule" title="dome membership: per pixel, or decided per res 6 cell and handed back to its pixels">
                    <option value="pixel">rule: per pixel (no H3)</option>
                    <option value="any">rule: res 6 cell, any pixel</option>
                    <option value="majority" selected>rule: res 6 cell, majority</option>
                    <option value="all">rule: res 6 cell, all pixels</option>
                  </select>
                </div>
                <div class="rf-dim rf-bndnote"></div>
                <div class="rf-legend"><span class="rf-num rf-lo"></span><div class="rf-grad"></div><span class="rf-num rf-hi"></span></div>
                <div class="rf-body">
                  <div class="rf-row"><span class="rf-k rf-meank">mean</span><span class="rf-num rf-v rf-mean">–</span></div>
                  <div class="rf-cell">
                    <div class="rf-row"><span class="rf-k rf-cname">–</span><span><span class="rf-num rf-v rf-cval">–</span> <button class="rf-clear" title="clear">×</button></span></div>
                    <div class="rf-how rf-num"></div>
                    <canvas class="rf-chart" height="96"></canvas>
                  </div>
                  <div class="rf-params">
                    <div class="rf-p"><label>half-life</label><input type="range" class="rf-half" min="1" max="72" step="1"><span class="rf-num rf-halfv"></span></div>
                    <div class="rf-p"><label>threshold</label><input type="range" class="rf-thr" min="15" max="40" step="0.5"><span class="rf-num rf-thrv"></span></div>
                    <div class="rf-p rf-p-rain"><label>rain flush</label><input type="range" class="rf-rain" min="0" max="1" step="0.05"><span class="rf-num rf-rainv"></span></div>
                    <div class="rf-p rf-p-wind"><label>wind vent</label><input type="range" class="rf-wind" min="0" max="1" step="0.05"><span class="rf-num rf-windv"></span></div>
                    <div class="rf-p"><label>opacity</label><input type="range" class="rf-opac" min="0" max="1" step="0.05" value="1"><span class="rf-num rf-opacv">1.00</span></div>
                    <div class="rf-dim rf-pnote"></div>
                  </div>
                  <div class="rf-win">
                    <input type="date" class="rf-d0" title="first UTC day, inclusive" aria-label="window start (UTC day)"><span class="rf-dim">to</span><input type="date" class="rf-d1" title="last UTC day, inclusive" aria-label="window end (UTC day)">
                    <button class="rf-b rf-load" title="read this window (the kernel refetches; the estimate is beside the dates)">load</button>
                    <span class="rf-dim rf-note"></span>
                  </div>
                  <div class="rf-dim rf-hint">click a pixel for its value and line · space plays · ← → step · I / L field · B boundaries · H hide · F fullscreen</div>
                </div>
              </div></div>
              <span class="rf-ruler rf-num"></span>
              <div class="rf-hud rf-bl"><div class="rf-card rf-transport">
                <button class="rf-b rf-prev" title="step back (←)">‹</button>
                <button class="rf-b rf-play" title="play / pause (space)">▶</button>
                <button class="rf-b rf-next" title="step forward (→)">›</button>
                <div class="rf-track"><div class="rf-ticks"></div><input class="rf-frame" type="range" min="0" max="0" value="0" step="1" aria-label="frame"></div>
                <div class="rf-stamp rf-num"><small class="rf-stampk">hour (UTC)</small><span class="rf-stampv">–</span></div>
                <select class="rf-fps" title="frames per second"><option>2</option><option>4</option><option>6</option><option>8</option><option>12</option><option>24</option></select>
                <button class="rf-b rf-full" title="fullscreen (F)">⛶</button>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".rf-map"), playBtn = q(".rf-play"), slider = q(".rf-frame"), ticks = q(".rf-ticks"),
                stampV = q(".rf-stampv"), fpsSel = q(".rf-fps"), grad = q(".rf-grad"),
                loEl = q(".rf-lo"), hiEl = q(".rf-hi"), chart = q(".rf-chart"), ruler = q(".rf-ruler"),
                ttl = q(".rf-ttl"), sub = q(".rf-sub"), meanEl = q(".rf-mean"), meanK = q(".rf-meank"),
                cname = q(".rf-cname"), cval = q(".rf-cval"), howEl = q(".rf-how"),
                d0In = q(".rf-d0"), d1In = q(".rf-d1"), loadBtn = q(".rf-load"), noteEl = q(".rf-note"),
                halfIn = q(".rf-half"), thrIn = q(".rf-thr"), rainIn = q(".rf-rain"), windIn = q(".rf-wind"),
                halfV = q(".rf-halfv"), thrV = q(".rf-thrv"), rainV = q(".rf-rainv"), windV = q(".rf-windv"), pnote = q(".rf-pnote"),
                ruleSel = q(".rf-rule"), bndBtn = q(".rf-bnd"), bndNote = q(".rf-bndnote"), toggle = q(".rf-toggle"),
                opacIn = q(".rf-opac"), opacV = q(".rf-opacv");

          let geo = {}, cfg = {}, ny = 0, nx = 0, N = 0, F = 0, P = 0;
          let corners = null, lidx = null, lab = null, pidx = null, cty = null, names = [], pcount = null;
          let frames = null, wx = null, load = null, loadHi = 1, meansI = null, meansL = null;
          let mesh = null, tex = [null, null], texData = [null, null], texK = 0, lutI = null, lutL = null;
          let labelIndex = new Map(), pixRow = null;   // res 7 id -> land row; flat grid index -> land row
          let frame = 0, field = "index", playing = false, timer = null, deck = null, selected = -1, gen = 0;
          let showBounds = true, buildMs = 0, domeInfo = "", paintMs = 0, renderMs = 0, renderT0 = 0;
          const BND_RGB = [247, 209, 61];  // gold, inferno's #f7d13d stop; alpha and width step by level
          let HOME = {longitude: -84, latitude: 37.5, zoom: 4.6, minZoom: 2, maxZoom: 12};

          const fmtC = v => Number.isFinite(v) ? v.toFixed(1) + "°C" : "no data";
          const hiAt = (f, i) => { const qv = frames[f * N + i]; return qv === 255 ? NaN : HI_OF(qv); };
          const loadAt = (f, i) => { const qv = load[f * N + i]; return qv === 255 ? NaN : qv / 10; };
          const valAt = (f, i) => field === "load" ? loadAt(f, i) : hiAt(f, i);
          const contours = () => (cfg.contours && cfg.contours.length) ? cfg.contours : [1, 3, 5, 10];
          const params = () => ({
            half: parseFloat(halfIn.value) || 12, thr: parseFloat(thrIn.value) || 27,
            rain: cfg.has_rain ? (parseFloat(rainIn.value) || 0) : 0, wind: cfg.has_wind ? (parseFloat(windIn.value) || 0) : 0,
          });

          // THE GEOMETRY, once: the label index, the pixel row index, the mesh.
          function loadStatic() {
            try { geo = JSON.parse(model.get("geom") || "{}"); } catch (e) { geo = {}; }
            ny = geo.ny | 0; nx = geo.nx | 0; N = geo.n | 0; P = geo.p | 0;
            corners = typed(bytesOf(model.get("corners")), Float32Array);
            lidx = typed(bytesOf(model.get("lidx")), Uint32Array);
            lab = typed(bytesOf(model.get("labels")), BigUint64Array);
            pidx = typed(bytesOf(model.get("pidx")), Uint32Array);
            cty = typed(bytesOf(model.get("cty")), Uint16Array);
            try { names = JSON.parse(model.get("names") || "[]"); } catch (e) { names = []; }
            if (geo.home) HOME = Object.assign({}, HOME, geo.home);
            if (!corners || !lidx || !lab || !pidx || !N) return;
            const t0 = performance.now();
            labelIndex = new Map();
            for (let i = 0; i < N; i++) labelIndex.set(lab[i].toString(16), i);
            pixRow = new Int32Array(ny * nx).fill(-1);
            for (let i = 0; i < N; i++) pixRow[lidx[i]] = i;
            pcount = new Uint16Array(P);
            for (let i = 0; i < N; i++) pcount[pidx[i]]++;
            // (ny+1)(nx+1) shared corner vertices in world coords, two triangles per pixel,
            // tex coords = the pixel grid. Built once; the texture is what changes.
            const V = (ny + 1) * (nx + 1);
            const pos = new Float32Array(V * 3), uv = new Float32Array(V * 2);
            for (let r = 0; r <= ny; r++) for (let c = 0; c <= nx; c++) {
              const v = r * (nx + 1) + c;
              pos[3 * v] = corners[2 * v]; pos[3 * v + 1] = corners[2 * v + 1]; pos[3 * v + 2] = 0;
              uv[2 * v] = c / nx; uv[2 * v + 1] = r / ny;
            }
            const idx = new Uint32Array(ny * nx * 6); let k = 0;
            for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
              const a = r * (nx + 1) + c, b = a + 1, d = a + nx + 1, e = d + 1;
              idx[k++] = a; idx[k++] = d; idx[k++] = b; idx[k++] = b; idx[k++] = d; idx[k++] = e;
            }
            mesh = {attributes: {positions: {value: pos, size: 3}, texCoords: {value: uv, size: 2}}, indices: {value: idx, size: 1}};
            // two canvases, alternated: SimpleMeshLayer re-uploads the texture only when the
            // prop is a different object (updateTriggers do not reach it), so one canvas
            // painted in place would leave the first frame on the GPU for ever.
            for (let k = 0; k < 2; k++) {
              tex[k] = document.createElement("canvas"); tex[k].width = nx; tex[k].height = ny;
              texData[k] = tex[k].getContext("2d").createImageData(nx, ny);
            }
            buildMs = Math.round(performance.now() - t0);
          }

          // THE FRAMES, per window.
          function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px";
            lutI = buildLut(cfg.index_stops || ["#08306b", "#f2f0e6", "#d94801"]);
            lutL = buildLut(cfg.load_stops || ["#000004", "#fcffa4"]);
            ttl.textContent = cfg.title || ""; sub.textContent = cfg.subtitle || "";
            syncWindow();
            const u8 = bytesOf(model.get("frames"));
            if (!u8 || !u8.length || !N) { frames = null; load = null; F = 0; legend(); return; }
            frames = new Uint8Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            F = Math.floor(frames.length / N);
            const w8 = bytesOf(model.get("wx"));
            wx = w8 && w8.length === frames.length ? new Uint8Array(w8.buffer.slice(w8.byteOffset, w8.byteOffset + w8.byteLength)) : new Uint8Array(frames.length);
            meansI = new Float32Array(F);
            for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let i = 0; i < N; i++) { const qv = frames[f * N + i]; if (qv !== 255) { s += HI_OF(qv); n++; } } meansI[f] = n ? s / n : NaN; }
            slider.max = String(Math.max(0, F - 1));
            if (frame >= F) frame = 0;
            fpsSel.value = String(cfg.fps || 8);
            if (!halfIn.dataset.set) {  // seed the sliders once from the kernel's defaults
              halfIn.value = cfg.half_life ?? 12; thrIn.value = cfg.threshold ?? 27;
              rainIn.value = cfg.rain_flush ?? 0.5; windIn.value = cfg.wind_vent ?? 0.3; halfIn.dataset.set = "1";
            }
            q(".rf-p-rain").classList.toggle("rf-off", !cfg.has_rain);
            q(".rf-p-wind").classList.toggle("rf-off", !cfg.has_wind);
            paramLabels();
            computeLoad();
            legend(); bndLabel();
            const labels = cfg.labels || [];
            let html = "";
            for (let f = 1; f < labels.length; f++) {
              const d0 = labels[f - 1].slice(0, 10), d1 = labels[f].slice(0, 10);
              if (d0 !== d1) html += `<i style="left:${(f / (F - 1) * 100).toFixed(2)}%"></i>`;
            }
            ticks.innerHTML = F > 1 ? html : "";
          }

          // THE ACCUMULATOR, over the whole film, into uint8 (0.1 degC steps, 255 = no data).
          function computeLoad() {
            if (!frames) return;
            const {half, thr, rain, wind} = params();
            const a = Math.pow(2, -1 / half), b = 1 - a;
            load = load && load.length === F * N ? load : new Uint8Array(F * N);
            const prev = new Float32Array(N), hist = new Uint32Array(256);
            meansL = new Float32Array(F);
            for (let f = 0; f < F; f++) {
              const base = f * N; let s = 0, n = 0;
              for (let i = 0; i < N; i++) {
                const k = base + i, qv = frames[k];
                if (qv === 255) { load[k] = 255; continue; }
                let ex = HI_OF(qv) - thr; if (ex < 0) ex = 0;
                if (wind) { const ws = wx[k] >> 4; ex *= Math.max(0, 1 - wind * ws / 10); }
                let L = a * prev[i] + b * ex;
                if (rain) { const mm = (wx[k] & 15) / 2; if (mm > 0) L *= 1 - rain * Math.min(1, mm / 2.5); }
                prev[i] = L;
                let lq = Math.round(L * 10); if (lq > 254) lq = 254;
                load[k] = lq; hist[lq]++; s += L; n++;
              }
              meansL[f] = n ? s / n : NaN;
            }
            let tot = 0; for (let i = 1; i < 255; i++) tot += hist[i];
            let acc = 0, top = 10;
            for (let i = 1; i < 255; i++) { acc += hist[i]; if (acc >= tot * 0.98) { top = i; break; } }
            loadHi = Math.max(1, top / 10);
            gen++;
          }
          function paramLabels() {
            const p = params();
            halfV.textContent = p.half + " h"; thrV.textContent = p.thr.toFixed(1) + "°C";
            rainV.textContent = p.rain.toFixed(2); windV.textContent = p.wind.toFixed(2);
            pnote.textContent = `sustained heat = exponentially weighted mean of heat index above ${p.thr.toFixed(1)}°C, half-life ${p.half} h (a smoothing, not a published index)` +
              (cfg.has_rain ? "" : " · rain not read") + (cfg.has_wind ? "" : " · wind not read");
          }
          let ptimer = null;
          const onParam = () => { paramLabels(); if (ptimer) clearTimeout(ptimer); ptimer = setTimeout(() => { computeLoad(); legend(); update(); }, 120); };
          halfIn.oninput = thrIn.oninput = rainIn.oninput = windIn.oninput = onParam;
          ruleSel.onchange = () => { bndLabel(); update(); };
          // raster opacity: a layer prop, no repaint
          opacIn.oninput = () => { opacV.textContent = parseFloat(opacIn.value).toFixed(2); update(); };

          // THE TEXTURE: one RGBA image of the grid, land pixels from the field at full
          // opacity, no-data land dim grey, everything else alpha 0.
          function paintTexture() {
            const tp = performance.now();
            texK ^= 1;
            const cv = tex[texK], td = texData[texK], d = td.data; d.fill(0);
            const src = field === "load" ? load : frames, lut = field === "load" ? lutL : lutI, base = frame * N;
            const lo = cfg.lo ?? 0, span = (cfg.hi ?? 1) - lo || 1;
            for (let i = 0; i < N; i++) {
              const o = lidx[i] * 4, qv = src ? src[base + i] : 255;
              if (qv === 255) { d[o] = 40; d[o + 1] = 44; d[o + 2] = 50; d[o + 3] = 60; continue; }
              let t;
              if (field === "load") { t = (qv / 10) / loadHi; if (t > 1) t = 1; }
              else { t = (HI_OF(qv) - lo) / span; t = t < 0 ? 0 : t > 1 ? 1 : t; }
              const j = Math.round(t * 255) * 3;
              d[o] = lut[j]; d[o + 1] = lut[j + 1]; d[o + 2] = lut[j + 2]; d[o + 3] = 255;
            }
            cv.getContext("2d").putImageData(td, 0, 0);
            paintMs = Math.round(performance.now() - tp);
            return cv;
          }

          // THE DOMES. Per pixel: how many contour levels its sustained heat clears (a
          // code 0..L; the sets are nested). Then the rule: per pixel as is, or per res 6
          // parent (count the pixels at each level, decide by any / majority / all, hand
          // the cell's code back to its pixels). H3 as the transformation, in ten lines.
          function domeLevels() {
            const codes = contours().map(v => Math.round(v * 10)), L = codes.length, base = frame * N, rule = ruleSel.value;
            const lv = new Uint8Array(N);
            for (let i = 0; i < N; i++) { const v = load[base + i]; if (v === 255) continue; let k = 0; while (k < L && v >= codes[k]) k++; lv[i] = k; }
            if (rule === "pixel") return lv;
            const cnt = new Uint32Array(L * P);
            for (let i = 0; i < N; i++) { const k = lv[i], p = pidx[i]; for (let m = 0; m < k; m++) cnt[m * P + p]++; }
            const cellLv = new Uint8Array(P);
            for (let p = 0; p < P; p++) {
              const n = pcount[p]; let k = 0;
              while (k < L) { const c = cnt[k * P + p]; if (!(rule === "any" ? c > 0 : rule === "all" ? c === n : c * 2 > n)) break; k++; }
              cellLv[p] = k;
            }
            const out = new Uint8Array(N); for (let i = 0; i < N; i++) out[i] = cellLv[pidx[i]];
            return out;
          }
          // PIXEL-EDGE BOUNDARIES: an edge between two grid cells whose codes differ is a
          // boundary segment for every level between them. One pass over the grid.
          function boundaryLayers() {
            domeInfo = "";
            if (!showBounds || !load || !F) return [];
            const t0 = performance.now();
            const levels = contours(), L = levels.length, lv = domeLevels();
            const g = new Uint8Array(ny * nx), counts = new Uint32Array(L + 1);
            for (let i = 0; i < N; i++) { g[lidx[i]] = lv[i]; counts[lv[i]]++; }
            const pos = levels.map(() => []), W = nx + 1;
            const seg = (b, a, v0, v1) => { for (let k = b; k < a; k++) pos[k].push(corners[2 * v0], corners[2 * v0 + 1], corners[2 * v1], corners[2 * v1 + 1]); };
            for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
              const i = r * nx + c, a = g[i]; if (!a) continue;
              const v = r * W + c;
              const t = r === 0 ? 0 : g[i - nx], bt = r === ny - 1 ? 0 : g[i + nx], l = c === 0 ? 0 : g[i - 1], rt = c === nx - 1 ? 0 : g[i + 1];
              if (t < a) seg(t, a, v, v + 1);
              if (bt < a) seg(bt, a, v + W, v + W + 1);
              if (l < a) seg(l, a, v, v + W);
              if (rt < a) seg(rt, a, v + 1, v + W + 1);
            }
            // ONE LineLayer for every level: instanced segments straight from typed
            // arrays with per-edge colour and width, no CPU tesselation (a PathLayer
            // cost ~1.2 s per frame at 54k edges; four LineLayers ~0.6 s in software GL).
            const info = []; let E = 0;
            for (let k = 0; k < L; k++) {
              let mem = 0; for (let m = k + 1; m <= L; m++) mem += counts[m];
              info.push(`≥${levels[k]}°C ${mem.toLocaleString()} px`); E += pos[k].length / 4;
            }
            const src = new Float32Array(E * 2), dst = new Float32Array(E * 2), col = new Uint8Array(E * 4), wid = new Float32Array(E);
            let e = 0;
            for (let k = 0; k < L; k++) {
              const xy = pos[k], a = 150 + Math.min(105, k * 35), w = 1 + k * 0.6;
              for (let j = 0; j < xy.length; j += 4, e++) {
                src[2 * e] = xy[j]; src[2 * e + 1] = xy[j + 1]; dst[2 * e] = xy[j + 2]; dst[2 * e + 1] = xy[j + 3];
                col[4 * e] = BND_RGB[0]; col[4 * e + 1] = BND_RGB[1]; col[4 * e + 2] = BND_RGB[2]; col[4 * e + 3] = a; wid[e] = w;
              }
            }
            domeInfo = `domes (${ruleSel.value}): ${info.join(" · ")} · ${E.toLocaleString()} edges · ${Math.round(performance.now() - t0)} ms`;
            if (!E) return [];
            return [new LineLayer({
              id: "bounds",
              data: {length: E, attributes: {getSourcePosition: {value: src, size: 2}, getTargetPosition: {value: dst, size: 2}, getColor: {value: col, size: 4}, getWidth: {value: wid, size: 1}}},
              coordinateSystem: COORDINATE_SYSTEM.CARTESIAN, widthUnits: "pixels", widthMinPixels: 1, pickable: false,
            })];
          }
          function bndLabel() {
            const r = ruleSel.value, how = r === "pixel" ? "per pixel" : `per res 6 cell (${r})`;
            bndNote.textContent = showBounds ? `boundaries on pixel edges: sustained heat ≥ ${contours().map(v => v + "°C").join(" / ")} (thin to thick), membership ${how}` : "";
          }
          bndBtn.onclick = () => { showBounds = !showBounds; bndBtn.classList.toggle("rf-on", showBounds); bndLabel(); update(); };

          const tiles = (id, url, opacity, dark) => new TileLayer({
            id, data: url, tileSize: 256, minZoom: 0, maxZoom: 19, opacity, pickable: false,
            renderSubLayers: p => { const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north],
              desaturate: dark ? 1 : 0, tintColor: dark ? (cfg.base_tint || [70, 76, 86]) : [255, 255, 255]}); },
          });
          function layers() {
            const out = [];
            if (cfg.base_tiles !== "") out.push(tiles("base", cfg.base_tiles || "https://tile.openstreetmap.org/{z}/{x}/{y}.png", 1.0, !!cfg.base_dark));
            if (mesh) {
              out.push(new SimpleMeshLayer({
                id: "raster", data: [0], mesh, texture: paintTexture(),
                coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                getPosition: d => [0, 0, 0], getColor: [255, 255, 255, 255],
                material: false, pickable: false, opacity: parseFloat(opacIn.value),
                textureParameters: {minFilter: "nearest", magFilter: "nearest"},
                parameters: {depthTest: false},
              }));
              out.push(...boundaryLayers());
              if (selected >= 0) {
                const g = lidx[selected], r = Math.floor(g / nx), c = g % nx, a = r * (nx + 1) + c, W = nx + 1;
                const ring = [a, a + 1, a + W + 1, a + W, a].map(v => [corners[2 * v], corners[2 * v + 1]]);
                out.push(new PathLayer({id: "picked", data: [ring], getPath: x => x, coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                  getColor: [255, 255, 255, 255], getWidth: 2, widthUnits: "pixels", pickable: false}));
              }
            }
            if (cfg.label_tiles) out.push(tiles("labels", cfg.label_tiles, 0.6));
            return out;
          }
          function legend() {
            const lut = field === "load" ? lutL : lutI;
            if (!lut) return;
            const stops = [];
            for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            if (field === "load") { loEl.textContent = "0"; hiEl.textContent = "+" + loadHi.toFixed(1) + "°C"; meanK.textContent = "mean sustained heat, land px"; }
            else { loEl.textContent = fmtC(cfg.lo); hiEl.textContent = fmtC(cfg.hi); meanK.textContent = "mean heat index, land px"; }
            root.querySelectorAll(".rf-fi").forEach(b => b.classList.toggle("rf-on", b.dataset.field === field));
          }
          const setField = f => { field = f; legend(); update(); };
          root.querySelectorAll(".rf-fi").forEach(b => { b.onclick = () => setField(b.dataset.field); });

          function stats() {
            const m = field === "load" ? meansL : meansI;
            meanEl.textContent = m && F ? (field === "load" ? "+" : "") + fmtC(m[frame]) : "–";
            if (selected >= 0 && F) cval.textContent = (field === "load" ? "+" : "") + fmtC(valAt(frame, selected));
          }
          function drawChart() {
            if (selected < 0 || !frames || F < 2) return;
            const w = chart.clientWidth || 300, h = chart.height;
            if (chart.width !== w) chart.width = w;
            const g = chart.getContext("2d");
            g.clearRect(0, 0, w, h);
            const L = 44, R = 4, T = 6, B = 14;
            const X = f => L + (w - L - R) * f / (F - 1);
            let lo = Infinity, hi = -Infinity;
            for (let f = 0; f < F; f++) { const v = valAt(f, selected); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
            if (!Number.isFinite(lo)) return;
            if (field === "load") lo = 0;
            if (hi - lo < 1) { hi += .5; lo = field === "load" ? 0 : lo - .5; }
            const Y = v => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(lo)); g.lineTo(w - R, Y(lo)); g.moveTo(L, Y(hi)); g.lineTo(w - R, Y(hi)); g.stroke();
            if (field === "index") {
              const thr = params().thr;
              if (thr > lo && thr < hi) { g.setLineDash([3, 3]); g.strokeStyle = "#8b929c"; g.beginPath(); g.moveTo(L, Y(thr)); g.lineTo(w - R, Y(thr)); g.stroke(); g.setLineDash([]); }
            } else {
              g.setLineDash([2, 3]); g.strokeStyle = "rgba(247,209,61,.5)";
              for (const c of contours()) if (c > lo && c < hi) { g.beginPath(); g.moveTo(L, Y(c)); g.lineTo(w - R, Y(c)); g.stroke(); }
              g.setLineDash([]);
            }
            g.fillStyle = "#8b929c"; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "right";
            g.fillText(fmtC(hi), L - 4, Y(hi) + 4); g.fillText(fmtC(lo), L - 4, Y(lo) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, h - 3);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - R, h - 3);
            g.strokeStyle = "#e6c14a"; g.lineWidth = 1.5; g.beginPath();
            let pen = false;
            for (let f = 0; f < F; f++) { const v = valAt(f, selected); if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
            g.stroke();
            g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), h - B); g.stroke();
            const cv = valAt(frame, selected);
            if (Number.isFinite(cv)) { g.fillStyle = "#ffffff"; g.beginPath(); g.arc(X(frame), Y(cv), 3, 0, 6.283); g.fill(); }
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
          // the read's cost for the dates picked: every 90-day store chunk the window
          // touches is fetched to its filled depth, unless it is on the disk mirror
          function costOf() {
            const w = cfg.win || {};
            if (!w.store_start || !w.store_hours) return w.cost || "30 s";
            const a = Date.parse(d0In.value), b = Date.parse(d1In.value), s0 = Date.parse(w.store_start);
            if (!Number.isFinite(a) || !Number.isFinite(b)) return w.cost || "30 s";
            const h0 = Math.floor((Math.min(a, b) - s0) / 36e5), h1 = Math.min(Math.floor((Math.max(a, b) - s0) / 36e5) + 23, w.store_hours - 1);
            const ch = w.chunk_h || 2160; let filled = 0, onDisk = true;
            for (let c = Math.floor(h0 / ch); c <= Math.floor(h1 / ch); c++) { filled += Math.min(ch, w.store_hours - c * ch); if (!(w.mirrored || []).includes(c)) onDisk = false; }
            if (onDisk) return "seconds (chunk on the disk mirror)";
            const s = Math.round(6 + 0.055 * filled);
            return (s < 90 ? `${s} s` : `${Math.round(s / 60)} min`) + " from S3";
          }
          function checkWindow() {
            const w = cfg.win || {};
            const n = dayCount(), lim = w.hourly_max || 14;
            let bad = "";
            if (!n) bad = "pick both days";
            else if (n > lim) bad = `${n} days is over the ${lim}-day limit`;
            noteEl.classList.toggle("rf-bad", !!bad);
            const mb = w.n_px ? ` · ${(n * 24 * w.n_px / 1e6).toFixed(0)} MB per field` : "";
            if (loading) noteEl.textContent = `loading ${n} days · ${costOf()}…`;
            else noteEl.textContent = bad || `${n} UTC days · ${n * 24} hourly frames${mb} · read ${costOf()} · limit ${lim} d`;
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
          d0In.onchange = d1In.onchange = checkWindow;
          loadBtn.onclick = () => {
            if (!checkWindow()) return;
            let d0 = d0In.value, d1 = d1In.value;
            if (d1 < d0) [d0, d1] = [d1, d0];
            loading = true; loadBtn.textContent = "loading"; frame = 0; checkWindow();
            model.set("window", JSON.stringify({d0, d1}));
            model.save_changes();
          };
          checkWindow();

          // THE PICK: H3 first (latLngToCell res 7 into the label index), the Lambert
          // snap when the cell holds no pixel centre; the readout says which one hit.
          function pick(lng, lat) {
            let i = -1, how = "";
            try { i = labelIndex.get(latLngToCell(lat, lng, geo.res_l || 7)) ?? -1; } catch (e) { i = -1; }
            if (i >= 0) how = `via H3 label (res ${geo.res_l || 7})`;
            else {
              const [x, y] = LCC(lng, lat);
              const c = Math.floor((x - geo.x0) / geo.dx), r = Math.floor((y - geo.y0) / geo.dy);
              if (r >= 0 && r < ny && c >= 0 && c < nx) { i = pixRow[r * nx + c]; how = i >= 0 ? "via Lambert snap (empty res 7 cell)" : "not land"; }
              else how = "outside the box";
            }
            if (i >= 0 && i !== selected) {
              selected = i;
              root.classList.add("rf-picked");
              if (root.classList.contains("rf-collapsed")) { root.classList.remove("rf-collapsed"); toggle.textContent = "hide"; }
              const g = lidx[i], r = Math.floor(g / nx), c = g % nx;
              const cn = cty && cty[i] !== 65535 ? names[cty[i]] : null;
              cname.textContent = cn ? `px (${r}, ${c}) · ${cn}` : `px (${r}, ${c})`;
              howEl.textContent = `${how} · cell ${lab[i].toString(16)}`;
            } else { selected = -1; root.classList.remove("rf-picked"); }
            update();
          }
          q(".rf-clear").onclick = () => { selected = -1; root.classList.remove("rf-picked"); update(); };

          const rulerText = () => `${N.toLocaleString()} land px · ${(ny * nx).toLocaleString()} quads · ${P.toLocaleString()} res 6 cells\nindex + mesh ${buildMs} ms · ${F} frames · paint ${paintMs} ms · render ${renderMs} ms` + (domeInfo ? "\n" + domeInfo : "");
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
          q(".rf-prev").onclick = () => step(-1);
          q(".rf-next").onclick = () => step(1);
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          fpsSel.onchange = () => { if (playing) setPlaying(true); };
          // "rf-collapsed", never "hidden": marimo's Tailwind owns `.hidden`
          toggle.onclick = () => { root.classList.toggle("rf-collapsed"); toggle.textContent = root.classList.contains("rf-collapsed") ? "show" : "hide"; };
          q(".rf-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen?.(); };
          mapEl.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 640) + "px"; });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "BUTTON") return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".rf-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { toggle.click(); }
            else if (ev.key === "i" || ev.key === "I") { setField("index"); }
            else if (ev.key === "l" || ev.key === "L") { setField("load"); }
            else if (ev.key === "b" || ev.key === "B") { bndBtn.click(); }
          });

          function boot() {
            loadStatic(); loadFrames();
            deck = new Deck({
              parent: mapEl, initialViewState: HOME, controller: true, layers: layers(),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); console.error(e); },
              onAfterRender: () => { if (renderT0) { renderMs = Math.round(performance.now() - renderT0); renderT0 = 0; ruler.textContent = rulerText(); } },
            });
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".rf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
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
        labels = traitlets.Bytes(b"").tag(sync=True)
        pidx = traitlets.Bytes(b"").tag(sync=True)
        cty = traitlets.Bytes(b"").tag(sync=True)
        names = traitlets.Unicode("[]").tag(sync=True)
        geom = traitlets.Unicode("{}").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        wx = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        # browser -> kernel, the one thing that crosses back: {"d0","d1"} JSON from the
        # HUD's load button ("" until the first load, meaning the default window).
        window = traitlets.Unicode("").tag(sync=True)

    return (RasterFilm,)


@app.cell
async def _(
    CACHE_DIR,
    CONUS_BOX,
    COUNTY_Z,
    NOT_CONUS,
    OVERTURE_RELEASE,
    PM_BUCKET,
    PM_PATH,
    S3Store,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    pq,
    struct,
):
    # THE COUNTIES, OUT OF ONE PMTILES OBJECT BY RANGED GET (x-sql-marimo's reader, by
    # copy). Disk-cached as parquet per release, zoom and box; the cache file is shared
    # with the x-sql-marimo notebooks.
    import time as _ctime

    _ct0 = _ctime.perf_counter()
    import pathlib as _pl

    _cache = (
        _pl.Path(CACHE_DIR) / f"counties-{OVERTURE_RELEASE}-z{COUNTY_Z}-{'-'.join(str(b) for b in CONUS_BOX)}.parquet"
        if CACHE_DIR
        else None
    )
    _rows, _x0, _y0, _x1, _y1, _t_fetch = [], 0, 0, -1, -1, 0.0
    if _cache is not None and _cache.exists():
        counties = pq.read_table(_cache)
        _how = f"from {_cache}"
    else:
        _pm_store = S3Store(PM_BUCKET, region="us-west-2", skip_signature=True)

        async def _pm_range(a, b):
            """Inclusive byte range [a, b]. obstore's `end` is exclusive."""
            return bytes(memoryview(await obstore.get_range_async(_pm_store, PM_PATH, start=a, end=b + 1)))

        def _varint(buf, i):
            r = s = 0
            while True:
                c = buf[i]
                i += 1
                r |= (c & 0x7F) << s
                if not c & 0x80:
                    return r, i
                s += 7

        def _parse_dir(buf):
            """A PMTiles v3 directory: four varint columns, tile ids delta-encoded."""
            n, i = _varint(buf, 0)
            ids, last = [0] * n, 0
            for k in range(n):
                v, i = _varint(buf, i)
                last += v
                ids[k] = last
            runs = [0] * n
            for k in range(n):
                runs[k], i = _varint(buf, i)
            lens = [0] * n
            for k in range(n):
                lens[k], i = _varint(buf, i)
            offs = [0] * n
            for k in range(n):
                v, i = _varint(buf, i)
                offs[k] = (offs[k - 1] + lens[k - 1]) if v == 0 and k > 0 else v - 1
            return list(zip(ids, offs, lens, runs))

        def _tile_id(z, x, y):
            """z/x/y -> PMTiles v3 tile id: Hilbert order within a level, levels stacked."""
            acc = sum((1 << t) * (1 << t) for t in range(z))
            n = 1 << z
            d, s = 0, n >> 1
            while s > 0:
                rx = 1 if x & s else 0
                ry = 1 if y & s else 0
                d += s * s * ((3 * rx) ^ ry)
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - x, s - 1 - y
                    x, y = y, x
                s >>= 1
            return acc + d

        def _find(entries, tid):
            """Binary search, falling back to the run that COVERS tid."""
            lo, hi = 0, len(entries) - 1
            while lo <= hi:
                m = (lo + hi) // 2
                if tid < entries[m][0]:
                    hi = m - 1
                elif tid > entries[m][0]:
                    lo = m + 1
                else:
                    return entries[m]
            if hi >= 0 and (entries[hi][3] == 0 or tid - entries[hi][0] < entries[hi][3]):
                return entries[hi]
            return None

        _hdr = await _pm_range(0, 126)
        assert _hdr[:7] == b"PMTiles" and _hdr[7] == 3, "not a PMTiles v3 archive"
        _rd_off, _rd_len, _, _, _ld_off, _, _td_off, _ = struct.unpack("<8Q", _hdr[8:72])
        assert COUNTY_Z <= _hdr[101], "COUNTY_Z above the pyramid"
        _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
        _leaf = {}

        def _fields(buf):
            """Iterate (field_number, wire_type, value) over one protobuf message."""
            i, n = 0, len(buf)
            while i < n:
                key, i = _varint(buf, i)
                f, w = key >> 3, key & 0x7
                if w == 0:
                    v, i = _varint(buf, i)
                elif w == 2:
                    ln, i = _varint(buf, i)
                    v = buf[i : i + ln]
                    i += ln
                elif w == 5:
                    v = buf[i : i + 4]
                    i += 4
                elif w == 1:
                    v = buf[i : i + 8]
                    i += 8
                else:
                    raise ValueError(f"wire type {w}")
                yield f, w, v

        def _value(buf):
            """An MVT Value message: exactly one of its fields is set."""
            for f, _w, v in _fields(buf):
                if f == 1:
                    return v.decode("utf-8")
                if f == 2:
                    return struct.unpack("<f", v)[0]
                if f == 3:
                    return struct.unpack("<d", v)[0]
                if f in (4, 5):
                    return v
                if f == 6:
                    return (v >> 1) ^ -(v & 1)
                if f == 7:
                    return bool(v)
            return None

        def _mvt_rings(geom):
            """Packed geometry commands -> rings of (x, y) tile coords, closed."""
            rings, ring = [], None
            x = y = 0
            i, n = 0, len(geom)
            while i < n:
                cmd, i = _varint(geom, i)
                op, count = cmd & 0x7, cmd >> 3
                if op == 1:
                    for _ in range(count):
                        dx, i = _varint(geom, i)
                        dy, i = _varint(geom, i)
                        x += (dx >> 1) ^ -(dx & 1)
                        y += (dy >> 1) ^ -(dy & 1)
                        ring = [(x, y)]
                        rings.append(ring)
                elif op == 2:
                    for _ in range(count):
                        dx, i = _varint(geom, i)
                        dy, i = _varint(geom, i)
                        x += (dx >> 1) ^ -(dx & 1)
                        y += (dy >> 1) ^ -(dy & 1)
                        ring.append((x, y))
                elif op == 7:
                    ring.append(ring[0])
                else:
                    raise ValueError(f"geometry op {op}")
            return rings

        def _area2(ring):
            a = 0
            for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
                a += x0 * y1 - x1 * y0
            return a

        def _division_areas(tile_buf):
            """The division_area layer: ([(properties, [(exterior, holes), ...]), ...], extent)."""
            for f, _w, v in _fields(tile_buf):
                if f != 3:
                    continue
                name, extent = None, 4096
                keys, values, feats = [], [], []
                for lf, _lw, lv in _fields(v):
                    if lf == 1:
                        name = lv.decode("utf-8")
                    elif lf == 2:
                        feats.append(lv)
                    elif lf == 3:
                        keys.append(lv.decode("utf-8"))
                    elif lf == 4:
                        values.append(_value(lv))
                    elif lf == 5:
                        extent = lv
                if name != "division_area":
                    continue
                out = []
                for fv in feats:
                    tags, gtype, geom = [], 0, b""
                    for ff, _fw, fvv in _fields(fv):
                        if ff == 2:
                            i = 0
                            while i < len(fvv):
                                t, i = _varint(fvv, i)
                                tags.append(t)
                        elif ff == 3:
                            gtype = fvv
                        elif ff == 4:
                            geom = fvv
                    if gtype != 3:
                        continue
                    props = {keys[tags[i]]: values[tags[i + 1]] for i in range(0, len(tags), 2)}
                    polys, cur = [], None
                    for ring in _mvt_rings(geom):
                        if _area2(ring) > 0:
                            cur = (ring, [])
                            polys.append(cur)
                        elif cur is not None:
                            cur[1].append(ring)
                    out.append((props, polys))
                return out, extent
            return [], 4096

        def _feature_wkb(polys, z, x, y, extent):
            """Tile-integer rings -> a lon/lat MultiPolygon WKB, closed-form Web Mercator."""
            n = 1 << z
            parts = []
            for ext, holes in polys:
                rings = []
                for r in (ext, *holes):
                    a = np.asarray(r, dtype=np.float64)
                    pts = np.empty_like(a)
                    pts[:, 0] = (x + a[:, 0] / extent) / n * 360.0 - 180.0
                    pts[:, 1] = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + a[:, 1] / extent) / n))))
                    rings.append(struct.pack("<I", len(a)) + pts.tobytes())
                parts.append(struct.pack("<BII", 1, 3, len(rings)) + b"".join(rings))
            return struct.pack("<BII", 1, 6, len(parts)) + b"".join(parts)

        _sem = asyncio.Semaphore(32)

        async def _tile_pieces(z, x, y):
            """One tile, walked to through the directories, decoded, filtered to CONUS counties."""
            tid, ents = _tile_id(z, x, y), _root
            blob = None
            for _ in range(4):
                e = _find(ents, tid)
                if e is None:
                    break
                if e[3] == 0:
                    lk = (e[1], e[2])
                    if lk not in _leaf:
                        _leaf[lk] = _parse_dir(gzip.decompress(await _pm_range(_ld_off + e[1], _ld_off + e[1] + e[2] - 1)))
                    ents = _leaf[lk]
                    continue
                async with _sem:
                    blob = await _pm_range(_td_off + e[1], _td_off + e[1] + e[2] - 1)
                break
            pieces = []
            if blob is not None:
                if blob[:2] == b"\x1f\x8b":
                    blob = gzip.decompress(blob)
                feats, extent = _division_areas(blob)
                for props, polys in feats:
                    if props.get("subtype") != "county":
                        continue
                    if props.get("is_land") is not True or not polys:
                        continue
                    if props.get("country") != "US":
                        continue
                    region = (props.get("region") or "").split("-", 1)[-1]
                    if region in NOT_CONUS:
                        continue
                    pieces.append(
                        {
                            "id": props.get("division_id") or props.get("id"),
                            "name": props.get("@name"),
                            "region": region,
                            "wkb": _feature_wkb(polys, z, x, y, extent),
                        }
                    )
            return pieces

        def _mtile(lon, lat, z):
            n = 1 << z
            xx = min(n - 1, max(0, int((lon + 180.0) / 360.0 * n)))
            la = min(85.05, max(-85.05, lat))
            yy = (1.0 - math.log(math.tan(math.radians(la)) + 1.0 / math.cos(math.radians(la))) / math.pi) / 2.0
            return xx, min(n - 1, max(0, int(yy * n)))

        _x0, _y0 = _mtile(CONUS_BOX[0], CONUS_BOX[3], COUNTY_Z)
        _x1, _y1 = _mtile(CONUS_BOX[2], CONUS_BOX[1], COUNTY_Z)
        _parts = await asyncio.gather(*(_tile_pieces(COUNTY_Z, xx, yy) for yy in range(_y0, _y1 + 1) for xx in range(_x0, _x1 + 1)))
        _rows = [p for tp in _parts for p in tp]
        _t_fetch = _ctime.perf_counter() - _ct0

        # THE SEAM DISSOLVE: tile geometry arrives clipped; union per division removes
        # every interior edge. con.register, not the replacement scan (marimo mangles
        # underscore locals).
        _pieces = pa.table(
            {
                "id": pa.array([r["id"] for r in _rows]),
                "name": pa.array([r["name"] for r in _rows]),
                "region": pa.array([r["region"] for r in _rows]),
                "wkb": pa.array([r["wkb"] for r in _rows], pa.binary()),
            }
        )
        con.register("pm_pieces", _pieces)
        counties = con.sql("""
            SELECT id, any_value(name) AS name, any_value(region) AS region,
                   CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
            FROM pm_pieces GROUP BY id
        """).to_arrow_table()
        con.unregister("pm_pieces")
        _how = "fetched"
        if _cache is not None:
            _cache.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(counties, _cache)

    county_stats = (
        f"{counties.num_rows:,} counties {_how} · "
        + (f"{(_x1 - _x0 + 1) * (_y1 - _y0 + 1)} tiles at z{COUNTY_Z} · {len(_rows):,} pieces · fetch {_t_fetch:.1f}s, with dissolve " if _rows else "")
        + f"{_ctime.perf_counter() - _ct0:.1f}s"
    )
    return counties, county_stats


@app.cell
def _(asyncio):
    # A read-only zarr v3 Store that mirrors byte ranges of the icechunk session store
    # to a local directory, keyed by (key, byte range) exactly as the sharding codec
    # asks. The caller says which keys are mirrorable: full time shards of the read
    # variables. By copy from x-sql-marimo/xsql-hrrr-heat-domes.py.
    import hashlib as _hashlib
    import os as _os

    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest

    class MirrorStore(Store):
        def __init__(self, inner: Store, root: str, mirrorable):
            super().__init__(read_only=True)
            self.inner, self.root, self.mirrorable = inner, root, mirrorable
            self.hits = self.misses = 0
            _os.makedirs(root, exist_ok=True)

        supports_writes = False
        supports_deletes = False
        supports_partial_writes = False
        supports_listing = True

        def __eq__(self, other):
            return isinstance(other, MirrorStore) and other.inner == self.inner and other.root == self.root

        def _tag(self, r):
            if r is None:
                return "all"
            if isinstance(r, RangeByteRequest):
                return f"r{r.start}-{r.end}"
            if isinstance(r, OffsetByteRequest):
                return f"o{r.offset}"
            if isinstance(r, SuffixByteRequest):
                return f"s{r.suffix}"
            return "x" + _hashlib.sha1(repr(r).encode()).hexdigest()[:12]

        def _path(self, key, r):
            return _os.path.join(self.root, key.replace("/", "__") + "." + self._tag(r))

        def _read(self, key, r, prototype):
            try:
                with open(self._path(key, r), "rb") as f:
                    self.hits += 1
                    return prototype.buffer.from_bytes(f.read())
            except FileNotFoundError:
                return None

        def _write(self, key, r, buf):
            p = self._path(key, r)
            tmp = f"{p}.{_os.getpid()}.{id(buf)}.tmp"
            with open(tmp, "wb") as f:
                f.write(buf.to_bytes())
            try:
                _os.replace(tmp, p)
            except FileNotFoundError:
                pass

        async def get(self, key, prototype, byte_range=None):
            if not self.mirrorable(key):
                return await self.inner.get(key, prototype, byte_range)
            buf = await asyncio.to_thread(self._read, key, byte_range, prototype)
            if buf is not None:
                return buf
            self.misses += 1
            buf = await self.inner.get(key, prototype, byte_range)
            if buf is not None:
                await asyncio.to_thread(self._write, key, byte_range, buf)
            return buf

        async def get_ranges(self, key, byte_ranges, *, prototype, max_concurrency=10, max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20):
            if not self.mirrorable(key):
                async for group in self.inner.get_ranges(key, byte_ranges, prototype=prototype, max_concurrency=max_concurrency,
                                                         max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes):
                    yield group
                return
            ranges = list(byte_ranges)
            held = await asyncio.gather(*(asyncio.to_thread(self._read, key, r, prototype) for r in ranges))
            hit = [(i, b) for i, b in enumerate(held) if b is not None]
            if hit:
                yield hit
            miss = [i for i, b in enumerate(held) if b is None]
            if not miss:
                return
            self.misses += len(miss)
            async for group in self.inner.get_ranges(key, [ranges[i] for i in miss], prototype=prototype, max_concurrency=max_concurrency,
                                                     max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes):
                out = []
                for j, buf in group:
                    i = miss[j]
                    if buf is not None:
                        await asyncio.to_thread(self._write, key, ranges[i], buf)
                    out.append((i, buf))
                yield out

        async def get_partial_values(self, prototype, key_ranges):
            return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

        async def exists(self, key):
            return await self.inner.exists(key)

        async def set(self, key, value):
            raise NotImplementedError("read-only mirror")

        async def delete(self, key):
            raise NotImplementedError("read-only mirror")

        def list(self):
            return self.inner.list()

        def list_prefix(self, prefix):
            return self.inner.list_prefix(prefix)

        def list_dir(self, prefix):
            return self.inner.list_dir(prefix)

        async def getsize(self, key):
            return await self.inner.getsize(key)

    return (MirrorStore,)


@app.cell
def _(ANALYSIS_BUCKET, ANALYSIS_PREFIX, CHUNK_CACHE_GB, MIRROR_DIR, MirrorStore, READ_RAIN, READ_WIND, np, xr, zarr):
    # THE STORE, opened once; only metadata and the 2-D lat/lon are read here.
    import os as _sos
    import time as _stime

    import icechunk

    _st0 = _stime.perf_counter()
    _storage = icechunk.s3_storage(bucket=ANALYSIS_BUCKET, prefix=ANALYSIS_PREFIX, region="us-west-2", anonymous=True)
    _sess = icechunk.Repository.open(
        _storage,
        config=icechunk.RepositoryConfig(caching=icechunk.CachingConfig(num_bytes_chunks=int(CHUNK_CACHE_GB * (1 << 30)))),
    ).readonly_session("main")
    VARS = ["temperature_2m", "relative_humidity_2m"]
    if READ_RAIN:
        VARS.append("precipitation_surface")
    if READ_WIND:
        VARS += ["wind_u_10m", "wind_v_10m"]
    # THE DISK MIRROR: full time shards of the read variables are read from disk after
    # the first kernel that fetched them; the youngest shard grows hourly and is never
    # mirrored. `mirrored` lists the time-chunk indices on disk for every read variable
    # (by file name; a partial chunk counts, its missing blocks come from the wire).
    _T = zarr.open_group(_sess.store, mode="r")["time"].shape[0]
    _young = (_T - 1) // 2160
    _mvars = set(VARS)

    def _mirrorable(key, _young=_young, _mvars=_mvars):
        p = key.split("/")
        return len(p) == 5 and p[0] in _mvars and p[1] == "c" and p[2].isdigit() and int(p[2]) < _young

    mirror = MirrorStore(_sess.store, MIRROR_DIR, _mirrorable) if MIRROR_DIR else None
    store = mirror if mirror is not None else _sess.store
    mirrored = []
    if mirror is not None and _sos.path.isdir(MIRROR_DIR):
        _seen = {v: set() for v in VARS}
        for _f in _sos.listdir(MIRROR_DIR):
            if "__c__" not in _f or _f.endswith(".tmp"):
                continue
            _v, _rest = _f.split("__c__", 1)
            if _v in _seen:
                _seen[_v].add(int(_rest.split("__", 1)[0]))
        mirrored = sorted(set.intersection(*_seen.values())) if _seen else []
    _ds = xr.open_zarr(store, consolidated=False, chunks=None)
    all_times = _ds["time"].values.astype("datetime64[m]")
    lat = _ds["latitude"].values.astype("float64")
    lon = _ds["longitude"].values.astype("float64")
    grid_y = _ds["y"].values.astype("float64")
    grid_x = _ds["x"].values.astype("float64")
    crs_wkt = _ds["spatial_ref"].attrs["crs_wkt"]
    source_note = "HRRR analysis"
    store_stats = (
        f"{source_note} · {len(VARS)} variables · grid {lat.shape[1]}x{lat.shape[0]} px · hourly "
        f"{np.datetime_as_string(all_times[0])} to {np.datetime_as_string(all_times[-1])} UTC "
        f"({all_times.size:,} steps) · open {_stime.perf_counter() - _st0:.1f}s"
        + (f" · disk mirror in {MIRROR_DIR}, time chunks {mirrored} for all read variables" if mirror is not None else "")
    )
    return VARS, all_times, crs_wkt, grid_x, grid_y, lat, lon, mirror, mirrored, source_note, store, store_stats


@app.cell
def _(BOX, RES_L, RES_T, change_resolution, con, coordinates_to_cells, counties, crs_wkt, grid_x, grid_y, lat, lon, np, pa):
    # THE GEOMETRY, ONCE. The box's bounding rectangle of the grid; pixel corners in web
    # mercator world coords from the store's own GeoTransform through pyproj (the mesh);
    # the res 7 LABEL per pixel from the store's own lat/lon and its res 6 PARENT; the
    # land mask and county per pixel from the res 6 polyfill join; the store blocks that
    # touch land inside the box (what the read fetches); the Lambert snap constants.
    import time as _gtime

    from pyproj import CRS, Transformer

    _gt0 = _gtime.perf_counter()
    _ny, _nx = lat.shape
    if BOX:
        _in = (lon >= BOX[0]) & (lon <= BOX[2]) & (lat >= BOX[1]) & (lat <= BOX[3])
        _rows, _cols = np.flatnonzero(_in.any(axis=1)), np.flatnonzero(_in.any(axis=0))
        r0, r1, c0, c1 = int(_rows[0]), int(_rows[-1]) + 1, int(_cols[0]), int(_cols[-1]) + 1
    else:
        r0, r1, c0, c1 = 0, _ny, 0, _nx
    ny, nx = r1 - r0, c1 - c0
    _lat, _lon = lat[r0:r1, c0:c1], lon[r0:r1, c0:c1]

    # the mesh: (ny+1)(nx+1) pixel corners, LCC metres -> lon/lat -> web mercator world
    # coords (512 units per world, y up as deck's CARTESIAN space wants)
    dx, dy = float(grid_x[1] - grid_x[0]), float(grid_y[1] - grid_y[0])  # dy is negative: row 0 is the north edge
    _cx = np.concatenate([grid_x[c0:c1] - dx / 2, [grid_x[c1 - 1] + dx / 2]])
    _cy = np.concatenate([grid_y[r0:r1] - dy / 2, [grid_y[r1 - 1] + dy / 2]])
    _CX, _CY = np.meshgrid(_cx, _cy)
    _to_ll = Transformer.from_crs(CRS.from_wkt(crs_wkt), "EPSG:4326", always_xy=True)
    _clon, _clat = _to_ll.transform(_CX.ravel(), _CY.ravel())
    _wx = (_clon + 180.0) / 360.0 * 512.0
    _wy = (1.0 - np.log(np.tan(np.radians(_clat)) + 1.0 / np.cos(np.radians(_clat))) / np.pi) / 2.0 * 512.0
    corners = np.stack([_wx, 512.0 - _wy], axis=-1).astype(np.float32).reshape(ny + 1, nx + 1, 2)
    _t_mesh = _gtime.perf_counter() - _gt0

    # the label layer: one res RES_L cell per pixel, its res RES_T parent
    label7 = np.asarray(coordinates_to_cells(_lat.ravel(), _lon.ravel(), int(RES_L))).astype(np.uint64)
    parent6 = np.asarray(change_resolution(pa.array(label7), int(RES_T))).astype(np.uint64)
    n_unique_labels = int(np.unique(label7).size)

    # land + county: the res RES_T cells whose centre falls in a county ('center' rule,
    # one county per cell); a pixel is land if its parent is one of them
    con.register("conus_divs", counties)
    _m = con.sql(
        """
        WITH parts AS (SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM conus_divs),
        filled AS (SELECT id, UNNEST(h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'center')) AS hex FROM parts)
        SELECT hex, any_value(id) AS id FROM filled GROUP BY hex ORDER BY hex
        """,
        params=[int(RES_T)],
    ).to_arrow_table()
    con.unregister("conus_divs")
    _cells6 = _m["hex"].to_numpy().astype(np.uint64)
    _pos = np.searchsorted(_cells6, parent6)
    _pos[_pos >= _cells6.size] = 0
    land = (_cells6[_pos] == parent6).reshape(ny, nx)
    _cid = counties["id"].to_pylist()
    _cpos = {i: k for k, i in enumerate(_cid)}
    _cell_county = np.fromiter((_cpos.get(i, 65535) for i in _m["id"].to_pylist()), dtype=np.uint16, count=_cells6.size)
    county_names = [f"{n}, {r}" for n, r in zip(counties["name"].to_pylist(), counties["region"].to_pylist())]

    # the land pixels, row-major inside the box: what crosses the bridge and what is read
    lidx = np.flatnonzero(land.ravel()).astype(np.uint32)
    N = int(lidx.size)
    lab = label7[lidx]
    par_cells, pidx = np.unique(parent6[lidx], return_inverse=True)
    pidx = pidx.astype(np.uint32)
    P = int(par_cells.size)
    pcount = np.bincount(pidx, minlength=P).astype(np.int32)
    cty = np.full(N, 65535, np.uint16)
    cty[:] = _cell_county[_pos[lidx]]

    # the store blocks (45x45, full-grid aligned) that hold a land pixel of the box:
    # (rows slice, cols slice, land rows they fill, flat index inside the slice)
    _B = 45
    _prow = np.full(ny * nx, -1, np.int64)
    _prow[lidx] = np.arange(N)
    _prow = _prow.reshape(ny, nx)
    blocks = []
    for _j in range(r0 // _B, -(-r1 // _B)):
        for _i in range(c0 // _B, -(-c1 // _B)):
            _ys = slice(max(_j * _B, r0), min((_j + 1) * _B, r1))
            _xs = slice(max(_i * _B, c0), min((_i + 1) * _B, c1))
            _sub = _prow[_ys.start - r0 : _ys.stop - r0, _xs.start - c0 : _xs.stop - c0].ravel()
            _loc = np.flatnonzero(_sub >= 0)
            if _loc.size:
                blocks.append((_ys, _xs, _sub[_loc], _loc))

    # the Lambert snap: edge origin of the box and the signed pixel steps; the home view
    _clon0, _clat0 = float(np.mean(_lon)), float(np.mean(_lat))
    _lonw = float(_lon.max() - _lon.min())
    geom = {
        "ny": ny, "nx": nx, "n": N, "p": P, "res_l": int(RES_L), "res_t": int(RES_T),
        "x0": float(grid_x[c0] - dx / 2), "y0": float(grid_y[r0] - dy / 2), "dx": dx, "dy": dy,
        "home": {"longitude": _clon0, "latitude": _clat0, "zoom": float(np.clip(np.log2(1000 * 360 / (256 * _lonw)), 3, 8))},
    }
    pix_stats = (
        f"box rows {r0}:{r1} cols {c0}:{c1} ({ny}x{nx} = {ny * nx:,} px) · {N:,} land px · "
        f"{n_unique_labels:,} res {RES_L} labels for {ny * nx:,} px · {P:,} res {RES_T} parents "
        f"({N / max(P, 1):.2f} land px per cell) · {len(blocks)} store blocks · "
        f"mesh {_t_mesh:.1f}s, all {_gtime.perf_counter() - _gt0:.1f}s · "
        f"{corners.nbytes / 1e6:.0f} MB mesh + {(lab.nbytes + lidx.nbytes + pidx.nbytes + cty.nbytes) / 1e6:.0f} MB indices to the browser"
    )
    return N, P, blocks, corners, county_names, cty, geom, lab, lidx, ny, nx, par_cells, pcount, pidx, pix_stats


@app.cell
def _():
    # Kernel-side memo across window loads: the last read window and its matrices.
    HOLD = {"key": None, "raw": None, "times": None, "stats": ""}
    return (HOLD,)


@app.cell
def _(RasterFilm, corners, county_names, cty, geom, json, lab, lidx, mo, pidx):
    # THE WIDGET, BUILT ONCE with the geometry and nothing else. Frames and config are
    # set from the wiring cell below, so a window change never rebuilds the mesh.
    film = mo.ui.anywidget(
        RasterFilm(
            corners=corners.astype("<f4").tobytes(),
            lidx=lidx.astype("<u4").tobytes(),
            labels=lab.astype("<u8").tobytes(),
            pidx=pidx.astype("<u4").tobytes(),
            cty=cty.astype("<u2").tobytes(),
            names=json.dumps(county_names),
            geom=json.dumps(geom),
        )
    )
    film
    return (film,)


@app.cell
def _(DAYS, HOURLY_MAX_DAYS, N, all_times, film, json, mirrored, mo, np):
    # THE WINDOW: the HUD's `window` trait once "load" has been pressed, else the
    # opening default. Read off the widget, not film.value (that packs every synced
    # trait, the frame bytes included). Over the limit stops with the reason.
    import datetime as _wdt

    _last = all_times[-1].astype("datetime64[D]").astype(_wdt.date)
    _first = all_times[0].astype("datetime64[D]").astype(_wdt.date)
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
    _h0 = int((np.datetime64(_d0.isoformat()) - all_times[0].astype("datetime64[D]")) // np.timedelta64(1, "h"))
    _h1 = int((np.datetime64(_d1.isoformat()) - all_times[0].astype("datetime64[D]")) // np.timedelta64(1, "h")) + 23
    _chunks = list(range(_h0 // 2160, min(_h1, all_times.size - 1) // 2160 + 1))
    _filled = sum(min(2160, all_times.size - _c * 2160) for _c in _chunks)
    read_cost_s = int(round(6 + 0.055 * _filled))
    win_cfg = {
        "first": _first.isoformat(),
        "last": _last.isoformat(),
        "d0": _d0.isoformat(),
        "d1": _d1.isoformat(),
        "hourly_max": HOURLY_MAX_DAYS,
        "cost": "seconds" if all(c in mirrored for c in _chunks) else (f"{read_cost_s} s" if read_cost_s < 90 else f"{read_cost_s / 60:.0f} min"),
        "chunk_h": 2160,
        "store_start": all_times[0].astype("datetime64[D]").astype(_wdt.date).isoformat(),
        "store_hours": int(all_times.size),
        "mirrored": [int(c) for c in mirrored],
        "n_px": int(N),
    }
    mo.stop(n_days > HOURLY_MAX_DAYS, mo.md(f"**{n_days} days is over the {HOURLY_MAX_DAYS}-day limit.** Shorten the window."))
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[m]")
    t1 = min((np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[m]"), all_times[-1])
    window_note = f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to {np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    return n_days, t0, t1, win_cfg, window_note


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the wait in the next cell comes from

    The cell below reads the window. There is no fold: the unit is the pixel, so the
    read is the store's own blocks, straight off the Zarr. For each variable and each
    45 × 45 store block that holds a land pixel of the box, the window's hours are
    sliced out of the block's 2,160-hour chunk and written into an (hours × land
    pixels) matrix; eight threads, through the disk mirror when the chunk is on it.
    From the mirror a week is seconds; from S3 it is the wire (~21 MB/s from a home
    link, ~2 min for a full chunk with two variables). The heat index is numpy, per
    hour, from the pixel's own temperature and humidity: no averaging anywhere.
    """)
    return


@app.cell
def _(HOLD, N, READ_THREADS, VARS, all_times, blocks, mirror, np, store, t0, t1, zarr):
    # THE READ, block-wise into (F, N): one zarr slice per (variable, block), written
    # straight into the land-pixel columns. Memoised on the window.
    import time as _rtime
    from concurrent.futures import ThreadPoolExecutor as _Pool

    _key = (str(t0), str(t1), tuple(VARS), int(N), len(blocks))
    if HOLD["key"] == _key and HOLD["raw"] is not None:
        raw, frame_times = HOLD["raw"], HOLD["times"]
        read_stats = HOLD["stats"] + " (memo)"
    else:
        _rt0 = _rtime.perf_counter()
        _i0 = int(np.searchsorted(all_times, t0))
        _i1 = int(np.searchsorted(all_times, t1, side="right"))
        frame_times = all_times[_i0:_i1]
        _F = int(frame_times.size)
        _g = zarr.open_group(store, mode="r")
        _za = {v: _g[v] for v in VARS}
        raw = {v: np.full((_F, N), np.nan, np.float32) for v in VARS}
        _h0, _m0 = (mirror.hits, mirror.misses) if mirror is not None else (0, 0)

        def _rd(job):
            v, (ys, xs, rows, loc) = job
            blk = _za[v][_i0:_i1, ys, xs]
            raw[v][:, rows] = blk.reshape(_F, -1)[:, loc]

        with _Pool(READ_THREADS) as _ex:
            list(_ex.map(_rd, [(v, b) for v in VARS for b in blocks]))
        read_stats = (
            f"{_F} hours · {len(VARS)} variables · {len(blocks)} blocks · {_F * N:,} pixel-hours · "
            f"read {_rtime.perf_counter() - _rt0:.1f}s"
            + (f" · mirror {mirror.hits - _h0} ranges from disk, {mirror.misses - _m0} fetched" if mirror is not None else "")
        )
        HOLD["key"], HOLD["raw"], HOLD["times"], HOLD["stats"] = _key, raw, frame_times, read_stats
    return frame_times, raw, read_stats


@app.cell
def _(PIVOT, SPAN, frame_times, np, raw):
    # THE FRAME MATRICES: F hours x N land pixels. Heat index per pixel (NWS: Steadman's
    # simple formula, the Rothfusz regression once the mean of it and T reaches 80 F,
    # with the two RH adjustments), uint8 in 0.5 degC steps from -40 (255 = no data).
    # Wind and rain, if read, packed in one byte: wind m/s rounded (0..15) in the high
    # nibble, rain in 0.5 mm/h steps (0..7.5) in the low.
    def _heat_index_c(tc, rh):
        T = tc * 9.0 / 5.0 + 32.0
        hi = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + rh * 0.094)
        m = (hi + T) / 2.0 >= 80.0
        T2, R2 = T[m], rh[m]
        h = (
            -42.379 + 2.04901523 * T2 + 10.14333127 * R2 - 0.22475541 * T2 * R2
            - 0.00683783 * T2 * T2 - 0.05481717 * R2 * R2 + 0.00122874 * T2 * T2 * R2
            + 0.00085282 * T2 * R2 * R2 - 0.00000199 * T2 * T2 * R2 * R2
        )
        a1 = (R2 < 13) & (T2 >= 80) & (T2 <= 112)
        h[a1] -= ((13 - R2[a1]) / 4.0) * np.sqrt((17 - np.abs(T2[a1] - 95.0)) / 17.0)
        a2 = (R2 > 85) & (T2 >= 80) & (T2 <= 87)
        h[a2] += ((R2[a2] - 85.0) / 10.0) * ((87.0 - T2[a2]) / 5.0)
        hi[m] = h
        return (hi - 32.0) * 5.0 / 9.0

    _tc, _rh = raw["temperature_2m"], raw["relative_humidity_2m"]
    F, N_ = _tc.shape
    hi_q = np.full((F, N_), 255, dtype=np.uint8)
    _lo, _hi, _sum, _cnt = np.inf, -np.inf, 0.0, 0
    _samples = []
    for _f in range(F):  # per hour: the float64 working set stays one frame wide
        _h = _heat_index_c(_tc[_f].astype(np.float64), _rh[_f].astype(np.float64))
        _ok = np.isfinite(_h)
        hi_q[_f, _ok] = np.clip(np.rint((_h[_ok] + 40.0) * 2.0), 0, 254).astype(np.uint8)
        if _ok.any():
            _lo, _hi = min(_lo, float(_h[_ok].min())), max(_hi, float(_h[_ok].max()))
            _samples.append(_h[_ok][:: max(1, _ok.sum() // 20000)])
    _vals = np.concatenate(_samples) if _samples else np.array([0.0])
    wx_q = np.zeros((F, N_), dtype=np.uint8)
    has_rain = "precipitation_surface" in raw
    has_wind = "wind_u_10m" in raw and "wind_v_10m" in raw
    if has_rain:
        wx_q |= np.clip(np.rint(np.nan_to_num(raw["precipitation_surface"]) * 3600.0 * 2.0), 0, 15).astype(np.uint8)
    if has_wind:
        _ws = np.sqrt(np.nan_to_num(raw["wind_u_10m"]) ** 2 + np.nan_to_num(raw["wind_v_10m"]) ** 2)
        wx_q |= np.clip(np.rint(_ws), 0, 15).astype(np.uint8) << 4
    frame_labels = [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in frame_times]

    _mid = float(np.median(_vals)) if PIVOT is None else float(PIVOT)
    _span = float(max(_mid - np.percentile(_vals, 2), np.percentile(_vals, 98) - _mid)) if SPAN is None else float(SPAN)
    ramp_lo, ramp_mid, ramp_hi = _mid - _span, _mid, _mid + _span
    frame_stats = (
        f"{F} frames x {N_:,} land px · heat index {_lo:.1f} to {_hi:.1f} °C · "
        f"ramp {ramp_lo:.1f} / {ramp_mid:.1f} / {ramp_hi:.1f} · {hi_q.nbytes / 1e6:.0f} MB per field to the browser"
    )
    return F, frame_labels, frame_stats, has_rain, has_wind, hi_q, ramp_hi, ramp_lo, ramp_mid, wx_q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using the map.** Space plays, arrows step, drag the slider to scrub, `I` / `L`
    switch between the heat index and the sustained heat, `B` toggles the boundaries,
    `H` folds the panel, `F` goes fullscreen. Click a pixel for its value and its line
    over the window: the readout names the pixel (row, col), its county, and whether
    the click was resolved through the H3 label or the grid's own Lambert inverse.
    The **rule** select decides dome membership per pixel or per res 6 cell; the
    ruler at the top right counts the member pixels and boundary edges per level under
    the rule in force. The sliders are the accumulator; every move recomputes the film
    in the browser. The window takes UTC days, inclusive, up to 14.
    """)
    return


@app.cell
def _(
    BASE_DARK,
    BASE_TILES,
    BASE_TINT,
    CONTOURS,
    FPS,
    HALF_LIFE,
    INDEX_STOPS,
    LABEL_TILES,
    LOAD_STOPS,
    MAP_HEIGHT,
    RAIN_FLUSH,
    THRESHOLD,
    WIND_VENT,
    film,
    frame_labels,
    frame_stats,
    has_rain,
    has_wind,
    hi_q,
    json,
    n_days,
    ramp_hi,
    ramp_lo,
    ramp_mid,
    read_stats,
    source_note,
    win_cfg,
    window_note,
    wx_q,
):
    # THE WIRING: re-runs on every window change and only pushes JSON + bytes at the
    # existing widget. Config, then wx, then frames: the JS recomputes on frames.
    film.config = json.dumps(
        {
            "labels": frame_labels,
            "lo": ramp_lo,
            "mid": ramp_mid,
            "hi": ramp_hi,
            "index_stops": INDEX_STOPS,
            "load_stops": LOAD_STOPS,
            "threshold": THRESHOLD,
            "half_life": HALF_LIFE,
            "rain_flush": RAIN_FLUSH,
            "wind_vent": WIND_VENT,
            "has_rain": has_rain,
            "has_wind": has_wind,
            "contours": CONTOURS,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "base_tiles": BASE_TILES or "",
            "base_dark": bool(BASE_DARK),
            "base_tint": BASE_TINT,
            "label_tiles": LABEL_TILES or "",
            "title": f"heat index on the native 3 km grid · {source_note}",
            "subtitle": f"{window_note} · {n_days} days · hourly · H3 res 7 label per pixel underneath, nothing hexagonal drawn",
            "meta": f"{read_stats} · {frame_stats}",
            "win": win_cfg,
            "autoplay": False,
        }
    )
    film.wx = wx_q.tobytes() if (has_rain or has_wind) else b""
    film.frames = hi_q.tobytes()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The domes as objects (DuckDB)

    The lines on the map are the browser's and follow the sliders. The table below is
    the kernel's: numpy runs the same accumulator at the constants-cell defaults and
    applies `DOME_RULE` per res 6 parent, DuckDB dissolves the member cells at or above
    each level for every hour with `h3_cells_to_multi_polygon_wkb` and `ST_Dump`s the
    multipolygons into blobs with an Albers area and a centroid. Blobs under
    `DOME_MIN_KM2` are dropped; the largest per hour at each level is the dome and its
    centroid over the hours is its track. Beside the dissolved (cell) area sits the
    plain pixel count × 9 km², so the table shows what the cell rule adds.
    """)
    return


@app.cell
def _(CONTOURS, DOME_MIN_KM2, DOME_RULE, HALF_LIFE, P, PIXEL_KM2, RAIN_FLUSH, THRESHOLD, WIND_VENT, con, frame_labels, has_rain, has_wind, hi_q, mo, np, pa, par_cells, pcount, pidx, wx_q):
    import time as _dtime

    _dt0 = _dtime.perf_counter()
    _F, _N = hi_q.shape
    _hi = np.where(hi_q == 255, np.nan, hi_q.astype(np.float32) / 2.0 - 40.0)
    _a = 2 ** (-1 / float(HALF_LIFE))
    _x = np.nan_to_num(np.maximum(_hi - float(THRESHOLD), 0.0))
    if has_wind and WIND_VENT:
        _x *= np.maximum(0.0, 1.0 - float(WIND_VENT) * (wx_q >> 4) / 10.0)
    _L = np.zeros_like(_x)
    for _f in range(1, _F):
        _L[_f] = _a * _L[_f - 1] + (1 - _a) * _x[_f]
        if has_rain and RAIN_FLUSH:
            _mm = (wx_q[_f] & 15) / 2.0
            _L[_f] *= np.where(_mm > 0, 1.0 - float(RAIN_FLUSH) * np.minimum(1.0, _mm / 2.5), 1.0)
    # per pixel: how many contour levels it clears (the sets are nested)
    _K = len(CONTOURS)
    _lv = np.zeros((_F, _N), dtype=np.int8)
    for _k, _c in enumerate(CONTOURS):
        _lv[_L >= _c] = _k + 1
    # THE RULE, per res 6 parent, per frame: count the pixels at each level, decide by
    # any / majority / all, hand the cell's level back to its pixels. "pixel" keeps the
    # per-pixel levels and the cell set is the parents of the member pixels (any).
    _cell_lv = np.zeros((_F, P), dtype=np.int8)
    for _f in range(_F):
        _ok = np.ones(P, dtype=bool)
        for _k in range(_K):
            _cnt = np.bincount(pidx[_lv[_f] > _k], minlength=P)
            if DOME_RULE == "pixel" or DOME_RULE == "any":
                _pass = _cnt > 0
            elif DOME_RULE == "all":
                _pass = _cnt == pcount
            else:
                _pass = _cnt * 2 > pcount
            _ok &= _pass
            _cell_lv[_f, _ok] = _k + 1
        if DOME_RULE != "pixel":
            _lv[_f] = _cell_lv[_f][pidx]
    # pixel area per (frame, level): member pixels x the nominal pixel area
    pix_km2 = np.stack([(_lv > _k).sum(axis=1) * float(PIXEL_KM2) for _k in range(_K)], axis=1)  # (F, K)
    _fi, _ci = np.nonzero(_cell_lv > 0)
    _codes = np.array([int(round(c * 10)) for c in CONTOURS], dtype=np.int16)
    con.register("dome_mask", pa.table({"f": _fi.astype(np.int32), "cell": par_cells[_ci], "lvl": _codes[_cell_lv[_fi, _ci] - 1]}))
    con.register("dome_lvl", pa.table({"lvl": _codes}))
    domes = con.sql(f"""
        WITH u AS (
          SELECT m.f, l.lvl, ST_GeomFromWKB(h3_cells_to_multi_polygon_wkb(list(m.cell))) AS g, count(*) AS ncell
          FROM dome_mask m JOIN dome_lvl l ON m.lvl >= l.lvl GROUP BY m.f, l.lvl),
        b AS (SELECT f, lvl, UNNEST(ST_Dump(g)).geom AS geom FROM u),
        m AS (
          SELECT f, lvl / 10.0 AS level,
                 ST_Area(ST_Transform(geom, 'EPSG:4326', 'EPSG:5070', always_xy := true)) / 1e6 AS km2,
                 ST_X(ST_Centroid(geom)) AS lon, ST_Y(ST_Centroid(geom)) AS lat
          FROM b)
        SELECT * FROM m WHERE km2 >= {float(DOME_MIN_KM2)}
    """).arrow().read_all()
    con.unregister("dome_mask")
    con.unregister("dome_lvl")
    con.register("domes", domes)
    dome_summary = con.sql("""
        SELECT level, count(*) AS blobs, round(max(km2)) AS max_km2,
               arg_max(f, km2) AS peak_frame, round(arg_max(lon, km2), 1) AS peak_lon, round(arg_max(lat, km2), 1) AS peak_lat
        FROM domes GROUP BY level ORDER BY level
    """).arrow().read_all()
    _track_level = float(CONTOURS[1] if len(CONTOURS) > 1 else CONTOURS[0])
    _track_k = CONTOURS.index(_track_level)
    dome_track = con.sql(f"""
        SELECT f, round(km2) AS km2, round(lon, 2) AS lon, round(lat, 2) AS lat,
               (SELECT round(sum(km2)) FROM domes d2 WHERE d2.f = d.f AND d2.level = {_track_level}) AS all_km2
        FROM (SELECT *, row_number() OVER (PARTITION BY f ORDER BY km2 DESC) AS rn FROM domes WHERE level = {_track_level}) d
        WHERE rn = 1 ORDER BY f
    """).arrow().read_all()
    con.unregister("domes")
    _rows = [
        {"level °C": r["level"], "blobs ≥ %.0f km²" % DOME_MIN_KM2: r["blobs"], "largest blob km² (cells)": f"{r['max_km2']:,.0f}",
         "all member px km² then": f"{pix_km2[int(r['peak_frame']), CONTOURS.index(float(r['level']))]:,.0f}",
         "at (UTC)": frame_labels[int(r["peak_frame"])], "centre lat, lon": f"{r['peak_lat']}, {r['peak_lon']}"}
        for r in dome_summary.to_pylist()
    ]
    _trk = [
        {"hour (UTC)": frame_labels[int(r["f"])], "dome km² (cells)": f"{r['km2']:,.0f}", "all blobs km²": f"{r['all_km2']:,.0f}",
         "member px km²": f"{pix_km2[int(r['f']), _track_k]:,.0f}", "centre lat": r["lat"], "centre lon": r["lon"]}
        for r in dome_track.to_pylist()
    ]
    dome_stats = (
        f"domes ({DOME_RULE}): {len(_fi):,} cell-hours over {CONTOURS[0]:g} °C · {domes.num_rows:,} blobs ≥ {DOME_MIN_KM2:g} km² · "
        f"accumulate + rule + dissolve + dump {_dtime.perf_counter() - _dt0:.1f}s"
    )
    mo.vstack([
        mo.md(f"<span style='color:#8b929c;font-size:.85em'>{dome_stats} · accumulator at threshold {THRESHOLD:g} °C, half-life {HALF_LIFE:g} h</span>"),
        mo.ui.table(_rows, selection=None, label="largest blob per level over the film (cell-dissolved area next to the plain pixel area at that hour)"),
        mo.ui.table(_trk, selection=None, page_size=12, label=f"the ≥ {_track_level:g} °C dome, hour by hour: largest blob's area and centroid, every blob's area, member pixels x {PIXEL_KM2:g} km²"),
    ])
    return dome_stats, dome_summary, dome_track, pix_km2


@app.cell
def _(county_stats, frame_stats, mo, pix_stats, read_stats, store_stats):
    mo.md(
        "<br>".join(
            f"<span style='color:#8b929c;font-size:.85em'>{s}</span>"
            for s in (store_stats, county_stats, pix_stats, read_stats, frame_stats)
        )
    )
    return


if __name__ == "__main__":
    app.run()
