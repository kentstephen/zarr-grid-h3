# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
# ]
# ///
"""S2 x LCMS x AEF x MTBS: the ground as a picture beside one H3 fill, one camera.

Two maps in one widget. LEFT: Earth Genome's Sentinel-2 yearly mosaic (true
color, 2022-2025) as tiles the kernel renders from the COGs; it is never
covered. RIGHT: one opaque H3 fill per hexagon over the same camera: the USFS
LCMS Change class for the label year (2022 or 2023), the AlphaEarth
displacement between the two ends of a window YOU set (1 - cos between the
cell's mean 64-vector in the from year and the to year, on viridis), or WHEN
inside that window the embedding first moved past the view's stable baseline
(the year, or never), or MTBS burn severity (the observational record of what burned, folded for the
label year and the year before, since MTBS books the burn year and LCMS the
year it shows). MTBS fire perimeters ride on BOTH panes as a line (the label
year solid, the year before dashed) from the PMTiles on source.coop.
Hover a hexagon on either side and its ring is drawn on both, so the pixels a
cell was folded from are always in view on the left. The S2 year and the
label year are separate controls: a late-season burn is booked by LCMS in the
following year and shows in the following year's mosaic, so the two are
meant to be stepped apart.

The fold is the H3 UDF inside DataFusion (repo rule): every raster's pixels
cross as one Dataset and the cell is the GROUP BY. Nothing is tessellated in
the kernel; the browser gets cell ids and rgba.

Run: uv run marimo edit s2-lcms-aef-mtbs-pair.py

Attribution: LCMS is the USDA Forest Service's (public domain). "The AlphaEarth
Foundations Satellite Embedding dataset is produced by Google and Google
DeepMind." (CC-BY 4.0.) Sentinel-2 yearly mosaics by Earth Genome (Copernicus
Sentinel data). MTBS (USGS / USDA Forest Service, public domain) via Carl
Boettiger's cboettig/fire on source.coop.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import json
    import math
    import os
    import tempfile
    import time
    import traceback
    import urllib.parse
    import urllib.request

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xarray as xr
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from pyproj import Transformer

    import io
    from PIL import Image

    return (
        GeoTIFF,
        Image,
        ObjectStore,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        io,
        json,
        math,
        mo,
        np,
        os,
        pa,
        pq,
        tempfile,
        time,
        traceback,
        traitlets,
        udf,
        urllib,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/zarr-grid-h3/blob/main/s2-lcms-aef-mtbs-pair.py)
    <small>the maps may be more responsive run locally:
    `uv run marimo edit s2-lcms-aef-mtbs-pair.py --sandbox`</small>

    # The S2 hex pair

    **Left**: the Sentinel-2 yearly mosaic, a picture, never covered. **Right**:
    one H3 fill over the same camera. Pan either map and both move. Hover a
    hexagon on either side and its ring is drawn on both.

    - **S2 year** (left header, keys `[` `]`): which mosaic, 2022 to 2024.
    - **label year** (right header, keys `,` `.`): the LCMS Change layer and
      the AlphaEarth window around it, 2022 or 2023.
    - **fill** (right header, keys `1` to `3`): what LCMS says happened · how
      much the AlphaEarth fingerprint changed · the year it changed.
    - **MTBS perimeters** on both panes, gold lines for the label year and the
      year before (key `P` hides them). Hover one and the strip names the fire,
      its ignition date, type and acreage.
    - Click a hexagon for its row. `L` toggles the basemap labels.

    The hexagons fold from zoom 9 up. Below it the right pane shows the LCMS
    raster for the label year and the left keeps the mosaic down to zoom 7.
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # Each source offers what it has (Stephen, 2026-09-01: "it'd be nice just to
    # set the window for the years"): LCMS Change 2022 and 2023 (the years a
    # mosaic exists for), Sentinel-2 yearly mosaics 2022-2025 (2025 is in the
    # bucket, not yet in the STAC), AlphaEarth 2017-2025. The AEF window is its
    # own control, a from year and a to year; it no longer hangs off the label
    # year. It opens at 2020..2023 (what the old Y-2..Y+1 window was for 2022).
    LABEL_YEARS = (2022, 2023)
    S2_YEARS = (2022, 2023, 2024, 2025)
    AEF_YEARS_ALL = tuple(range(2017, 2026))
    AEF_FROM0, AEF_TO0 = 2020, 2023
    LABEL_YEAR0, S2_YEAR0 = 2022, 2022

    # The zoom -> H3 ladder: BASE_RES at ZOOM0, one step finer every PER_RES zoom
    # units, clamped, then coarsened until the view's expected cell count fits
    # CELL_BUDGET. The pane is half the width of the old single map, so the same
    # zoom holds half the cells.
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 10
    CELL_BUDGET = 120_000
    LC_PX_PER_CELL = 30
    LC_MAX_PX = 12_000_000
    MOSAIC_MIN_RES = 11
    AEF_LEVEL_FOR_RES = {5: 7, 6: 7, 7: 5, 8: 4, 9: 3, 10: 1}
    AEF_MAX_FILES = 2500

    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    # the mosaic pyramid ends at z9 (L5, 306 m); z7-8 are rendered from L5 by
    # decimation (a z7 tile reads up to nine 1024 px windows: slow, so no lower)
    S2_TILE_MIN_Z, S2_PYRAMID_Z, S2_TCI_MAX_Z = 7, 9, 14
    # MTBS (cboettig/fire on source.coop): burn severity, one WGS84 COG per year
    # (30 m, nine overviews, nodata 0), and the perimeters as PMTiles
    MTBS_PREFIX = "cboettig/fire/mtbs-severity-1984-2024-conus"
    MTBS_NAME = "mtbs-severity-conus-{year}-cog.tif"
    MTBS_PMTILES = "https://data.source.coop/cboettig/fire/mtbs-perimeters-1984-2024.pmtiles"
    MTBS_LAYER = "mtbs-perimeters-1984-2024"
    MTBS_PX_PER_CELL = 30
    MTBS_MAX_PX = 12_000_000
    # 1 unburned to low, 2 low, 3 moderate, 4 high, 5 increased greenness, 6 mask.
    # An orange lightness ramp (the warm leg a protanope keeps), greenness teal.
    MTBS_CLASSES = {
        1: ("Unburned to low", (255, 236, 170)),
        2: ("Low severity", (250, 190, 90)),
        3: ("Moderate severity", (230, 120, 20)),
        4: ("High severity", (140, 50, 0)),
        5: ("Increased greenness", (0, 163, 152)),
    }

    VIEW_W, VIEW_H = 700, 720  # one pane
    # the strip under the map, minimal (Stephen, 2026-09-01: "it's kind of like
    # machine language to me... comment out a lot of that printout"): the
    # legend, the fire under the pointer and the three-sentence story stay;
    # the small numbers line under the story, the status line (res, fold
    # timings, tile counts) and the keys hint are hidden. Flip to bring them
    # back; the kernel still writes them.
    STRIP_MINIMAL = True
    PAD = 1.3
    SETTLE = 0.35
    HEX_ZOOM = 9.0
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    # home: the northern Sierra (the Dixie Fire, burned Jul-Oct 2021; LCMS books
    # most of it in 2022), zoomed to the first hexagon rung
    HOME = {"longitude": -120.95, "latitude": 40.15, "zoom": 9.2}

    # a cell SAYS CHANGE when at least this share of its LCMS pixels carry a
    # change code (1..14); below it the cell is Stable
    CHG_MIN = 0.05
    # the stable baseline: D0 is the displacement quantile (1 - FA) of the
    # view's stable cells; a cell "moved" above it
    FA = 0.05
    MIN_STABLE_CELLS = 30
    # the MTBS severity fill ("mtbs") is built (mtbs_fold, the frame's columns)
    # but not offered: the perimeter line carries the fire (Stephen, 2026-09-01:
    # "I don't think we need a button for that"). Add "mtbs" back here and set
    # MTBS_FOLD_YEARS to (-1, 0) to wire it in again.
    FILLS = ("lcms", "shift", "when")
    FILL_NAMES = {"lcms": "what LCMS says happened", "shift": "how much the AlphaEarth fingerprint changed",
                  "when": "the year the AlphaEarth fingerprint changed", "mtbs": "MTBS burn severity"}
    FILL_SHORT = {"lcms": "LCMS says", "shift": "AEF changed", "when": "AEF change year", "mtbs": "MTBS burned"}
    MTBS_FOLD_YEARS = ()  # offsets from the label year to fold severity for; () = off
    ALPHA_FILL = 235
    ALPHA_QUIET = 70  # Stable / never: drawn faintly so the grid stays legible
    VIRIDIS = "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725"

    LCMS_PREFIX = "ganzk/lcms/change"
    LCMS_NAME = "LCMS_CONUS_v2025-11_Change_{year}.tif"
    LCMS_NODATA = 255
    LCMS_NPA = 16
    LCMS_STABLE = 15
    LCMS_GROWTH = 14
    LCMS_CRS = "EPSG:5071"
    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-lcms")

    # LCMS Change Level 3, NOT the product's colors (three of those are reds).
    # Blue / orange / purple / yellow and lightness; Stable light grey.
    CLASSES = {
        1: ("Wind", (86, 180, 233)),
        2: ("Hurricane", (0, 114, 178)),
        3: ("Snow or Ice Transition", (235, 240, 255)),
        4: ("Desiccation", (204, 152, 46)),
        5: ("Inundation", (10, 218, 255)),
        6: ("Prescribed Fire", (240, 228, 66)),
        7: ("Wildfire", (230, 110, 0)),
        8: ("Mechanical Land Transformation", (60, 60, 60)),
        9: ("Tree Removal", (175, 222, 28)),
        10: ("Defoliation", (194, 145, 213)),
        11: ("Southern Pine Beetle", (120, 40, 140)),
        12: ("Insect, Disease, or Drought Stress", (243, 146, 104)),
        13: ("Other Loss", (110, 110, 110)),
        14: ("Vegetation Successional Growth", (0, 163, 152)),
        15: ("Stable", (222, 222, 222)),
        16: ("Non-Processing Area Mask", (245, 245, 245)),
    }
    # the `when` fill: the first step whose displacement is above D0. Step 0 is
    # keyed by the YEAR the embedding first moved (the step ending in that
    # year); 2021 blue / 2022 orange / 2023 purple as before, the rest from
    # Okabe-Ito less the red, plus a brown and a near-black. -1 never, -2 no
    # embedding. No red, nothing hangs on red vs green.
    WHEN_RGB = {2018: (86, 180, 233), 2019: (0, 158, 115), 2020: (240, 228, 66), 2021: (0, 114, 178),
                2022: (230, 159, 0), 2023: (204, 121, 167), 2024: (140, 86, 75), 2025: (45, 45, 45),
                -1: (222, 222, 222), -2: (150, 150, 150)}
    return (
        AEF_INDEX_URL,
        AEF_LEVEL_FOR_RES,
        AEF_MAX_FILES,
        AEF_NODATA,
        AEF_PREFIX,
        AEF_FROM0,
        AEF_RES,
        AEF_TO0,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS_ALL,
        ALPHA_FILL,
        ALPHA_QUIET,
        BASE_RES,
        CACHE_DIR,
        CELL_BUDGET,
        CHG_MIN,
        CLASSES,
        FA,
        FILLS,
        FILL_NAMES,
        FILL_SHORT,
        HEX_ZOOM,
        HOME,
        LABELS_SLOT,
        LABEL_YEAR0,
        LABEL_YEARS,
        LCMS_CRS,
        LCMS_GROWTH,
        LCMS_NAME,
        LCMS_NODATA,
        LCMS_NPA,
        LCMS_PREFIX,
        LCMS_STABLE,
        LC_MAX_PX,
        LC_PX_PER_CELL,
        MAX_RES,
        MIN_RES,
        MIN_STABLE_CELLS,
        MOSAIC_MIN_RES,
        MTBS_CLASSES,
        MTBS_FOLD_YEARS,
        MTBS_LAYER,
        MTBS_MAX_PX,
        MTBS_NAME,
        MTBS_PMTILES,
        MTBS_PREFIX,
        MTBS_PX_PER_CELL,
        PAD,
        PER_RES,
        RASTER_TILE,
        S2_COLLECTION,
        S2_PYRAMID_Z,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEAR0,
        S2_YEARS,
        SETTLE,
        STRIP_MINIMAL,
        VIEW_H,
        VIEW_W,
        VIRIDIS,
        WHEN_RGB,
        ZOOM0,
    )


@app.cell
def _(BASE_RES, CELL_BUDGET, MAX_RES, MIN_RES, PAD, PER_RES, VIEW_H, VIEW_W, ZOOM0, math):
    # ---- the camera -> box and res --------------------------------------------
    CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) of ONE pane; the widget reports
        the pane's canvas size (`w`, `h`) with every move."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (
            vs["longitude"] - half_lon,
            _y_to_lat(yc + half_y),
            vs["longitude"] + half_lon,
            _y_to_lat(yc - half_y),
        )

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (max(-179.9, b[0] - dx), max(-85.0, b[1] - dy), min(179.9, b[2] + dx), min(85.0, b[3] + dy))

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box):
        r = max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

    return CELL_KM2, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule). One context, every fold.
    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )
    return (ctx,)


@app.cell
def _(
    CLASSES,
    GeoTIFF,
    Image,
    LCMS_CRS,
    LCMS_GROWTH,
    LCMS_NAME,
    LCMS_NODATA,
    LCMS_NPA,
    LCMS_PREFIX,
    LCMS_STABLE,
    LC_MAX_PX,
    LC_PX_PER_CELL,
    RASTER_TILE,
    S3Store,
    Transformer,
    Window,
    asyncio,
    ctx,
    io,
    math,
    np,
    time,
    xr,
):
    # ---- LCMS Change: one CONUS COG per year on source.coop --------------------
    # One window of one level per fold: the box's edges go to Albers through
    # pyproj, the window is their pixel bounds, the pixel centres come back to
    # lon/lat for the h3 UDF. The level is the one whose pixel gives about
    # LC_PX_PER_CELL pixels per cell. The tile renderer (`lc_tile_png`) draws
    # the label year's raster on the RIGHT pane below HEX_ZOOM.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _fwd = Transformer.from_crs("EPSG:4326", LCMS_CRS, always_xy=True)
    _inv = Transformer.from_crs(LCMS_CRS, "EPSG:4326", always_xy=True)
    _open = {}
    _sem = asyncio.Semaphore(16)
    _win = {}
    _fold_lock = asyncio.Lock()
    _LEVELS = 7  # 30, 60, 120, 240, 480, 960, 1920 m
    _CELL_M2 = {5: 252.9e6, 6: 36.13e6, 7: 5.161e6, 8: 0.7373e6, 9: 0.1053e6, 10: 15050.0, 11: 2150.0, 12: 307.1}
    lc_bounds = (-125.5, 23.5, -66.0, 50.0)
    _cmap = np.zeros((256, 4), np.uint8)
    for _code, (_nm, _rgb) in CLASSES.items():
        _cmap[_code, :3] = _rgb
        _cmap[_code, 3] = 255
    _cmap[LCMS_NPA, 3] = 0
    _cmap[LCMS_NODATA, 3] = 0
    _png_cache = {}

    async def _get(year):
        if year not in _open:
            async with _sem:
                _open[year] = await GeoTIFF.open(f"{LCMS_PREFIX}/{LCMS_NAME.format(year=year)}", store=_store)
        return _open[year]

    def _grid(g, k):
        lv = g if k == 0 else g.overviews[k - 1]
        H, W = lv.shape
        t = g.transform
        return lv, t.a * (g.width / W), t.c, t.f, W, H

    async def lc_window(year, k, box):
        g = await _get(year)
        lv, px, x0, y0, W, H = _grid(g, k)
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 20), np.full(20, E_), np.linspace(E_, W_, 20), np.full(20, W_)])
        lats = np.concatenate([np.full(20, N_), np.linspace(N_, S_, 20), np.full(20, S_), np.linspace(S_, N_, 20)])
        ax, ay = _fwd.transform(lons, lats)
        c0 = max(0, int(math.floor((np.nanmin(ax) - x0) / px)))
        c1 = min(W, int(math.ceil((np.nanmax(ax) - x0) / px)))
        r0 = max(0, int(math.floor((y0 - np.nanmax(ay)) / px)))
        r1 = min(H, int(math.ceil((y0 - np.nanmin(ay)) / px)))
        if c1 <= c0 or r1 <= r0:
            return None
        key = (year, k, r0, r1, c0, c1)
        a = _win.get(key)
        if a is None:
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            a = np.asarray(np.ma.filled(ra.as_masked(), LCMS_NODATA)).reshape(r1 - r0, c1 - c0).astype(np.uint8)
            _win[key] = a
            if len(_win) > 200:
                _win.pop(next(iter(_win)))
        return a, px, x0 + c0 * px, y0 - r0 * px

    async def lc_tile_png(z, x, y, year):
        """RGBA PNG bytes for Web Mercator tile (z, x, y) of the year's LCMS
        Change, or None where every pixel is nodata / mask. The level is the one
        whose pixel is nearest the tile's own (in metres at the tile's latitude)."""
        key = (year, z, x, y)
        if key in _png_cache:
            return _png_cache[key]
        T = RASTER_TILE
        n = 2 ** z
        lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        if lon1 < lc_bounds[0] or lon0 > lc_bounds[2] or lat1 < lc_bounds[1] or lat0 > lc_bounds[3]:
            _png_cache[key] = None
            return None
        m_tile = 2 * math.pi * 6378137.0 / (n * T) * math.cos(math.radians((lat0 + lat1) / 2))
        k = max(0, min(_LEVELS - 1, int(round(math.log2(max(m_tile, 30.0) / 30.0)))))
        got = await lc_window(year, k, (lon0, lat0, lon1, lat1))
        if got is None:
            _png_cache[key] = None
            return None
        arr, px, wx0, wy0 = got
        ys = np.pi * (1 - 2 * (y + (np.arange(T) + 0.5) / T) / n)
        lat_c = np.degrees(np.arctan(np.sinh(ys)))
        lon_c = lon0 + (np.arange(T) + 0.5) * (lon1 - lon0) / T
        LON, LAT = np.meshgrid(lon_c, lat_c)
        AX, AY = _fwd.transform(LON, LAT)
        ci = ((np.asarray(AX) - wx0) / px).astype(np.int64)
        ri = ((wy0 - np.asarray(AY)) / px).astype(np.int64)
        ok = (ci >= 0) & (ci < arr.shape[1]) & (ri >= 0) & (ri < arr.shape[0])
        pxv = np.where(ok, arr[np.clip(ri, 0, arr.shape[0] - 1), np.clip(ci, 0, arr.shape[1] - 1)], LCMS_NODATA)
        rgba = _cmap[pxv]
        if not rgba[..., 3].any():
            _png_cache[key] = None
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png_cache[key] = buf.getvalue()
        if len(_png_cache) > 4000:
            _png_cache.pop(next(iter(_png_cache)))
        return _png_cache[key]

    async def lc_fold(box, res, year):
        """Per res cell over the box: the majority code (`maj`), the share of
        pixels carrying a change code (`p_chg`, 1..14) and the majority AMONG
        the change pixels (`chg`; Stable when there are none). Nodata and the
        Non-Processing mask are left out. Returns (arrow table, stats string)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        pitch = min(1920.0, max(30.0, math.sqrt(_CELL_M2[res] / LC_PX_PER_CELL)))
        k = max(0, min(_LEVELS - 1, int(round(math.log2(pitch / 30.0)))))
        while True:
            got = await lc_window(year, k, box)
            if got is None:
                return None, f"LCMS {year}: nothing under the view (off CONUS)"
            if got[0].size <= LC_MAX_PX or k >= _LEVELS - 1:
                break
            k += 1
        arr, px, wx0, wy0 = got
        h, w = arr.shape
        tr = time.time()
        xs = wx0 + (np.arange(w) + 0.5) * px
        ys = wy0 - (np.arange(h) + 0.5) * px
        X, Y = np.meshgrid(xs, ys)
        lon, lat = _inv.transform(X, Y)
        t1 = time.time()
        async with _fold_lock:
            try:
                ctx.deregister_table("lc")
            except Exception:
                pass
            ctx.from_dataset(
                "lc",
                xr.Dataset(
                    {"cls": (("y", "x"), arr), "lat": (("y", "x"), np.asarray(lat)), "lon": (("y", "x"), np.asarray(lon))},
                    coords={"y": np.arange(h), "x": np.arange(w)},
                ),
                chunks={"y": 512},
            )
            out = ctx.sql(f"""
                WITH c AS (
                    SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, cls, count(*) AS n
                    FROM lc
                    WHERE cls >= 1 AND cls <= {LCMS_STABLE}
                      AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                    GROUP BY 1, 2
                )
                SELECT cell,
                       first_value(cls ORDER BY n DESC, cls ASC) AS maj,
                       first_value(cls ORDER BY (CASE WHEN cls <= {LCMS_GROWTH} THEN n ELSE -1 END) DESC, cls ASC) AS chg,
                       sum(n) AS npx,
                       CAST(sum(CASE WHEN cls <= {LCMS_GROWTH} THEN n ELSE 0 END) AS DOUBLE) / sum(n) AS p_chg,
                       CAST(max(n) AS DOUBLE) / sum(n) AS purity
                FROM c GROUP BY cell
            """).to_arrow_table()
        return out, (
            f"LCMS {year} {w:,}x{h:,} px ({30 * 2 ** k:.0f} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return lc_bounds, lc_fold, lc_tile_png


@app.cell
async def _(
    AEF_INDEX_URL,
    AEF_LEVEL_FOR_RES,
    AEF_MAX_FILES,
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS_ALL,
    CACHE_DIR,
    GeoTIFF,
    MOSAIC_MIN_RES,
    ObjectStore,
    S3Store,
    Transformer,
    Window,
    asyncio,
    ctx,
    duckdb,
    np,
    os,
    pq,
    time,
    xr,
):
    # ---- AlphaEarth: the COG overviews (mosaic past res 10), one fold per year --
    # `aef_fold(box, res, year)` for any year in AEF_YEARS_ALL (2017..2025, the
    # whole run; the window control picks from them); each year has its own COG index
    # slice (cached as parquet under tmp) and its own mosaic time index.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _mstore = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, prefix=AEF_PREFIX)
    _ds = xr.open_zarr(ObjectStore(_mstore, read_only=True), chunks=None, consolidated=False)
    _ti = {y: int(np.where(_ds.time.values == y)[0][0]) for y in AEF_YEARS_ALL}

    os.makedirs(CACHE_DIR, exist_ok=True)
    _IDX, _PATHS, _CRS = {}, {}, {}
    for _y in AEF_YEARS_ALL:
        _idx_path = os.path.join(CACHE_DIR, f"aef_index_{_y}_world.parquet")
        if not os.path.exists(_idx_path):
            _c = duckdb.connect()
            _c.execute("INSTALL httpfs; LOAD httpfs")
            _t = _c.execute(f"""
                SELECT path, crs, utm_west, utm_south, utm_east, utm_north,
                       wgs84_west, wgs84_south, wgs84_east, wgs84_north
                FROM read_parquet('{AEF_INDEX_URL}')
                WHERE year = {_y}
            """).arrow().read_all()
            pq.write_table(_t, _idx_path)
            _c.close()
        _tab = pq.read_table(_idx_path)
        _IDX[_y] = {k: _tab[k].to_numpy() for k in _tab.column_names if k not in ("path", "crs")}
        _PATHS[_y] = _tab["path"].to_pylist()
        _CRS[_y] = _tab["crs"].to_pylist()

    _open = {}
    _sem = asyncio.Semaphore(64)
    _tf_fwd, _tf_inv = {}, {}

    def _tf(crs):
        if crs not in _tf_fwd:
            _tf_fwd[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            _tf_inv[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return _tf_fwd[crs], _tf_inv[crs]

    async def _get(path):
        rel = path.split("source.coop/")[1]
        if rel not in _open:
            async with _sem:
                _open[rel] = await GeoTIFF.open(rel, store=_store)
        return _open[rel]

    async def _read_cog(year, i, li, box):
        """One file's overview window over the box: (int8 (64, h, w), lon, lat)
        or None. Through the file's affine (these COGs are stored south-up)."""
        g = await _get(_PATHS[year][i])
        ov = g.overviews[li]
        H, W = ov.shape
        t = g.transform
        sx, sy = t.a * (g.width / W), t.e * (g.height / H)
        fwd, inv = _tf(_CRS[year][i])
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 5), np.full(5, E_), np.linspace(E_, W_, 5), np.full(5, W_)])
        lats = np.concatenate([np.full(5, N_), np.linspace(N_, S_, 5), np.full(5, S_), np.linspace(S_, N_, 5)])
        ux, uy = fwd.transform(lons, lats)
        cc = (np.asarray(ux) - t.c) / sx
        rr = (np.asarray(uy) - t.f) / sy
        c0 = max(0, int(np.floor(np.nanmin(cc))))
        c1 = min(W, int(np.ceil(np.nanmax(cc))))
        r0 = max(0, int(np.floor(np.nanmin(rr))))
        r1 = min(H, int(np.ceil(np.nanmax(rr))))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await ov.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, r1 - r0, c1 - c0)
        xs = t.c + (np.arange(c0, c1) + 0.5) * sx
        ys = t.f + (np.arange(r0, r1) + 0.5) * sy
        X, Y = np.meshgrid(xs, ys)
        lon, lat = inv.transform(X, Y)
        return a, lon, lat

    _DEQ = ", ".join(f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64))
    _fold_lock = asyncio.Lock()

    async def _fold_rows(res, box, cols, lat, lon):
        W_, S_, E_, N_ = box
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)} | {"lat": (("i",), lat), "lon": (("i",), lon)},
            coords={"i": np.arange(lat.size)},
        )
        async with _fold_lock:
            try:
                ctx.deregister_table("aef")
            except Exception:
                pass
            ctx.from_dataset("aef", ds1, chunks={"i": 262_144})
            return ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS naef, {_DEQ}
                FROM aef
                WHERE e00 != {AEF_NODATA}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()

    async def aef_fold(box, res, year):
        """Mean AlphaEarth vector per res cell over the box for one year.
        Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        if res >= MOSAIC_MIN_RES:
            x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
            y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
            loop = asyncio.get_running_loop()
            ti = _ti[year]
            emb = await loop.run_in_executor(
                None, lambda: _ds.embeddings.isel(time=ti, y=slice(y0, y1), x=slice(x0, x1)).values
            )
            lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
            lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
            LON, LAT = np.meshgrid(lon, lat)
            t1 = time.time()
            out = await _fold_rows(res, box, emb.reshape(64, -1), LAT.ravel(), LON.ravel())
            return out, f"AEF {year} mosaic {t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        li = AEF_LEVEL_FOR_RES[res]
        ix = _IDX[year]
        hit = np.where(
            (ix["wgs84_east"] > W_) & (ix["wgs84_west"] < E_) & (ix["wgs84_north"] > S_) & (ix["wgs84_south"] < N_)
        )[0]
        if len(hit) == 0:
            return None, f"AEF {year}: no COG tiles under the view"
        if len(hit) > AEF_MAX_FILES:
            return None, f"AEF {year}: {len(hit):,} tiles under the view; zoom in"
        parts = await asyncio.gather(*(_read_cog(year, int(i), li, box) for i in hit))
        parts = [p for p in parts if p is not None]
        if not parts:
            return None, f"AEF {year}: nothing read"
        cols = np.concatenate([p[0].reshape(64, -1) for p in parts], axis=1)
        lon = np.concatenate([p[1].ravel() for p in parts])
        lat = np.concatenate([p[2].ravel() for p in parts])
        t1 = time.time()
        out = await _fold_rows(res, box, cols, lat, lon)
        return out, (
            f"AEF {year} ov{li} ({10 * 2 ** (li + 1)} m) {len(parts)} files {cols.shape[1] / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return (aef_fold,)


@app.cell
def _(
    GeoTIFF,
    Image,
    RASTER_TILE,
    S2_COLLECTION,
    S2_PYRAMID_Z,
    S2_STAC,
    S2_TCI_MAX_Z,
    S2_TILE_MIN_Z,
    S3Store,
    Window,
    asyncio,
    io,
    json,
    math,
    np,
    time,
    urllib,
):
    # ---- Sentinel-2 TCI tiles, by YEAR: the left pane ---------------------------
    # STAC once per (year, z9 ancestor tile), every footprint under the tile
    # composited in numpy (black = nodata -> alpha 0; first footprint to paint a
    # pixel wins), one PNG. The year lives in the item id
    # (`10SFJ_2024-01-01_2025-01-01`); the STAC datetime filter does not
    # constrain these items, so it is enforced on the id.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _R = 6378137.0
    _items = {}  # item id -> {tci: path, bbox}
    _boxes = {}  # (year, rounded box) -> item ids
    _open = {}
    _sem = asyncio.Semaphore(32)
    _png = {}  # (year, z, x, y) -> PNG bytes or None
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}

    def _stac(box):
        body = json.dumps({"collections": [S2_COLLECTION], "bbox": list(box), "limit": 100}).encode()
        req = urllib.request.Request(S2_STAC, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["features"]

    async def _s2_items(box, year):
        key = (year, tuple(round(v, 2) for v in box))
        if key not in _boxes:
            loop = asyncio.get_running_loop()
            feats = await loop.run_in_executor(None, _stac, box)
            ids = []
            for f in feats:
                if not f["id"].endswith(f"{year}-01-01_{year + 1}-01-01"):
                    continue
                _items[f["id"]] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox")}
                ids.append(f["id"])
            if not ids and feats:
                # the STAC lags the bucket (2025 is there for every tile round
                # Dixie, uploaded 2026-01-31, and the search does not know it):
                # the same MGRS tile's path with the year swapped, the sibling's
                # bbox; a tile that is not there reads as empty, not an error
                seen = set()
                for f in feats:
                    tile = f["id"].split("_")[0]
                    if tile in seen:
                        continue
                    seen.add(tile)
                    iid = f"{tile}_{year}-01-01_{year + 1}-01-01"
                    base = f["assets"]["TCI"]["href"].split("source.coop/")[1].rsplit("/", 2)[0]
                    _items[iid] = {"tci": f"{base}/{iid}/TCI.tif", "bbox": f.get("bbox")}
                    ids.append(iid)
            _boxes[key] = ids
        return _boxes[key]

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                try:
                    _open[rel] = await GeoTIFF.open(rel, store=_store)
                except Exception:
                    _open[rel] = None  # not in the bucket (a synthesized year): empty
        return _open[rel]

    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    async def _items_for_tile(z, x, y, year):
        # STAC per z9 ancestor tile from z9 up; below it, per the tile itself
        d = max(0, z - S2_PYRAMID_Z)
        ids = await _s2_items(_tile_ll(z - d, x >> d, y >> d), year)
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        out = []
        for i in ids:
            b = _items[i].get("bbox")
            if not b or (b[0] < E_ and b[2] > W_ and b[1] < N_ and b[3] > S_):
                out.append(i)
        return out

    async def s2_tile_png(z, x, y, year):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's TCI mosaic, or
        None (below S2_TILE_MIN_Z, or no footprint under the tile)."""
        key = (year, z, x, y)
        if key in _png:
            return _png[key]
        if z < S2_TILE_MIN_Z or z > S2_TCI_MAX_Z:
            _tstat["blank"] += 1
            return None
        ids = await _items_for_tile(z, x, y, year)
        if not ids:
            _tstat["blank"] += 1
            return None
        t0 = time.time()
        T = RASTER_TILE
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        tx0, ty1 = -world / 2 + x * world / n, world / 2 - y * world / n
        xs = tx0 + (np.arange(T) + 0.5) * tpx
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        out = np.zeros((T, T, 4), np.uint8)
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)  # L5 (306 m) below z9: decimated
        for iid in ids:
            g = await _get(_items[iid]["tci"])
            if g is None:
                continue
            lv = [g, *g.overviews][li]
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            px = (R_ - L) / W
            c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
            r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
            if c1 <= c0 or r1 <= r0:
                continue
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, r1 - r0, c1 - c0)[:3]
            cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
            rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
            okc, okr = (cols >= 0) & (cols < c1 - c0), (rows >= 0) & (rows < r1 - r0)
            rgb = a[:, np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]].transpose(1, 2, 0)
            valid = okr[:, None] & okc[None, :] & (rgb.sum(2) > 0) & (out[..., 3] == 0)
            out[valid, :3] = rgb[valid]
            out[valid, 3] = 255
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(out), mode="RGBA").save(buf, format="PNG")
        _png[key] = buf.getvalue()
        if len(_png) > 6000:
            _png.pop(next(iter(_png)))
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return _png[key]

    def s2_raster_stats():
        return dict(_tstat, cached=len(_png))

    return s2_raster_stats, s2_tile_png


@app.cell
def _(
    GeoTIFF,
    MTBS_MAX_PX,
    MTBS_NAME,
    MTBS_PREFIX,
    MTBS_PX_PER_CELL,
    S3Store,
    Window,
    asyncio,
    ctx,
    math,
    np,
    time,
    xr,
):
    # ---- MTBS burn severity: one WGS84 COG per year (cboettig/fire) --------------
    # The LCMS fold's shape without the Albers leg: the COG is EPSG:4326, so the
    # lon/lat box IS the window. Level k has a pixel of ~30 * 2^k m; the fold
    # reads the one nearest MTBS_PX_PER_CELL pixels per cell. Missing years
    # (CONUS 2004, 2017) are not in the overlap here.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _open = {}
    _sem = asyncio.Semaphore(16)
    _fold_lock = asyncio.Lock()
    _CELL_M2 = {5: 252.9e6, 6: 36.13e6, 7: 5.161e6, 8: 0.7373e6, 9: 0.1053e6, 10: 15050.0, 11: 2150.0, 12: 307.1}

    async def _get(year):
        if year not in _open:
            async with _sem:
                _open[year] = await GeoTIFF.open(f"{MTBS_PREFIX}/{MTBS_NAME.format(year=year)}", store=_store)
        return _open[year]

    async def mtbs_fold(box, res, year):
        """Per res cell over the box: the majority severity among the burned
        pixels (codes 1..5; `sev`), the share of the cell's pixels that carry one
        (`p_burn`). Cells with no burned pixel are absent. (table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        g = await _get(year)
        L, B, R_, T_ = g.bounds
        if E_ <= L or W_ >= R_ or N_ <= B or S_ >= T_:
            return None, f"MTBS {year}: off CONUS"
        levels = [g, *g.overviews]
        px0 = (R_ - L) / g.width  # degrees per native pixel (~30 m)
        pitch = max(30.0, math.sqrt(_CELL_M2[res] / MTBS_PX_PER_CELL))
        k = max(0, min(len(levels) - 1, int(round(math.log2(pitch / 30.0)))))
        while True:
            lv = levels[k]
            H, W = lv.shape
            px = px0 * (g.width / W)
            c0, c1 = max(0, int(math.floor((W_ - L) / px))), min(W, int(math.ceil((E_ - L) / px)))
            r0, r1 = max(0, int(math.floor((T_ - N_) / px))), min(H, int(math.ceil((T_ - S_) / px)))
            if c1 <= c0 or r1 <= r0:
                return None, f"MTBS {year}: nothing under the view"
            if (c1 - c0) * (r1 - r0) <= MTBS_MAX_PX or k >= len(levels) - 1:
                break
            k += 1
        async with _sem:
            ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        arr = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(r1 - r0, c1 - c0).astype(np.uint8)
        tr = time.time()
        if not arr.any():
            return None, f"MTBS {year}: nothing burned under the view ({tr - t0:.1f} s)"
        h, w = arr.shape
        lon = L + (c0 + np.arange(w) + 0.5) * px
        lat = T_ - (r0 + np.arange(h) + 0.5) * px
        LON, LAT = np.meshgrid(lon, lat)
        async with _fold_lock:
            try:
                ctx.deregister_table("mtbs")
            except Exception:
                pass
            ctx.from_dataset(
                "mtbs",
                xr.Dataset(
                    {"sev": (("y", "x"), arr), "lat": (("y", "x"), LAT), "lon": (("y", "x"), LON)},
                    coords={"y": np.arange(h), "x": np.arange(w)},
                ),
                chunks={"y": 512},
            )
            out = ctx.sql(f"""
                WITH c AS (
                    SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, sev, count(*) AS n
                    FROM mtbs
                    WHERE lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                    GROUP BY 1, 2
                )
                SELECT cell,
                       first_value(sev ORDER BY (CASE WHEN sev BETWEEN 1 AND 5 THEN n ELSE -1 END) DESC, sev ASC) AS sev,
                       CAST(sum(CASE WHEN sev BETWEEN 1 AND 5 THEN n ELSE 0 END) AS DOUBLE) / sum(n) AS p_burn
                FROM c GROUP BY cell
                HAVING sum(CASE WHEN sev BETWEEN 1 AND 5 THEN n ELSE 0 END) > 0
            """).to_arrow_table()
        return out, (
            f"MTBS {year} {w:,}x{h:,} px ({30 * 2 ** k:.0f} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - tr:.1f} s"
        )

    return (mtbs_fold,)


@app.cell
def _(
    ALPHA_FILL,
    ALPHA_QUIET,
    CHG_MIN,
    CLASSES,
    FA,
    LCMS_STABLE,
    MIN_STABLE_CELLS,
    MTBS_CLASSES,
    VIRIDIS,
    WHEN_RGB,
    duckdb,
    np,
    pa,
):
    # ---- a FRAME: the join, the displacement, the stable baseline, the three fills --
    _stops = np.array([[int(VIRIDIS[i + j:i + j + 2], 16) for j in (0, 2, 4)] for i in range(0, len(VIRIDIS), 6)], np.float64)
    RAMP = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    RAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in RAMP[i]) for i in range(0, 256, 17)]
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3")
    _E = [f"e{i:02d}" for i in range(64)]
    _GREY = np.array([128, 128, 128], np.uint8)

    def build_frame(lc_cells, aef_by_year, y0, y1, mtbs_by_year=None):
        """Join the label year's LCMS fold with each AlphaEarth year in the window
        y0..y1 (LEFT: a cell keeps its LCMS word with or without an embedding).
        `disp` is the displacement between the two ENDS (1 - cos of the y0 and
        y1 vectors): the "AEF changed" fill. The consecutive steps inside the
        window give D0 (the stable cells' quantile of their largest step) and
        `when`, the year of the first step above D0: the "AEF change year"
        fill. The MTBS folds join the same way; a cell burned in both keeps the
        later year."""
        con.register("lc_cells", lc_cells)
        years = [y for y in range(y0, y1 + 1) if aef_by_year.get(y) is not None]
        sel = ["l.*"]
        joins = []
        for y in years:
            con.register(f"aef_{y}", aef_by_year[y])
            sel += [f"a{y}.{e} AS {e}_{y}" for e in _E]
            joins.append(f"LEFT JOIN aef_{y} a{y} USING (cell)")
        j = con.execute(f"SELECT {', '.join(sel)} FROM lc_cells l {' '.join(joins)} ORDER BY cell").arrow().read_all()
        n = j.num_rows
        mtbs_sev = np.zeros(n, np.int64)
        mtbs_year = np.zeros(n, np.int64)
        for y in sorted((mtbs_by_year or {}).keys()):
            tab = mtbs_by_year[y]
            if tab is None or tab.num_rows == 0:
                continue
            con.register("mtbs_y", tab)
            con.register("j_cells", j)
            m = con.execute("SELECT m.sev FROM j_cells j LEFT JOIN mtbs_y m USING (cell) ORDER BY j.cell").arrow().read_all()
            sv = m["sev"].to_numpy(zero_copy_only=False).astype(np.float64)  # nulls -> NaN
            has = ~np.isnan(sv)
            mtbs_sev = np.where(has, np.nan_to_num(sv).astype(np.int64), mtbs_sev)
            mtbs_year = np.where(has, y, mtbs_year)
        maj = j["maj"].to_numpy().astype(np.int64)
        chg = j["chg"].to_numpy().astype(np.int64)
        p_chg = j["p_chg"].to_numpy().astype(np.float32)
        says = p_chg >= CHG_MIN
        cls = np.where(says, chg, LCMS_STABLE).astype(np.int64)

        def _V(y):
            V = np.stack([j[f"{e}_{y}"].to_numpy(zero_copy_only=False) for e in _E], axis=1).astype(np.float32)
            nrm = np.linalg.norm(V, axis=1)
            V = V / np.maximum(nrm, 1e-9)[:, None]
            V[~np.isfinite(nrm) | (nrm == 0)] = np.nan
            return V

        # step k compares years[k] with years[k+1] (consecutive years present)
        Vs = {y: _V(y) for y in years} if n else {}
        step_years = [(years[k], years[k + 1]) for k in range(len(years) - 1)]
        steps = np.full((max(1, len(step_years)), n), np.nan, np.float32)
        for k, (ya, yb) in enumerate(step_years):
            steps[k] = (1.0 - np.einsum("ij,ij->i", Vs[ya], Vs[yb])).astype(np.float32)
        # the ends: what the fingerprint did between y0 and y1, whatever the path
        if y0 in Vs and y1 in Vs and y0 != y1:
            disp = (1.0 - np.einsum("ij,ij->i", Vs[y0], Vs[y1])).astype(np.float32)
        else:
            disp = np.full(n, np.nan, np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            disp_max = np.nanmax(steps, axis=0) if n else np.zeros(0, np.float32)
        scored = ~np.isnan(disp)
        stepped = ~np.isnan(disp_max)
        stable_ok = stepped & ~says
        when = np.full(n, -2, np.int64)  # unscored
        if stable_ok.sum() >= MIN_STABLE_CELLS:
            ds = disp_max[stable_ok].astype(np.float64)
            D0 = float(np.quantile(ds, 1 - FA))
            above = steps > D0  # NaN > D0 is False
            first = above.argmax(axis=0)
            yrs = np.array([yb for _, yb in step_years] or [-1], np.int64)
            when = np.where(above.any(axis=0), yrs[first], np.where(stepped, -1, -2)).astype(np.int64)
        else:
            D0 = float("nan")
        moved = when >= 0
        when_name = {
            -1: f"AlphaEarth saw no change {y0} to {y1} (every step under the quiet level)",
            -2: "no AlphaEarth embedding here",
        }
        for ya, yb in step_years:
            when_name[yb] = f"AlphaEarth changed in {yb} (its {ya} vs {yb} fingerprints)"
        cells = pa.table({
            "cell": j["cell"],
            "cls": pa.array(cls.astype(np.uint8)),
            "name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in cls]),
            "maj": pa.array(maj.astype(np.uint8)),
            "maj_name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in maj]),
            "p_chg": pa.array(p_chg),
            "purity": j["purity"],
            "disp": pa.array(disp.astype(np.float32)),
            "disp_max": pa.array(disp_max.astype(np.float32)),
            **{f"step_{yb}": pa.array(steps[k]) for k, (_, yb) in enumerate(step_years)},
            "moved": pa.array(moved),
            "when": pa.array(when.astype(np.int16)),
            "when_name": pa.array([when_name[int(w)] for w in when]),
            "mtbs_sev": pa.array(mtbs_sev.astype(np.int8)),
            "mtbs_name": pa.array([MTBS_CLASSES.get(int(v), ("not mapped burned",))[0] for v in mtbs_sev]),
            "mtbs_year": pa.array(mtbs_year.astype(np.int16)),
        })
        cellid = cells["cell"].to_numpy().astype(np.uint64)
        ok = disp[scored]
        if len(ok) >= 2:
            lo, hi = (float(q) for q in np.percentile(ok, [2, 98]))
            if hi <= lo:
                hi = lo + 1e-6
        else:
            lo, hi = 0.0, 1.0
        rgb_cls = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
        _t = np.clip((np.where(scored, disp, lo) - lo) / (hi - lo), 0, 1)
        rgb_shift = np.where(scored[:, None], RAMP[(_t * 255).round().astype(np.int64)], _GREY).astype(np.uint8)
        rgb_when = np.array([WHEN_RGB.get(int(w), (45, 45, 45)) for w in when], np.uint8) if n else np.zeros((0, 3), np.uint8)
        rgb_mtbs = np.array([MTBS_CLASSES.get(int(v), ("?", (222, 222, 222)))[1] for v in mtbs_sev], np.uint8) if n else np.zeros((0, 3), np.uint8)
        burned = mtbs_sev > 0

        def fill(kind, hit=None):
            """(N, 4) uint8 rgba: the right pane's getFillColor. The picked cell
            keeps its color (the stroke is the widget's, gold, on both panes)."""
            if kind == "mtbs":
                c = rgb_mtbs
                a = np.where(burned, ALPHA_FILL, ALPHA_QUIET)
            elif kind == "shift":
                c = rgb_shift
                a = np.where(scored, ALPHA_FILL, ALPHA_QUIET)
            elif kind == "when":
                c = rgb_when
                a = np.where(when >= 0, ALPHA_FILL, ALPHA_QUIET)
            else:
                c = rgb_cls
                a = np.where(says, ALPHA_FILL, ALPHA_QUIET)
            return np.ascontiguousarray(np.concatenate([c, a[:, None].astype(np.uint8)], axis=1)).astype(np.uint8)

        def legend(kind):
            tot = max(1, n)
            if kind == "mtbs":
                items = []
                for code in (4, 3, 2, 1, 5):
                    for y in sorted(set(int(v) for v in mtbs_year[mtbs_sev == code])):
                        m = (mtbs_sev == code) & (mtbs_year == y)
                        items.append({"name": f"{MTBS_CLASSES[code][0]} {y}", "hex": "#%02x%02x%02x" % MTBS_CLASSES[code][1],
                                      "pct": round(100 * int(m.sum()) / tot, 1)})
                if not burned.any():
                    items.append({"name": "no MTBS burn in this view for these years", "hex": "#dedede"})
                return items
            if kind == "shift":
                return [{"ramp": RAMP_HEX, "lo": f"{y0} to {y1} shift {lo:.3f}", "hi": f"{hi:.3f}",
                         "title": f"1 - cos between the cell's AlphaEarth vectors in {y0} and {y1} (the two ends of the window, whatever happened between), stretched to this view's p2-p98"}]
            if kind == "when":
                items = []
                for w in [yb for _, yb in step_years] + [-1, -2]:
                    m = when == w
                    if m.any():
                        items.append({"name": when_name[w], "hex": "#%02x%02x%02x" % WHEN_RGB.get(w, (45, 45, 45)), "pct": round(100 * int(m.sum()) / tot, 1)})
                # the quiet level as a legend line, off with STRIP_MINIMAL (the
                # status line carries it too): keep it in the ramp's title instead
                # if not np.isnan(D0):
                #     items.append({"name": f"quiet level D0 {D0:.3f} (the stable cells' {100 * (1 - FA):.0f}th percentile of their largest step)", "hex": "#ffffff"})
                return items
            items = []
            cc, cn = np.unique(cls, return_counts=True)
            for c, k in sorted(zip(cc, cn), key=lambda t: -t[1]):
                nm, rgb = CLASSES.get(int(c), ("?", (128, 128, 128)))
                items.append({"name": nm, "hex": "#%02x%02x%02x" % rgb, "pct": round(100 * int(k) / tot, 1)})
            return items

        n_moved = int(moved.sum())
        score = (
            f"AEF {y0}..{y1} · D0 {D0:.3f} · {n_moved:,} of {int(stepped.sum()):,} scored cells moved" if not np.isnan(D0)
            else f"AEF {y0}..{y1} unscored (no embedding, one year only, or too few stable cells)"
        )
        return {"cells": cells, "cellid": cellid, "cls": cls, "disp": disp, "when": when, "years": years,
                "y0": y0, "y1": y1, "steps": steps, "step_years": step_years,
                "D0": D0, "fill": fill, "legend": legend, "score": score, "n_burned": int(burned.sum())}

    return build_frame, con


@app.cell
def _(anywidget, asyncio, traitlets):
    class PairMap(anywidget.AnyWidget):
        """Two maplibre maps in a row, one camera. LEFT: the S2 mosaic as tiles the
        kernel renders (custom messages, PNG bytes back), keyed by year. RIGHT:
        an H3HexagonLayer (highPrecision) from cell ids + rgba. Hover on either
        pane: h3-js cell at the frame's res, its ring drawn on BOTH panes.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `config`
        (JSON), `status` / `panel` / `legend` (strings for the strip).
        Browser -> kernel: `view` (JSON lon/lat/zoom + the pane's w/h on every
        moveend), `pick` (JSON: the clicked cell as hex, or null), `ctl` (JSON:
        s2 year, label year, fill, labels)."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("[]").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (src, z, x, y, year) -> PNG bytes or None; src "s2" | "lcms"
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c):
            """A FAILURE IS AN ERROR, never an empty tile (deck caches an empty
            tile as loaded and the area stays blank for good)."""
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            try:
                png = await self.tile_fn(c.get("src", "s2"), int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]))
            except Exception as e:
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@3.2.1";

        // the PMTiles protocol, registered once per page (maplibre keeps a global table)
        if (!window.__spPM) { window.__spPM = new PMProtocol(); maplibregl.addProtocol("pmtiles", window.__spPM.tile); }

        const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        function copyOf(u8) { return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength); }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const font = "font:12px ui-sans-serif,system-ui,sans-serif";
          const mono = "font:11px ui-monospace,Menlo,monospace";
          const root = document.createElement("div");
          root.className = "sp-root";
          root.style.cssText = "width:100%;background:#fff;color:#222;" + font;
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:4px;width:100%";
          const mkPane = (side) => {
            const pane = document.createElement("div");
            pane.className = "sp-pane sp-" + side;
            pane.style.cssText = "position:relative;flex:1 1 0;min-width:0;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
            const mapEl = document.createElement("div");
            mapEl.className = "sp-map";
            mapEl.style.cssText = "position:absolute;inset:0";
            const head = document.createElement("div");
            head.className = "sp-head";
            // a small card, two rows at most, never into maplibre's control column
            head.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;display:flex;flex-direction:column;gap:.3rem;align-items:flex-start;" +
              "max-width:calc(100% - 72px);background:rgba(255,255,255,.94);color:#1d1d1b;padding:4px 8px;border-radius:6px;" +
              "box-shadow:0 1px 3px rgba(0,0,0,.18);white-space:nowrap;font-variant-numeric:tabular-nums";
            pane.append(mapEl, head);
            return {pane, mapEl, head};
          };
          const L = mkPane("left"), R = mkPane("right");
          row.append(L.pane, R.pane);
          const strip = document.createElement("div");
          strip.style.cssText = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";  // rewritten by paneHeight (stripCss) in full screen
          const status = document.createElement("div");
          status.className = "sp-status";
          status.style.cssText = "font:14px ui-sans-serif,system-ui,sans-serif;color:#444;white-space:pre-wrap";
          const legend = document.createElement("div");
          legend.className = "sp-legend";
          legend.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          const panel = document.createElement("div");
          panel.className = "sp-panel";
          panel.style.cssText = "font-size:14px";
          // the fire under the pointer (the perimeter's own attributes, no kernel)
          const fire = document.createElement("div");
          fire.className = "sp-fire";
          fire.style.cssText = "font-size:14px;min-height:1.2em;color:#7a5a00";
          strip.append(legend, fire, panel, status);
          status.hidden = !!cfg.minimal;  // STRIP_MINIMAL: see say()
          root.append(row, strip);
          el.append(css, root);

          // ---- the controls: the pane headers ------------------------------
          const ACCENT = "#2a5db0";
          const btnCss = font + ";padding:.15rem .55rem;border:0;background:transparent;color:#1d1d1b;cursor:pointer;line-height:1.4;font-variant-numeric:tabular-nums";
          const onCss = (b, on) => { b.style.background = on ? ACCENT : "transparent"; b.style.color = on ? "#fff" : "#1d1d1b"; };
          let s2y = cfg.s2_year, ly = cfg.label_year, fill = cfg.fill || "lcms", labelsOn = cfg.labels !== false;
          let perimsOn = cfg.perims !== false;
          let y0 = cfg.aef_from, y1 = cfg.aef_to;
          const send = (act) => {
            model.set("ctl", JSON.stringify({act, s2y, ly, fill, y0, y1, labels: labelsOn, perims: perimsOn, n: Date.now()}));
            model.save_changes();
          };
          // an eyebrow label + a segmented control (joined buttons, one border)
          const mkGroup = (head, title, values, get, set, act, cls, isOn) => {
            const wrap = document.createElement("span");
            wrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const lab = document.createElement("span"); lab.textContent = title;
            lab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
            const seg = document.createElement("span");
            seg.style.cssText = "display:inline-flex;border:1px solid rgba(29,29,27,.28);border-radius:5px;overflow:hidden";
            const btns = values.map((v, i) => {
              const b = document.createElement("button"); b.textContent = String(v.label != null ? v.label : v); b.style.cssText = btnCss;
              if (i) b.style.borderLeft = "1px solid rgba(29,29,27,.18)";
              b.className = cls; b.dataset.value = String(v.value != null ? v.value : v); if (v.title) b.title = v.title;
              b.onclick = () => { set(v.value != null ? v.value : v); style(); send(act); };
              seg.appendChild(b); return b;
            });
            wrap.append(lab, seg);
            rowOf(head).appendChild(wrap);
            const style = () => btns.forEach((b) => onCss(b, isOn ? isOn(b) : b.dataset.value === String(get())));
            style();
            return style;
          };
          // the card is a column of shrink-wrapped rows; groups land in the last row
          const newRow = (head) => { const r = document.createElement("span"); r.style.cssText = "display:inline-flex;gap:.6rem;align-items:center"; head.appendChild(r); return r; };
          const rowOf = (head) => head.lastElementChild || newRow(head);
          const rowBreak = (head) => { newRow(head); };
          const styleS2 = mkGroup(L.head, "S2", cfg.s2_years || [], () => s2y, (v) => { s2y = v; }, "s2", "sp-s2y");
          const styleLy = mkGroup(R.head, "LCMS", cfg.label_years || [], () => ly, (v) => { ly = v; }, "label", "sp-ly");
          const fills = (cfg.fills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          // LCMS and FILL share the first row, the AEF slider has the second (Stephen, 2026-09-01: "two lines instead of 3")
          const styleFill = mkGroup(R.head, "fill", fills, () => fill, (v) => { fill = v; }, "fill", "sp-fill");
          // the AlphaEarth window: a stepped two-handle slider (Stephen,
          // 2026-09-01: "simple and intuitive... a stepped two way slider").
          // Two range inputs on one track; the handle you grab moves, the
          // other stays, and they never cross. The kernel hears the release
          // (change), not every notch (input), so a drag is one fold.
          rowBreak(R.head);
          const aefYears = cfg.aef_years || [];
          const sty = document.createElement("style");
          sty.textContent = [
            ".sp-range{position:relative;width:300px;height:30px}",
            ".sp-range input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}",
            ".sp-range input:focus{outline:none}",
            ".sp-range input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-range input::-moz-range-track{background:none;height:22px}",
            ".sp-range input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range input::-moz-range-thumb{pointer-events:auto;width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-range .spn{position:absolute;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            ".sp-range .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-range .tks span{width:0;display:flex;justify-content:center}",
          ].join("\n");
          el.appendChild(sty);
          const aefWrap = document.createElement("span");
          aefWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const aefLab = document.createElement("span"); aefLab.textContent = "AEF";
          aefLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const rng = document.createElement("span"); rng.className = "sp-range";
          const trk = document.createElement("span"); trk.className = "trk";
          const spn = document.createElement("span"); spn.className = "spn";
          const tks = document.createElement("span"); tks.className = "tks";
          for (const y of aefYears) { const t = document.createElement("span"); const i = document.createElement("i"); i.style.fontStyle = "normal"; i.textContent = String(y); t.appendChild(i); tks.appendChild(t); }
          const mkRange = () => {
            const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = Math.max(0, aefYears.length - 1); r.step = 1;
            r.title = "AlphaEarth window: drag either end (whole years); release to fold"; return r;
          };
          const rFrom = mkRange(), rTo = mkRange();
          const aefTxt = document.createElement("span");
          aefTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:6.5em";
          rng.append(trk, spn, tks, rFrom, rTo);
          aefWrap.append(aefLab, rng, aefTxt);
          rowOf(R.head).appendChild(aefWrap);
          const styleAef = () => {
            const i0 = Math.max(0, aefYears.indexOf(y0)), i1 = Math.max(0, aefYears.indexOf(y1)), n = Math.max(1, aefYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            // the higher handle sits on top so it can always be grabbed at the far end
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = rng.clientWidth - 16;
            spn.style.left = (8 + usable * i0 / n) + "px"; spn.style.width = (usable * (i1 - i0) / n) + "px";
            aefTxt.textContent = y0 + " to " + y1;
          };
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(aefYears.length - 1, b);
            y0 = aefYears[a]; y1 = aefYears[b]; styleAef();
          };
          rFrom.addEventListener("input", () => onDrag("from"));
          rTo.addEventListener("input", () => onDrag("to"));
          const aefRelease = () => { if (y0 !== aefSent[0] || y1 !== aefSent[1]) { aefSent = [y0, y1]; send("aef"); } };
          let aefSent = [y0, y1];
          rFrom.addEventListener("change", aefRelease);
          rTo.addEventListener("change", aefRelease);
          setTimeout(styleAef, 0);
          try { new ResizeObserver(styleAef).observe(rng); } catch (e) {}  // the lit span is in px: redo it when the track gets its width
          // full screen. The widget lives in a shadow root, so the document's
          // fullscreenElement is the shadow HOST, never our root: walk down the
          // shadow roots before comparing (the deck notebook's realFs), or the
          // full-screen layout never applies and the strip is clipped at the
          // viewport (Stephen, 2026-09-01: "we still have that problem with the
          // strip"). In full screen the panes take the whole viewport and the
          // strip floats over their foot, translucent, scrolling inside itself
          // past 45vh, the way the deck notebooks' HUD does.
          const isFull = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement) fe = fe.shadowRoot.fullscreenElement;
            return fe === root;
          };
          const stripCss = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";
          const paneHeight = () => {
            const full = isFull();
            root.style.position = full ? "relative" : "";
            root.style.height = full ? "100vh" : "";
            root.style.boxSizing = "border-box";
            for (const pn of [L.pane, R.pane]) pn.style.height = full ? "100vh" : (cfg.height || 720) + "px";
            strip.style.cssText = full
              ? stripCss + ";position:absolute;left:0;right:0;bottom:0;z-index:30;background:rgba(255,255,255,.94);max-height:45vh;overflow-y:auto;box-sizing:border-box;box-shadow:0 -1px 4px rgba(0,0,0,.18)"
              : stripCss;
            styleAef();
          };
          const toggleFull = () => {
            if (isFull()) document.exitFullscreen();
            else root.requestFullscreen().catch((e) => say("fullscreen: " + e.message));
          };
          document.addEventListener("fullscreenchange", () => { setTimeout(paneHeight, 30); });
          window.addEventListener("resize", () => { paneHeight(); });
          const hint = document.createElement("div");
          hint.style.cssText = mono + ";opacity:.55";
          hint.textContent = "keys: [ ] S2 year · , . label year · 1-3 fill · - = AEF from · _ + AEF to · P perimeters · L labels · F full screen · hover a perimeter for the fire · click a hexagon for its row";
          hint.style.color = "#666";
          strip.appendChild(hint);
          hint.hidden = !!cfg.minimal;
          const step = (arr, cur, d) => { const i = arr.indexOf(cur); return arr[Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i) + d))]; };
          root.tabIndex = 0;
          root.addEventListener("keydown", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            const k = e.key;
            if (k === "[" || k === "]") { s2y = step(cfg.s2_years || [], s2y, k === "]" ? 1 : -1); styleS2(); send("s2"); }
            else if (k === "," || k === ".") { ly = step(cfg.label_years || [], ly, k === "." ? 1 : -1); styleLy(); send("label"); }
            else if (k >= "1" && k <= "9") { const f = fills[Number(k) - 1]; if (f) { fill = f.value; styleFill(); send("fill"); } }
            else if (k === "-" || k === "=") { const v = step(aefYears, y0, k === "=" ? 1 : -1); if (v < y1) { y0 = v; styleAef(); aefRelease(); } }
            else if (k === "_" || k === "+") { const v = step(aefYears, y1, k === "+" ? 1 : -1); if (v > y0) { y1 = v; styleAef(); aefRelease(); } }
            else if (k === "l" || k === "L") { labelsOn = !labelsOn; labels(labelsOn); send("labels"); }
            else if (k === "p" || k === "P") { perimsOn = !perimsOn; perims(); send("perims"); }
            else if (k === "f" || k === "F") { toggleFull(); }
            else return;
            e.preventDefault();
          });

          // STRIP_MINIMAL: the status line shows only while a fold is running,
          // after a failure, or when the hexagons are off for zoom (Stephen,
          // 2026-09-01: "we need some status like folding if it is"); the
          // timings and tile counts of a finished fold stay hidden
          const say = (t) => {
            status.textContent = t || "";
            if (cfg.minimal) status.hidden = !/folding|failed|zoom in past/.test(t || "");
          };
          const renderLegend = () => {
            legend.replaceChildren();
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); } catch (e) { items = []; }
            for (const it of items) {
              const s = document.createElement("span");
              s.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
              if (it.ramp) {
                const bar = document.createElement("span");
                bar.style.cssText = "display:inline-block;width:11rem;height:12px;border-radius:2px;background:linear-gradient(90deg," + it.ramp.join(",") + ")";
                const lo = document.createElement("span"); lo.textContent = it.lo; lo.style.opacity = ".75";
                const hi = document.createElement("span"); hi.textContent = it.hi; hi.style.opacity = ".75";
                s.append(lo, bar, hi); s.title = it.title || "";
              } else {
                const chip = document.createElement("span");
                chip.style.cssText = "display:inline-block;width:12px;height:12px;border-radius:2px;background:" + it.hex;
                const t = document.createElement("span"); t.textContent = it.name + (it.pct != null ? " " + it.pct + "%" : "");
                s.append(chip, t);
              }
              legend.appendChild(s);
            }
          };
          model.on("change:status", () => say(model.get("status")));
          model.on("change:panel", () => { panel.innerHTML = model.get("panel") || ""; });
          model.on("change:legend", renderLegend);

          // ---- the data ----------------------------------------------------
          let hexes = [], N = 0, colors = null, res = -1, hexIndex = new Map(), dataObj = null;
          const raw = {cells: null, colors: null};
          const grab = (k) => {
            try { const u8 = bytesOf(model.get(k)); raw[k] = u8 && u8.length ? copyOf(u8) : null; }
            catch (e) { raw[k] = null; say("grab " + k + ": " + e.message); }
          };
          function loadCells() {
            const buf = raw.cells;
            if (!buf || !buf.byteLength) { hexes = []; N = 0; hexIndex = new Map(); res = -1; return; }
            const ids = new BigUint64Array(buf);
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
          }
          function loadAttrs() {
            const c8 = raw.colors;
            colors = c8 && c8.byteLength === N * 4 ? new Uint8Array(c8) : null;
            dataObj = N && colors ? {length: N} : null;
          }

          // ---- S2 tiles: ask the kernel ------------------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (msg, buffers) => {
            if (!msg || msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.err) { tstat.err++; say("tile: " + msg.err); p.reject(new Error(msg.err)); return; }
            if (msg.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(
              (b) => { tstat.got++; p.resolve(b); },
              (e) => { tstat.err++; p.reject(e instanceof Error ? e : new Error("decode")); });
          });
          const getTileDataFor = (src, year) => ({index, signal}) => new Promise((resolve, reject) => {
            const id = ++tseq;
            tstat.asked++;
            pending.set(id, {resolve, reject});
            model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => {
              pending.delete(id); tstat.abort++;
              const e = new Error("aborted"); e.name = "AbortError"; reject(e);
            });
          });

          // ---- the layers ---------------------------------------------------
          let mapL = null, mapR = null, ovL = null, ovR = null;
          let hover = null;  // the hovered cell (hex string), mirrored on both panes
          // deck's layers go under the MTBS perimeter lines once those exist
          // (they are added before the first update), else under the labels
          const labelSlot = () => cfg.labels_slot || "watername_ocean";
          const slotFor = (m) => (m && m.getLayer && m.getLayer("mtbs-prev")) ? "mtbs-prev" : labelSlot();
          let slot = labelSlot;  // rebound per pane inside layersLeft / layersRight
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const outline = (id, h, color, width) => {
            const r = h ? ring(h) : null;
            if (!r) return null;
            return new PathLayer({id, data: [r], getPath: (d) => d, getColor: color,
              widthUnits: "pixels", getWidth: width, widthMinPixels: 1, beforeId: slot()});
          };
          const mkRaster = (src, year, maxZ, extent, visible) => new TileLayer({
            id: src + "-" + year,
            getTileData: getTileDataFor(src, year),
            onTileError: (e) => { if (!e || e.name !== "AbortError") say(src + " tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256,
            minZoom: 0, maxZoom: maxZ,
            extent: extent || null,
            visible: visible,
            refinementStrategy: "best-available",
            beforeId: slot(),
            renderSubLayers: (p) => {
              if (!p.data) return null;
              const {west, south, east, north} = p.tile.bbox;
              return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
            },
          });
          const hexZoomOk = () => !!mapR && mapR.getZoom() >= (cfg.hex_zoom || 9);
          function layersLeft() {
            slot = () => slotFor(mapL);
            const out = [];
            out.push(mkRaster("s2", cfg.s2_year, 14, null, true));
            const h = outline("hover-l", hover, [255, 255, 255, 255], 2);
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked-l", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function layersRight() {
            slot = () => slotFor(mapR);
            const out = [];
            // below the hexagon zoom the label year's LCMS raster stands in
            out.push(mkRaster("lcms", cfg.label_year, 13, cfg.extent || null, !hexZoomOk() || !dataObj));
            if (dataObj) out.push(new H3HexagonLayer({
              id: "hexes",
              data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj], getHexagon: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              pickable: false,
              beforeId: slot(),
            }));
            const h = outline("hover-r", hover, [255, 255, 255, 255], 2);
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked-r", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          // MTBS perimeters: a maplibre vector source per map from the PMTiles,
          // two line layers (the label year solid, the year before dashed),
          // filtered on the ignition date's year, under the basemap labels
          const PER = cfg.perims_src, PLAYER = cfg.perims_layer;
          const perimFilter = (yr) => ["==", ["slice", ["get", "Ig_Date"], 0, 4], String(yr)];
          function addPerims(m) {
            if (!PER || m.getSource("mtbs")) return;
            try {
              m.addSource("mtbs", {type: "vector", url: "pmtiles://" + PER});
              const before = m.getLayer(labelSlot()) ? labelSlot() : undefined;
              m.addLayer({id: "mtbs-hit", type: "fill", source: "mtbs", "source-layer": PLAYER,
                filter: ["any", perimFilter(cfg.label_year - 1), perimFilter(cfg.label_year)],
                paint: {"fill-color": "#000", "fill-opacity": 0}}, before);
              m.addLayer({id: "mtbs-prev", type: "line", source: "mtbs", "source-layer": PLAYER,
                filter: perimFilter(cfg.label_year - 1),
                paint: {"line-color": "#ffc828", "line-width": 1.6}}, before);
              m.addLayer({id: "mtbs-cur", type: "line", source: "mtbs", "source-layer": PLAYER,
                filter: perimFilter(cfg.label_year),
                paint: {"line-color": "#ffc828", "line-width": 2.2}}, before);
            } catch (e) { say("perimeters: " + e.message); }
          }
          function perims() {
            for (const m of [mapL, mapR]) {
              if (!m || !m.getSource("mtbs")) continue;
              m.setFilter("mtbs-prev", perimFilter(cfg.label_year - 1));
              m.setFilter("mtbs-cur", perimFilter(cfg.label_year));
              m.setFilter("mtbs-hit", ["any", perimFilter(cfg.label_year - 1), perimFilter(cfg.label_year)]);
              for (const id of ["mtbs-prev", "mtbs-cur", "mtbs-hit"]) m.setLayoutProperty(id, "visibility", perimsOn ? "visible" : "none");
            }
          }
          const firesAt = (m, pt) => {
            if (!m || !m.getSource("mtbs")) return [];
            try {
              return m.queryRenderedFeatures(pt, {layers: ["mtbs-hit"]}).map((f) => ({
                name: f.properties.Incid_Name, date: f.properties.Ig_Date, type: f.properties.Incid_Type, acres: f.properties.BurnBndAc}));
            } catch (e) { return []; }
          };
          function update() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
          }
          function updateHover() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
          }

          function labels(on) {
            for (const m of [mapL, mapR]) {
              if (!m || !m.isStyleLoaded()) continue;
              const st = m.getStyle();
              if (!st || !st.layers) continue;
              st.layers.forEach((l) => {
                if (l.layout && l.layout["text-field"] !== undefined)
                  m.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
              });
            }
          }

          let seq = 0, lastView = "";
          function sendView() {
            if (!mapL) return;
            const c = mapL.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: mapL.getZoom(), w: L.mapEl.clientWidth, h: L.mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key;
            v.n = ++seq;
            model.set("view", JSON.stringify(v));
            model.save_changes();
          }

          const cellAt = (lngLat) => {
            if (res < 0) return null;
            try { const h = latLngToCell(lngLat.lat, lngLat.lng, res); return hexIndex.has(h) ? h : null; }
            catch (e) { return null; }
          };

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            const mk = (elm) => new maplibregl.Map({
              container: elm, style: STYLE,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: true},
            });
            mapL = mk(L.mapEl); mapR = mk(R.mapEl);
            // maplibre's own controls: full screen takes the WHOLE widget (both
            // panes and the strip), not the one map it sits on
            mapR.addControl(new maplibregl.FullscreenControl({container: root}), "top-right");
            mapR.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-right");
            ovL = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck L: " + (e && e.message ? e.message : e))});
            ovR = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck R: " + (e && e.message ? e.message : e))});
            mapL.addControl(ovL); mapR.addControl(ovR);
            // one camera: whichever map moves, the other jumps to it (guarded
            // against the echo). moveend on either sends the view (deduped).
            let syncing = false;
            const follow = (a, b) => () => {
              if (syncing) return;
              syncing = true;
              b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()});
              syncing = false;
            };
            mapL.on("move", follow(mapL, mapR));
            mapR.on("move", follow(mapR, mapL));
            let ready = 0;
            const onLoad = () => { ready++; if (ready === 2) { labels(labelsOn); addPerims(mapL); addPerims(mapR); perims(); update(); sendView(); } };
            mapL.on("load", onLoad); mapR.on("load", onLoad);
            mapL.on("moveend", sendView); mapR.on("moveend", sendView);
            mapL.on("zoom", () => update());
            mapR.on("zoom", () => update());
            for (const [m, other] of [[mapL, mapR], [mapR, mapL]]) {
              let lastFire = "";
              m.on("mousemove", (e) => {
                const h = cellAt(e.lngLat);
                if (h !== hover) { hover = h; updateHover(); }
                const fs = firesAt(m, e.point), seen = new Set(), parts = [];
                for (const f of fs) { const k = f.name + f.date; if (seen.has(k)) continue; seen.add(k);
                  parts.push((f.name || "?") + " · ignited " + String(f.date || "").slice(0, 10) + " · " + (f.type || "") + " · " + Number(f.acres || 0).toLocaleString() + " acres (MTBS perimeter)"); }
                const t = parts.join("  |  ");
                if (t !== lastFire) { lastFire = t; fire.textContent = t; }
              });
              m.on("mouseout", () => { if (hover) { hover = null; updateHover(); } if (lastFire) { lastFire = ""; fire.textContent = ""; } });
              m.on("click", (e) => {
                const h = cellAt(e.lngLat);
                // the fires under the click: the perimeters are polygons in the
                // tiles, so a point inside one hits its line layer's feature
                const seen = new Set(), fires = [];
                for (const f of firesAt(m, e.point)) { const k = f.name + f.date; if (!seen.has(k)) { seen.add(k); fires.push(f); } }
                model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, fires, n: ++seq}));
                model.save_changes();
              });
              m.on("error", (ev) => { if (ev && ev.error && ev.error.message) say("map: " + ev.error.message); });
              new ResizeObserver(() => { try { m.resize(); } catch (e) {} }).observe(m.getContainer());
            }
            window.__spTiles = tstat;
            window.__spMaps = () => [mapL, mapR];
            window.__spLayers = () => ({left: layersLeft().map((l) => l.id), right: layersRight().map((l) => l.id), N, res});
          }

          let pendingLoad = null, needCells = false;
          const flush = () => {
            pendingLoad = null;
            try { if (needCells) loadCells(); needCells = false; loadAttrs(); update(); }
            catch (e) { say("load: " + e.message); console.error(e); }
          };
          const reload = () => { needCells = true; if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          const reattr = () => { if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          model.on("change:cells", () => { grab("cells"); reload(); });
          model.on("change:colors", () => { grab("colors"); reattr(); });
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (cfg.s2_year !== s2y) { s2y = cfg.s2_year; styleS2(); }
            if (cfg.label_year !== ly) { ly = cfg.label_year; styleLy(); }
            if (cfg.fill && cfg.fill !== fill) { fill = cfg.fill; styleFill(); }
            if (cfg.labels !== was.labels) { labelsOn = cfg.labels !== false; labels(labelsOn); }
            if (cfg.label_year !== was.label_year || cfg.perims !== was.perims) { perimsOn = cfg.perims !== false; perims(); }
            update();
          });
          try { grab("cells"); grab("colors"); loadCells(); loadAttrs(); renderLegend(); say(model.get("status")); boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { mapL && mapL.remove(); mapR && mapR.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (PairMap,)


@app.cell
def _(AEF_FROM0, AEF_TO0, AEF_YEARS_ALL, FILLS, FILL_NAMES, FILL_SHORT, HEX_ZOOM, HOME, LABELS_SLOT, LABEL_YEAR0, LABEL_YEARS, MTBS_LAYER, MTBS_PMTILES, PairMap, RASTER_TILE, S2_YEAR0, S2_YEARS, STRIP_MINIMAL, VIEW_H, json, lc_bounds):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    pair = PairMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "label_year": LABEL_YEAR0, "fill": FILLS[0],
        "s2_years": list(S2_YEARS), "label_years": list(LABEL_YEARS),
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "hex_zoom": HEX_ZOOM, "extent": list(lc_bounds),
        "perims": True, "perims_src": MTBS_PMTILES, "perims_layer": MTBS_LAYER,
        "minimal": STRIP_MINIMAL,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2y": S2_YEAR0, "ly": LABEL_YEAR0, "fill": FILLS[0], "labels": True, "perims": True,
        "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "memo": {}, "aef": {}, "mtbs": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
    }
    pair
    return HOLD, pair


@app.cell
def _(
    AEF_YEARS_ALL,
    CELL_KM2,
    CLASSES,
    FILLS,
    FILL_NAMES,
    HEX_ZOOM,
    HOLD,
    HOME,
    LABEL_YEARS,
    MTBS_FOLD_YEARS,
    S2_YEARS,
    SETTLE,
    STRIP_MINIMAL,
    aef_fold,
    asyncio,
    build_frame,
    con,
    contains,
    json,
    lc_fold,
    lc_tile_png,
    mtbs_fold,
    np,
    pad_box,
    pair,
    res_for_view,
    s2_raster_stats,
    s2_tile_png,
    time,
    traceback,
    view_to_bbox,
):
    # ---- wiring: the camera loop and the controls. Re-runs freely. ---------------
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] += 1

    async def _tile_fn(src, z, x, y, year):
        if src == "lcms":
            return await lc_tile_png(z, x, y, year)
        return await s2_tile_png(z, x, y, year)

    pair.tile_fn = _tile_fn

    def _say(msg):
        try:
            pair.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(pair.config or "{}")
        c.update(kw)
        pair.config = json.dumps(c)

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    def _hexes_off(msg):
        if HOLD["sent"] is not None:
            with pair.hold_sync():
                pair.cells, pair.colors = b"", b""
            HOLD["sent"] = None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        _cfg(hit=None)
        pair.legend = "[]"
        pair.panel = ""
        _say(msg)

    def _paint():
        fr = HOLD["frame"]
        if fr is None:
            return False
        rgba = fr["fill"](HOLD["fill"], HOLD["hit"])
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None)
        with pair.hold_sync():
            if HOLD["sent"] is not fr:
                pair.cells = fr["cellid"].astype("<u8").tobytes()
                HOLD["sent"] = fr
            pair.colors = rgba.tobytes()
        pair.legend = json.dumps(fr["legend"](HOLD["fill"]))
        return True

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · LCMS {HOLD['ly']} raster on the right · zoom in past {HEX_ZOOM:g} for the hexagons")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and not force:
            if res_for_view(vsd, box) <= HOLD["res"]:
                _say(HOLD.get("last_status", "") + " · held")
                return
        res = res_for_view(vsd, box)
        ly, y0, y1 = HOLD["ly"], HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (ly, y0, y1, res, rbox)
        t0 = time.time()
        years = list(range(y0, y1 + 1))
        _say(f"folding LCMS {ly} and AlphaEarth {y0} to {y1} ({len(years)} years)…" if STRIP_MINIMAL
             else f"res {res} · folding LCMS {ly}, AlphaEarth {y0}..{y1} ({len(years)} years)… (wiring run {HOLD['runs']})")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox)  # the fold cache is per res and BOX (2026-09-01: key[2] became y1 when the window joined the key, and every pan reused the first box's AEF fold)
            need = [y for y in years if (y, bkey) not in HOLD["aef"]]
            myears = [ly + d for d in MTBS_FOLD_YEARS]
            mneed = [y for y in myears if (y, bkey) not in HOLD["mtbs"]]
            got = await asyncio.gather(
                lc_fold(box, res, ly), *(aef_fold(box, res, y) for y in need), *(mtbs_fold(box, res, y) for y in mneed)
            )
            nl, s1 = got[0]
            for y, (tab, st) in zip(need, got[1:1 + len(need)]):
                HOLD["aef"][(y, bkey)] = (tab, st)
            for y, (tab, st) in zip(mneed, got[1 + len(need):]):
                HOLD["mtbs"][(y, bkey)] = (tab, st)
            for k_ in ("aef", "mtbs"):
                if len(HOLD[k_]) > 40:
                    HOLD[k_].pop(next(iter(HOLD[k_])))
            if nl is None or nl.num_rows == 0:
                _say(f"res {res} · {s1}")
                return
            aef_by_year = {y: HOLD["aef"][(y, bkey)][0] for y in years if (y, bkey) in HOLD["aef"]}
            mtbs_by_year = {y: HOLD["mtbs"][(y, bkey)][0] for y in myears if (y, bkey) in HOLD["mtbs"]}
            s2s = " · ".join(HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"])
            s3s = " · ".join(HOLD["mtbs"][(y, bkey)][1] for y in myears if (y, bkey) in HOLD["mtbs"])
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, nl, aef_by_year, y0, y1, mtbs_by_year)
            stats = " · ".join(x for x in (f"res {res}", s1, s2s, s3s, f"frame {time.time() - t1:.1f} s") if x)
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        st = s2_raster_stats()
        HOLD["last_status"] = (
            f"{stats} · {fr['score']}" + (f" · {fr['n_burned']:,} cells MTBS-burned" if MTBS_FOLD_YEARS else "")
            + f" · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
            f" · S2 tiles {st['served']:,} served, {st['blank']:,} empty"
        )
        _say(HOLD["last_status"])

    async def refresh(vs, force=False, settle=True):
        """ONE serve at a time; the latest request wins while one is in flight."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            HOLD["pending_force"] = HOLD["pending_force"] or force
            return
        HOLD["busy"] = True
        try:
            while True:
                if settle:
                    await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    force, HOLD["pending_force"] = HOLD["pending_force"], False
                    settle = True
                    continue
                await _serve(vs, force)
                vs = HOLD["pending"]
                if vs is None:
                    return
                force, HOLD["pending"], HOLD["pending_force"] = HOLD["pending_force"], None, False
                settle = False
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"failed: {type(exc).__name__}: {exc}{where}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"], HOLD["pending_force"] = False, None, False

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _request(force=False):
        vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
        HOLD["task"] = _spawn(refresh(vs, force, settle=False))

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            pair.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    pair.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    def _f(v, d=3):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    def _on_pick(change):
        fr = HOLD["frame"]
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        if fr is None:
            return
        try:
            cellh = p.get("cell")
            if not cellh:
                HOLD["hit"] = None
                fires = p.get("fires") or []
                pair.panel = ("<b>MTBS perimeter</b>: " + "; ".join(
                    f"{f.get('name') or '?'} ({str(f.get('date') or '')[:10]}, {f.get('type') or ''}, {int(f.get('acres') or 0):,} ac)"
                    for f in fires)) if fires else ""
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT name, maj_name, p_chg, purity, disp, disp_max, when_name, mtbs_name, mtbs_year "
                "FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            ci = int(np.searchsorted(fr["cellid"], np.uint64(cell)))
            row_steps = fr["steps"][:, ci] if r is not None and ci < len(fr["cellid"]) else []
            fires = p.get("fires") or []
            fires_html = (" · <b>MTBS perimeter</b>: " + "; ".join(
                f"{f.get('name') or '?'} ({str(f.get('date') or '')[:10]}, {f.get('type') or ''}, {int(f.get('acres') or 0):,} ac)"
                for f in fires)) if fires else ""
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                pair.panel = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                nm, majn, pc, pur, dsp, dmx, wn, mn, my = r
                y0, y1 = fr["y0"], fr["y1"]
                ly = HOLD["ly"]
                d0 = fr["D0"]
                # BLUNT (Stephen): one plain sentence per source, then the numbers small
                if nm == "Stable":
                    l1 = f"LCMS {ly} says nothing happened here."
                else:
                    l1 = f"LCMS {ly} says <b>{nm}</b> ({100 * pc:.0f}% of the hexagon's pixels)."
                if wn.startswith("no AlphaEarth"):
                    l2 = "AlphaEarth has no embedding here."
                elif wn.startswith("AlphaEarth saw no change"):
                    l2 = f"AlphaEarth saw <b>no change</b> from {y0} to {y1}."
                else:
                    yr = wn.split("changed in ")[1].split(" ")[0]
                    l2 = f"AlphaEarth saw the ground <b>change in {yr}</b>."
                fires = p.get("fires") or []
                if fires:
                    f0 = fires[0]
                    l3 = (f"MTBS: the <b>{f0.get('name') or '?'}</b> fire, ignited {str(f0.get('date') or '')[:10]}, "
                          f"{int(f0.get('acres') or 0):,} acres" + (f" (+{len(fires) - 1} more)" if len(fires) > 1 else "") + ".")
                else:
                    l3 = "MTBS: no mapped fire here in these years."
                detail = (
                    f"shift {y0} to {y1} {_f(dsp)} · steps "
                    + " · ".join(f"{ya}→{yb} {_f(v)}" for (ya, yb), v in zip(fr["step_years"], row_steps))
                    + (f" · a step counts as change above {_f(d0)}" if not np.isnan(d0) else "")
                    + f" · majority pixel class {majn} ({pur:.2f}) · {CELL_KM2.get(HOLD['res'], 0):.3f} km²{where}"
                )
                pair.panel = (
                    f"<div style='font-size:14px;line-height:1.5'>{l1}<br>{l2}<br>{l3}</div>"
                    + ("" if STRIP_MINIMAL else f"<div style='font-size:12px;color:#777'>{detail}</div>")
                )
        except Exception as e:
            pair.panel = f"<span style='opacity:.7'>click: {e}</span>"
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            pair.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    pair.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "s2":
            y = int(c.get("s2y", HOLD["s2y"]))
            if y in S2_YEARS and y != HOLD["s2y"]:
                HOLD["s2y"] = y
                _cfg(s2_year=y)
                _say((HOLD.get("last_status") or "") + f" · Sentinel-2 {y}")
            return
        if act == "label":
            y = int(c.get("ly", HOLD["ly"]))
            if y in LABEL_YEARS and y != HOLD["ly"]:
                HOLD["ly"] = y
                _cfg(label_year=y)
                _request(force=True)
            return
        if act == "aef":
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in AEF_YEARS_ALL and b in AEF_YEARS_ALL and a < b and (a, b) != (HOLD["y0"], HOLD["y1"]):
                HOLD["y0"], HOLD["y1"] = a, b
                _cfg(aef_from=a, aef_to=b)
                _request(force=True)
            return
        if act == "fill":
            f = c.get("fill")
            if f in FILLS and f != HOLD["fill"]:
                HOLD["fill"] = f
                _cfg(fill=f)
                if _paint():
                    _say((HOLD.get("last_status") or "") + f" · {FILL_NAMES[f]}")
            return
        if act == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
            return
        if act == "perims":
            HOLD["perims"] = bool(c.get("perims", True))
            _cfg(perims=HOLD["perims"])
            return

    def _on_ctl(change):
        try:
            _on_ctl_body(change)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"control failed: {type(e).__name__}: {e}{where}")

    if HOLD.get("h_ctl") is not None:
        try:
            pair.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    pair.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    if HOLD["frame"] is None and not HOLD["busy"]:
        _request()
    else:
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    DuckDB over the CURRENT view's cells (press the button after the map
    settles): `cls` / `name` (the cell's main change class, or Stable when
    under `CHG_MIN` of its pixels changed), `p_chg`, `disp` (the AlphaEarth
    displacement between the window's two ends), `disp_max` (its largest single
    step) with one `step_YYYY` per step ending in YYYY, `moved`, `when` (the
    year, or -1 never, -2 no embedding) / `when_name`, and
    `mtbs_sev` / `mtbs_name` / `mtbs_year` (the burn severity majority and the
    year it burned, label year or the year before; 0 when not mapped burned).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="tables for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no view folded yet*"))
    con.register("view_cells", HOLD["frame"]["cells"])
    per_class = mo.sql(
        """
        SELECT cls, name, count(*) AS cells,
               round(avg(p_chg), 3) AS p_chg_mean,
               round(100 * avg(CASE WHEN moved THEN 1 ELSE 0 END) FILTER (WHERE disp IS NOT NULL), 1) AS pct_moved,
               round(median(disp), 4) AS disp_p50,
               round(median(disp_max), 4) AS disp_max_p50
        FROM view_cells GROUP BY cls, name ORDER BY cells DESC
        """,
        engine=con,
    )
    return (per_class,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None)
    when_by_class = mo.sql(
        """
        PIVOT (SELECT name, when_name FROM view_cells)
        ON when_name USING count(*) GROUP BY name ORDER BY name
        """,
        engine=con,
    )
    return (when_by_class,)


if __name__ == "__main__":
    app.run()
