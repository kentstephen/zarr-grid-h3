# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pillow",
#     "python-dotenv",
# ]
# ///


"""S2 x CTrees: the ground as a picture beside one H3 fill of biomass change.

A fork of the settlement pair (s2-wsf-aef-overture-pair.py) with the label
side swapped for one source and the whole thing made simpler. Two maps in one
widget, one camera. LEFT: a picture, never covered. Earth Genome's Sentinel-2
yearly mosaic (2022-2025) and OpenGeoHub's Landsat bi-monthly mosaics
(GLAD ARD v2 on the Copernicus Data Space, 1997-2021), both as live tiles the
kernel renders from the COGs, one slider across both sensors. RIGHT: one opaque H3 fill per hexagon at res 9 and coarser, from

  CTrees global aboveground biomass (AWS Open Data, Icechunk): 100 m, annual
  2000-2025, Mg/ha, with a residual standard error per pixel per year. All
  26 years are read for the box at once; the window (from year, to year)
  picks two ends and the years between, and the fills are the change
  between the ends (blue gain, orange loss), the year the biomass first fell
  past a threshold, and the stock at the to-end.

The uncertainty is opacity: a change smaller than the uncertainty of its
two ends is drawn faint.

The fold is the H3 UDF inside DataFusion (repo rule): the pixels cross as one
Dataset and the cell is the GROUP BY. Nothing is tessellated in the kernel;
the browser gets cell ids and rgba.

Run: uv run marimo edit s2-ctrees-pair.py

Landsat needs a Copernicus Data Space S3 key pair in the environment
(CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY; a .env in the repo is read). Without
it a Landsat year draws nothing and the legend says so.

Attribution: CTrees global AGB (CC-BY 4.0, doi 10.82924/7vmb-zv66; Yang,
Saatchi et al. 2026). Sentinel-2 yearly mosaics and Sentinel-2 L2A temporal
mosaics (CC-BY 4.0) by Earth Genome (Copernicus Sentinel data). Landsat
bi-monthly mosaics by OpenGeoHub (GLAD ARD v2, Consoli et al. 2024) via the
Copernicus Data Space Ecosystem. Place search by Photon (komoot),
OpenStreetMap data (ODbL).
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
    import xarray as xr
    import icechunk
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from dotenv import load_dotenv

    import io
    from PIL import Image

    load_dotenv()
    return (
        GeoTIFF,
        Image,
        S3Store,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        icechunk,
        io,
        json,
        math,
        mo,
        np,
        os,
        pa,
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
    # The S2 CTrees pair

    **Left**: a picture, never covered. **Right**: one H3 fill of biomass
    over the same camera. Pan either map and both move. Hover a hexagon on
    either side and its ring is drawn on both.

    - **PICTURE** (left header, a slider, the arrow keys or `[` `]`): 1997
      to 2025. 2022 to 2025 are the Sentinel-2 yearly mosaics, live tiles
      (the temporal median fills the yearly's cloud holes). 1997 to 2021 are
      the Landsat bi-monthly mosaics (July-August), live tiles too, read
      from Copernicus with your key and kept on disk; the year below the one
      you look at is fetched ahead. The `scale` slider is a gain on both
      sensors' true colour. **FIND**: a
      Photon geocoder; Enter or a click flies both maps there.
    - **FILL** (keys `1` to `3`): **change**, the biomass at the window's
      to-end minus the from-end, blue where it rose and orange where it fell,
      faint where the change is smaller than the uncertainty of its two
      ends · **loss year**, the first year inside the window the biomass fell
      past a quarter of its running maximum (and at least 10 Mg/ha), dark
      early to light late · **stock**, the biomass at the to-end, on greens,
      faint where the uncertainty exceeds it.
    - **WINDOW** (the slider, keys `-` `=` for the from end and `_` `+` for
      the to end): whole years 2000 to 2025. All 26 years are read for the
      box once; a window change is a frame, never a fetch.
    - Click a hexagon for its story and its 26-year series. `L` toggles the
      basemap labels, `F` full screen.

    The hexagons fold from zoom 9 and stop at res 9, about one CTrees pixel,
    while the picture keeps zooming. Below zoom 9 the right pane is empty.
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # The picture years: Landsat 1997..2021 then the Sentinel-2 mosaics
    # 2022..2025, both live tiles. One slider, one picture at a time
    # (Stephen, 2026-09-04: "we would just show one at a time").
    S2_YEARS = (2022, 2023, 2024, 2025)
    LS_YEARS = tuple(range(1997, 2022))
    PIC_YEARS = LS_YEARS + S2_YEARS
    S2_YEAR0 = 2022
    S2_SCALE0 = 1.0
    # the biomass window: the whole run by default (the story is movement)
    CT_YEARS = tuple(range(2000, 2026))
    WIN_FROM0, WIN_TO0 = 2000, 2025

    # The zoom -> H3 ladder: BASE_RES at ZOOM0, one step finer every PER_RES
    # zoom units, clamped, then coarsened until the view's expected cell
    # count fits CELL_BUDGET. Capped at res 9, one hexagon per CTrees pixel
    # (Stephen: "res nine hexagons are fine"); the picture keeps zooming.
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 9
    CELL_BUDGET = 300_000

    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    S2_FILL_COLLECTION = "sentinel2-temporal-mosaics"
    S2_TILE_MIN_Z, S2_PYRAMID_Z, S2_TCI_MAX_Z = 7, 9, 14
    # NDVI as the picture (Stephen, 2026-09-04: "the ndvi should cut through
    # the tci noise"): B04 and B08 of the yearly mosaic (uint16, the same
    # pyramid as the TCI, on source.coop; the temporal mosaic's bands sit on
    # a bucket that refuses anonymous reads, so NDVI has no backfill), and
    # B03 / B04 of the Landsat tiles. One fixed scale for every year and both
    # sensors, on Carto's Emrld. The yearly mosaic's reflectance carries the
    # Sentinel-2 offset of 1000 (its red never reads below ~1177); the temporal
    # median does not (red from ~200): the offset is removed per item.
    S2_OFFSET = 1000
    NDVI_LO, NDVI_HI = -0.1, 0.9
    # dark low, bright high (Stephen, 2026-09-04: "the bright end of the cmap
    # should be high ndvi"): Carto's Emrld reversed
    EMRLD = ("#074050", "#105965", "#217a79", "#4c9b82", "#6cc08b", "#97e196", "#d3f2a3")
    PIC_MODES = (("tc", "true colour"), ("ndvi", "NDVI"))

    # ---- CTrees AGB: one Icechunk store, 26 annual slices, 100 m 4326 -------
    # int16 stored, /10 for Mg/ha, nodata -9999; `uncertainty` the residual
    # standard error, same shape and scale. Chunks (1, 2000, 2000). Pixel
    # edges from (-180, 90); centres half a pixel in.
    CT_BUCKET = "ctrees-agb-100m-global"
    CT_PREFIX = "agb_100m_global"
    CT_GROUP = "aboveground_biomass"
    CT_RES = 360.0 / 405000
    CT_X0, CT_Y0 = -180.0, 90.0
    CT_NODATA = -9999
    # samples per year the fold reads: a stride fits the window to this
    CT_MAX_PX = 3_000_000
    CT_READERS = 8
    # the loss year: the first year the drop from the window's running
    # maximum passes both of these (a smoothed product ramps; a threshold on
    # the cumulative drop reads better than one on a single step)
    LOSS_FRAC = 0.25
    LOSS_ABS = 10.0

    # ---- Landsat: OpenGeoHub bi-monthly mosaics on the Copernicus Data Space --
    # The COGs are s3://eodata behind a CDSE key pair, at a path that follows
    # from the year, the period and the 1-degree cell name (no STAC call:
    # the search endpoint rate-limits at a handful of calls a second, and
    # the STAC items only repeat what the path already says).
    LS_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
    LS_BUCKET = "eodata"
    LS_PATH = "Global-Mosaics/Landsat/OLM_SWA_ARD2/v1/{year}/{mm}/01/Landsat_mosaic_{year}_{period}_{cell}_V1.0.1/{band}_{year}.tif"
    # the bi-monthly period that stands for a year
    LS_PERIOD = "07-08"
    LS_BANDS = ("B03", "B02", "B01", "B04")  # red, green, blue, NIR
    # the tile zooms: 30 m is about z12 at the equator; deck scales the z12
    # tile past it. Below z7 a tile spans more than a cell's coarsest overview
    LS_TILE_MIN_Z, LS_MAX_Z = 7, 12
    # the largest window one band read may ask for (pixels)
    LS_WIN_MAX = 4_000_000
    LS_CONNECTIONS = 4  # the quota for a general user
    # after a live tile, the same tile for this many years below is fetched
    # in the background once the live requests go quiet
    LS_PREFETCH = 1
    LS_KEY = os.environ.get("CDSE_S3_ACCESS_KEY", "")
    LS_SECRET = os.environ.get("CDSE_S3_SECRET_KEY", "")

    VIEW_W, VIEW_H = 700, 720
    STRIP_MINIMAL = True
    PAD = 1.3
    SETTLE = 0.35
    HEX_ZOOM = 9.0
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    # home: Ji-Parana, Rondonia, on the arc of deforestation; the geocoder is
    # the real home (Stephen: "this is just like a worldwide notebook")
    HOME = {"longitude": -61.95, "latitude": -10.88, "zoom": 10.4}
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "ctrees-pair")

    FILLS = ("change", "lossyear", "stock")
    FILL_NAMES = {
        "change": "biomass at the window's to-end minus the from-end (Mg/ha): blue rose, orange fell; faint where smaller than the uncertainty",
        "lossyear": "the first year inside the window the biomass fell past a quarter of its running maximum (and 10 Mg/ha)",
        "stock": "biomass at the window's to-end (Mg/ha); faint where the uncertainty exceeds it",
    }
    FILL_SHORT = {"change": "change", "lossyear": "loss year", "stock": "stock"}
    ALPHA_FILL = 235
    ALPHA_QUIET = 70
    # gain blue, loss orange, quiet white: the axis a protanope keeps
    DIV_RAMP = ("#1f5fa8", "#5b8fd0", "#a9c5e8", "#f2f2f2", "#f4b46b", "#e07b1c", "#a85200")
    # the loss year: cividis, a lightness ramp, dark early to light late
    CIVIDIS = ("#00204c", "#00336f", "#39486b", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#ffea46")
    # the stock: matplotlib Greens less its white end (greens are fine)
    GREENS = ("#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b")
    GREY = (222, 222, 222)
    return (
        ALPHA_FILL,
        ALPHA_QUIET,
        BASE_RES,
        CACHE_DIR,
        CELL_BUDGET,
        CIVIDIS,
        CT_BUCKET,
        CT_GROUP,
        CT_MAX_PX,
        CT_NODATA,
        CT_PREFIX,
        CT_READERS,
        CT_RES,
        CT_X0,
        CT_Y0,
        CT_YEARS,
        DIV_RAMP,
        EMRLD,
        FILLS,
        FILL_NAMES,
        FILL_SHORT,
        GREENS,
        GREY,
        HEX_ZOOM,
        HOME,
        LABELS_SLOT,
        LOSS_ABS,
        LOSS_FRAC,
        LS_BANDS,
        LS_BUCKET,
        LS_CONNECTIONS,
        LS_ENDPOINT,
        LS_KEY,
        LS_MAX_Z,
        LS_PATH,
        LS_PERIOD,
        LS_PREFETCH,
        LS_SECRET,
        LS_TILE_MIN_Z,
        LS_WIN_MAX,
        LS_YEARS,
        MAX_RES,
        MIN_RES,
        NDVI_HI,
        NDVI_LO,
        PAD,
        PER_RES,
        PIC_MODES,
        PIC_YEARS,
        RASTER_TILE,
        S2_COLLECTION,
        S2_FILL_COLLECTION,
        S2_OFFSET,
        S2_PYRAMID_Z,
        S2_SCALE0,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEAR0,
        S2_YEARS,
        SETTLE,
        STRIP_MINIMAL,
        VIEW_H,
        VIEW_W,
        WIN_FROM0,
        WIN_TO0,
        ZOOM0,
    )


@app.cell
def _(
    BASE_RES,
    CELL_BUDGET,
    MAX_RES,
    MIN_RES,
    PAD,
    PER_RES,
    VIEW_H,
    VIEW_W,
    ZOOM0,
    math,
):
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

    def merc_y(lat):
        return _lat_to_y(lat)

    def merc_lat(y):
        return _y_to_lat(y)

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
def _(EMRLD, NDVI_HI, NDVI_LO, np):
    # ---- the NDVI picture: one ramp, one fixed scale, both sensors ------------
    _st = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in EMRLD], np.float64)
    NDVI_RAMP = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_st)), _st[:, k]) for k in range(3)], 1).round().astype(np.uint8)
    NDVI_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in NDVI_RAMP[i]) for i in range(0, 256, 17)]

    def ndvi_rgb(red, nir):
        """(rgb (h, w, 3) uint8, valid (h, w) bool) from two reflectance
        arrays, NDVI on the fixed NDVI_LO..NDVI_HI scale; valid where both
        bands are positive."""
        r = red.astype(np.float32)
        n = nir.astype(np.float32)
        valid = (r > 0) & (n > 0)
        den = np.where(valid, r + n, 1.0)
        v = np.where(valid, (n - r) / den, NDVI_LO)
        i = np.clip((v - NDVI_LO) / (NDVI_HI - NDVI_LO) * 255, 0, 255).round().astype(np.int64)
        return NDVI_RAMP[i], valid

    return NDVI_HEX, ndvi_rgb


@app.cell
def _(
    CT_BUCKET,
    CT_GROUP,
    CT_MAX_PX,
    CT_NODATA,
    CT_PREFIX,
    CT_READERS,
    CT_RES,
    CT_X0,
    CT_Y0,
    CT_YEARS,
    asyncio,
    ctx,
    icechunk,
    math,
    np,
    time,
    xr,
):
    # ---- CTrees: the Icechunk store, one fold per (box, res), all 26 years ------
    # The grid is plate carree, so a lon/lat box IS a window. The fold reads
    # every year of `agb` and `uncertainty` under the box on a stride s so the
    # samples fit CT_MAX_PX per year (an unbiased sample of the cell's mean),
    # the years in parallel threads, then ONE DataFusion query with 52 means.
    # A window change is a frame; the fetch happens once per (box, res).
    _st = icechunk.s3_storage(bucket=CT_BUCKET, prefix=CT_PREFIX, region="us-west-2", anonymous=True)
    _repo = icechunk.Repository.open(_st)
    _ds = xr.open_zarr(_repo.readonly_session(branch="main").store, zarr_format=3, group=CT_GROUP, chunks=None, mask_and_scale=False)
    _ti = {int(y): i for i, y in enumerate(_ds.time.dt.year.values)}
    _win = {}
    _sem = asyncio.Semaphore(CT_READERS)
    _fold_lock = asyncio.Lock()

    def _window_ix(box):
        W_, S_, E_, N_ = box
        H, W = _ds.agb.shape[1], _ds.agb.shape[2]
        c0, c1 = max(0, int(math.floor((W_ - CT_X0) / CT_RES))), min(W, int(math.ceil((E_ - CT_X0) / CT_RES)))
        r0, r1 = max(0, int(math.floor((CT_Y0 - N_) / CT_RES))), min(H, int(math.ceil((CT_Y0 - S_) / CT_RES)))
        return c0, c1, r0, r1

    # Reads are serialised at about 0.25 s per chunk from here whatever the
    # threading (measured 2026-09-04: 26 years in one call 6.0 s, 8 threads
    # per year 8.2 s, 52 threaded reads 14.6 s). So the biomass comes as ONE
    # call over the time axis, and the uncertainty only for the two window
    # ends, cached per year.
    async def _read(var, tsel, r0, r1, c0, c1, stride):
        loop = asyncio.get_running_loop()
        async with _sem:
            return await loop.run_in_executor(
                None, lambda: np.asarray(_ds[var].isel(time=tsel, y=slice(r0, r1, stride), x=slice(c0, c1, stride)).values)
            )

    async def ct_window(box, stride, years):
        """(A (26, h, w) int16 biomass, U {year: (h, w) int16 uncertainty for
        the `years` asked}, lon of the columns, lat of the rows) under the
        box, every `stride`-th pixel; or None."""
        c0, c1, r0, r1 = _window_ix(box)
        if c1 <= c0 or r1 <= r0:
            return None
        key = (stride, r0, r1, c0, c1)
        got = _win.get(key)
        if got is None:
            A = (await _read("agb", slice(None), r0, r1, c0, c1, stride)).astype(np.int16)
            got = (A, {})
            _win[key] = got
            if len(_win) > 6:
                _win.pop(next(iter(_win)))
        A, U = got
        need = [y for y in years if y not in U]
        if need:
            parts = await asyncio.gather(*(_read("uncertainty", _ti[y], r0, r1, c0, c1, stride) for y in need))
            for y, u in zip(need, parts):
                U[y] = u.astype(np.int16)
        lon = CT_X0 + (c0 + stride * np.arange(A.shape[2]) + 0.5) * CT_RES
        lat = CT_Y0 - (r0 + stride * np.arange(A.shape[1]) + 0.5) * CT_RES
        return A, U, lon, lat

    _AGG = ", ".join(f"avg(CASE WHEN a_{y} >= 0 THEN CAST(a_{y} AS DOUBLE) / 10.0 END) AS a_{y}" for y in CT_YEARS)

    async def ct_fold(box, res, y0, y1):
        """Per res cell over the box: the sample count (`npx`), how many were
        valid in the last year (`nok`), the mean biomass per year in Mg/ha
        (`a_2000`..`a_2025`, null where no sample was valid) and the mean
        uncertainty at the window's two ends (`u0`, `u1`). (table or None,
        stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        c0, c1, r0, r1 = _window_ix(box)
        if c1 <= c0 or r1 <= r0:
            return None, "CTrees: nothing under the view"
        full = (c1 - c0) * (r1 - r0)
        stride = max(1, int(math.ceil(math.sqrt(full / CT_MAX_PX))))
        got = await ct_window(box, stride, (y0, y1))
        if got is None:
            return None, "CTrees: nothing under the view"
        A, U, lon, lat = got
        tr = time.time()
        h, w = A.shape[1], A.shape[2]
        LON, LAT = np.meshgrid(lon, lat)
        last = CT_YEARS[-1]
        data = {f"a_{y}": (("y", "x"), A[i]) for i, y in enumerate(CT_YEARS)}
        data |= {"u0": (("y", "x"), U[y0]), "u1": (("y", "x"), U[y1])}
        data |= {"lat": (("y", "x"), LAT), "lon": (("y", "x"), LON)}
        async with _fold_lock:
            try:
                ctx.deregister_table("ct")
            except Exception:
                pass
            ctx.from_dataset("ct", xr.Dataset(data, coords={"y": np.arange(h), "x": np.arange(w)}), chunks={"y": 256})
            out = ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell,
                       count(*) AS npx,
                       sum(CASE WHEN a_{last} >= 0 THEN 1 ELSE 0 END) AS nok,
                       {_AGG},
                       avg(CASE WHEN u0 >= 0 THEN CAST(u0 AS DOUBLE) / 10.0 END) AS u0,
                       avg(CASE WHEN u1 >= 0 THEN CAST(u1 AS DOUBLE) / 10.0 END) AS u1
                FROM ct
                WHERE lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                  AND a_{last} > {CT_NODATA}
                GROUP BY cell
            """).to_arrow_table()
        return out, (
            f"CTrees {w:,}x{h:,} samples x 26 years (stride {stride}, {int(100 * stride)} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - tr:.1f} s ({y0}, {y1} uncertainty)"
        )

    return (ct_fold,)


@app.cell
def _(
    GeoTIFF,
    Image,
    RASTER_TILE,
    S2_COLLECTION,
    S2_FILL_COLLECTION,
    S2_OFFSET,
    S2_PYRAMID_Z,
    S2_SCALE0,
    S2_STAC,
    S2_TCI_MAX_Z,
    S2_TILE_MIN_Z,
    S3Store,
    Window,
    asyncio,
    io,
    json,
    math,
    ndvi_rgb,
    np,
    time,
    urllib,
):
    # ---- Sentinel-2 tiles, by YEAR and MODE: the left pane (TCI from docs/13) --
    # STAC once per (year, z9 ancestor tile), every footprint under the tile
    # composited in numpy (black = nodata -> alpha 0; first footprint to paint a
    # pixel wins), one PNG. The yearly footprints come first, then the same
    # year's S2_FILL_COLLECTION footprints (ids suffixed `#fill`):
    # first-to-paint-wins is the backfill.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _R = 6378137.0
    _items = {}
    _boxes = {}
    _open = {}
    _sem = asyncio.Semaphore(32)
    _png = {}
    _arr = {}
    _gain = {"v": float(S2_SCALE0)}
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}
    _fill = {}

    def _band(f, name):
        """The band's path on source.coop, or None (the temporal mosaic's
        bands live on ei-imagery, not readable anonymously)."""
        h = (f["assets"].get(name) or {}).get("href", "")
        return h.split("source.coop/")[1] if "source.coop/" in h else None

    def _encode(key, out, mode="tc"):
        g = _gain["v"] if mode == "tc" else 1.0
        rgba = out if g == 1.0 else np.concatenate(
            [np.clip(out[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8), out[..., 3:]], axis=2)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png[key] = buf.getvalue()
        if len(_png) > 6000:
            _png.pop(next(iter(_png)))
        return _png[key]

    def _stac(box):
        body = json.dumps(
            {"collections": [S2_COLLECTION, S2_FILL_COLLECTION], "bbox": list(box), "limit": 200}
        ).encode()
        req = urllib.request.Request(S2_STAC, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["features"]

    async def _s2_items(box, year):
        key = (year, tuple(round(v, 2) for v in box))
        if key not in _boxes:
            loop = asyncio.get_running_loop()
            both = await loop.run_in_executor(None, _stac, box)
            feats = [f for f in both if f.get("collection") != S2_FILL_COLLECTION]
            ids, fill_ids = [], []
            for f in both:
                if not f["id"].endswith(f"{year}-01-01_{year + 1}-01-01"):
                    continue
                if f.get("collection") == S2_FILL_COLLECTION:
                    if "source.coop/" not in f["assets"]["TCI"]["href"]:
                        continue  # the 2024 temporal items sit on a private bucket (ei-imagery)
                    iid = f["id"] + "#fill"
                    _items[iid] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox"), "fill": True,
                                   "b04": _band(f, "B04"), "b08": _band(f, "B08")}
                    fill_ids.append(iid)
                    continue
                _items[f["id"]] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox"),
                                   "b04": _band(f, "B04"), "b08": _band(f, "B08")}
                ids.append(f["id"])
            if not ids and feats:
                seen = set()
                for f in feats:
                    tile = f["id"].split("_")[0]
                    if tile in seen:
                        continue
                    seen.add(tile)
                    iid = f"{tile}_{year}-01-01_{year + 1}-01-01"
                    base = f["assets"]["TCI"]["href"].split("source.coop/")[1].rsplit("/", 2)[0]
                    _items[iid] = {"tci": f"{base}/{iid}/TCI.tif", "bbox": f.get("bbox"),
                                   "b04": f"{base}/{iid}/B04.tif", "b08": f"{base}/{iid}/B08.tif"}
                    ids.append(iid)
            _boxes[key] = ids + fill_ids
        return _boxes[key]

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                try:
                    _open[rel] = await GeoTIFF.open(rel, store=_store)
                except Exception:
                    _open[rel] = None
        return _open[rel]

    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    async def _items_for_tile(z, x, y, year):
        d = max(0, z - S2_PYRAMID_Z)
        ids = await _s2_items(_tile_ll(z - d, x >> d, y >> d), year)
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        out = []
        for i in ids:
            b = _items[i].get("bbox")
            if not b or (b[0] < E_ and b[2] > W_ and b[1] < N_ and b[3] > S_):
                out.append(i)
        return out

    async def _window(g, li, tx0, ty1, T, tpx, xs, ys):
        """One file's level-li window under the tile, nearest-sampled onto
        the tile's pixels: (array (bands, T, T), inside mask) or None."""
        lv = [g, *g.overviews][min(li, len(g.overviews))]
        L, _B, R_, Tt = g.bounds
        H, W = lv.shape
        px = (R_ - L) / W
        c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
        r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, r1 - r0, c1 - c0)
        cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
        rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
        okc, okr = (cols >= 0) & (cols < c1 - c0), (rows >= 0) & (rows < r1 - r0)
        v = a[:, np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]]
        return v, okr[:, None] & okc[None, :]

    async def _ndvi_tile(key, akey, ids, z, x, y, t0):
        T = RASTER_TILE
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        tx0, ty1 = -world / 2 + x * world / n, world / 2 - y * world / n
        xs = tx0 + (np.arange(T) + 0.5) * tpx
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)
        out = np.zeros((T, T, 4), np.uint8)
        for iid in ids:
            it = _items[iid]
            if not it.get("b04") or not it.get("b08"):
                continue
            gr, gn = await _get(it["b04"]), await _get(it["b08"])
            if gr is None or gn is None:
                continue
            wr, wn = await asyncio.gather(_window(gr, li, tx0, ty1, T, tpx, xs, ys), _window(gn, li, tx0, ty1, T, tpx, xs, ys))
            if wr is None or wn is None:
                continue
            # nodata is raw 0 in either band; the offset-corrected values floor
            # at 1 so water and shadow (raw NIR near the offset) go to the dark
            # end instead of dropping out (Stephen, 2026-09-04: "are we
            # clipping for low ndvi values")
            # the yearly mosaic carries the offset (red p1 1177 over 20LPP 2022),
            # the temporal median does not (red p1 201): per item, not per tile
            off = 0 if it.get("fill") else S2_OFFSET
            red = np.maximum(wr[0][0].astype(np.int32) - off, 1)
            nir = np.maximum(wn[0][0].astype(np.int32) - off, 1)
            rgb, valid = ndvi_rgb(red, nir)
            valid = wr[1] & wn[1] & (wr[0][0] > 0) & (wn[0][0] > 0) & (out[..., 3] == 0)
            out[valid, :3] = rgb[valid]
            out[valid, 3] = 255
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            _arr[akey] = None
            return None
        _arr[akey] = out
        if len(_arr) > 2000:
            _arr.pop(next(iter(_arr)))
        png = _encode(key, out, "ndvi")
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    async def s2_tile_png(z, x, y, year, mode="tc"):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's mosaic, true
        colour (`tc`, the TCI, the yearly backfilled by the temporal) or
        `ndvi` (B04 and B08 of the yearly, on the fixed Emrld scale); or None
        (below S2_TILE_MIN_Z, or no footprint under the tile)."""
        key = (mode, year, z, x, y, _gain["v"] if mode == "tc" else 1.0)
        if key in _png:
            return _png[key]
        akey = (mode, year, z, x, y)
        if akey in _arr:
            out = _arr[akey]
            return _encode(key, out, mode) if out is not None else None
        if z < S2_TILE_MIN_Z or z > S2_TCI_MAX_Z:
            _tstat["blank"] += 1
            return None
        ids = await _items_for_tile(z, x, y, year)
        if not ids:
            _tstat["blank"] += 1
            return None
        t0 = time.time()
        if mode == "ndvi":
            return await _ndvi_tile(key, akey, ids, z, x, y, t0)
        T = RASTER_TILE
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        tx0, ty1 = -world / 2 + x * world / n, world / 2 - y * world / n
        xs = tx0 + (np.arange(T) + 0.5) * tpx
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        out = np.zeros((T, T, 4), np.uint8)
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)
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
            n_new = int(valid.sum())
            fy = _fill.setdefault(year, [0, 0])
            fy[1] += n_new
            if _items[iid].get("fill"):
                fy[0] += n_new
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            _arr[akey] = None
            return None
        _arr[akey] = out
        if len(_arr) > 2000:
            _arr.pop(next(iter(_arr)))
        png = _encode(key, out)
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    def s2_set_scale(v):
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def s2_raster_stats():
        fill = {y: f / p for y, (f, p) in _fill.items() if f and p}
        return dict(_tstat, cached=len(_png), scale=_gain["v"], fill=fill)

    return s2_raster_stats, s2_set_scale, s2_tile_png


@app.cell
def _(
    CACHE_DIR,
    GeoTIFF,
    Image,
    LS_BANDS,
    LS_BUCKET,
    LS_CONNECTIONS,
    LS_ENDPOINT,
    LS_KEY,
    LS_MAX_Z,
    LS_PATH,
    LS_PERIOD,
    LS_PREFETCH,
    LS_SECRET,
    LS_TILE_MIN_Z,
    LS_WIN_MAX,
    LS_YEARS,
    RASTER_TILE,
    S3Store,
    Window,
    asyncio,
    io,
    json,
    math,
    ndvi_rgb,
    np,
    os,
    time,
):
    # ---- Landsat tiles, by YEAR and MODE: the left pane's other sensor --------
    # The same shape as the Sentinel-2 tile cell: the browser's TileLayer asks
    # for (z, x, y, year, mode) and gets one PNG. No STAC: the bi-monthly
    # mosaics are 1 by 1 degree cells (EPSG:4326, 0.00025 deg, uint8 per
    # band, COGs with overviews) at a path that follows from the year, the
    # period and the cell name (LS_PATH), so a tile knows its files from its
    # own bounds. A missing file (ocean, no data) is remembered as None. Every
    # read waits on LS_CONNECTIONS (the CDSE quota). Tiles are cached in
    # memory and on disk by (year, mode, z, x, y): a second session reads
    # nothing from Copernicus. ONE stretch for every tile of every year: the
    # p2-p98 of the coarsest overview of the first file opened, written next
    # to the cache; delete `stretch.json` there to lock a new one. Once the
    # live requests go quiet a worker fetches the same tiles for the year
    # below (LS_PREFETCH years), because the slider walks down from 2022.
    # (Was a still per box, fetched on a key; Stephen, 2026-09-04: "only one
    # tile is updating, the rest of the tiles are staying the same".)
    _dir = os.path.join(CACHE_DIR, "landsat")
    os.makedirs(_dir, exist_ok=True)
    _store = (
        S3Store(LS_BUCKET, endpoint=LS_ENDPOINT, access_key_id=LS_KEY, secret_access_key=LS_SECRET,
                region="default", virtual_hosted_style_request=False)
        if LS_KEY and LS_SECRET else None
    )
    _R = 6378137.0
    _open = {}
    _sem = asyncio.Semaphore(LS_CONNECTIONS)
    _png = {}
    _gain = {"v": 1.0}
    _stat = {"served": 0, "blank": 0, "ms": 0.0, "bytes": 0, "live": 0, "prefetched": 0}
    _stretch = {"v": None}
    _stretch_fp = os.path.join(_dir, "stretch.json")
    if os.path.exists(_stretch_fp):
        try:
            with open(_stretch_fp) as _f:
                _stretch["v"] = [tuple(v) for v in json.load(_f)]
        except Exception:
            pass
    _queue = []
    _queued = set()
    _worker = {"task": None}

    def ls_ready():
        return _store is not None

    def ls_bytes():
        return _stat["bytes"]

    def ls_raster_stats():
        return dict(_stat, cached=len(_png), stretch=_stretch["v"])

    def ls_set_scale(v):
        """The picture scale slider: one gain for both sensors' true colour."""
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def _cell_name(lat0, lon0):
        """The mosaic cell whose south-west corner is (lat0, lon0), named by
        the corner nearest the equator and the prime meridian: [-63, -12,
        -62, -11] is 11S062W, [-101, 40, -100, 41] is 40N100W, [120, -31,
        121, -30] is 30S120E (probed 2026-09-04)."""
        la = lat0 if lat0 >= 0 else -(lat0 + 1)
        lo = lon0 if lon0 >= 0 else -(lon0 + 1)
        return f"{la:02d}{'N' if lat0 >= 0 else 'S'}{lo:03d}{'E' if lon0 >= 0 else 'W'}"

    def _path(year, cell, band):
        return LS_PATH.format(year=year, mm=LS_PERIOD.split("-")[0], period=LS_PERIOD, cell=cell, band=band)

    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    def _fp(year, mode, z, x, y):
        return os.path.join(_dir, str(year), mode, f"{z}_{x}_{y}.png")

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                try:
                    _open[rel] = await GeoTIFF.open(rel, store=_store)
                except Exception:
                    _open[rel] = None
        return _open[rel]

    async def _lock_stretch(g):
        """p2-p98 per band from the coarsest overview of the first file: one
        stretch for the session (and the disk cache), so no seams."""
        if _stretch["v"] is not None:
            return
        lv = [g, *g.overviews][-1]
        async with _sem:
            ra = await lv.read()
        a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1)
        a = a[(a > 0) & (a < 255)]
        if a.size < 100:
            return
        lo, hi = (float(q) for q in np.percentile(a, [2, 98]))
        _stretch["v"] = [(lo, hi if hi > lo else lo + 1.0)] * 3
        with open(_stretch_fp, "w") as f:
            json.dump(_stretch["v"], f)

    async def _band_into(rel, out, lon_c, lat_c, level_px):
        """One band file, nearest-sampled into `out` (T, T) where the file
        covers the tile pixel and the value is nonzero; the bytes read."""
        g = await _get(rel)
        if g is None:
            return 0
        if _stretch["v"] is None:
            await _lock_stretch(g)
        L, _B, R_, Tt = g.bounds
        levels = [g, *g.overviews]
        li = 0
        px0 = (R_ - L) / g.width
        while li + 1 < len(levels) and px0 * 2 ** (li + 1) <= level_px:
            li += 1
        lv = levels[li]
        H, W = lv.shape
        px = (R_ - L) / W
        c0, c1 = max(0, int(math.floor((lon_c[0] - L) / px))), min(W, int(math.ceil((lon_c[-1] - L) / px)) + 1)
        r0, r1 = max(0, int(math.floor((Tt - lat_c[0]) / px))), min(H, int(math.ceil((Tt - lat_c[-1]) / px)) + 1)
        if c1 <= c0 or r1 <= r0 or (c1 - c0) * (r1 - r0) > LS_WIN_MAX:
            return 0
        async with _sem:
            ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, r1 - r0, c1 - c0)[0]
        cols = np.floor((lon_c - (L + c0 * px)) / px).astype(np.int64)
        rows = np.floor(((Tt - r0 * px) - lat_c) / px).astype(np.int64)
        okc, okr = (cols >= 0) & (cols < c1 - c0), (rows >= 0) & (rows < r1 - r0)
        v = a[np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]]
        valid = okr[:, None] & okc[None, :] & (v > 0)
        out[valid] = v[valid]
        return int(a.size) * a.dtype.itemsize

    def _encode(rgba):
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def _gained(png):
        """The true-colour PNG under the picture scale (gain 1 is the PNG)."""
        g = _gain["v"]
        if g == 1.0:
            return png
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        out = np.concatenate([np.clip(a[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8), a[..., 3:]], axis=2)
        return _encode(out)

    def _remember(year, mode, z, x, y, png):
        """The gain-1 PNG (or None: blank) to memory and disk."""
        _png[(year, mode, z, x, y)] = png
        if len(_png) > 6000:
            _png.pop(next(iter(_png)))
        fp = _fp(year, mode, z, x, y)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "wb") as f:
            f.write(png or b"")

    async def _render(year, mode, z, x, y):
        """The gain-1 PNG for the tile, or None (blank): the read."""
        T = RASTER_TILE
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        lon_c = W_ + (np.arange(T) + 0.5) * (E_ - W_) / T
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        ty1 = world / 2 - y * world / n
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        lat_c = np.degrees(np.arctan(np.sinh(ys / _R)))
        level_px = (E_ - W_) / T
        bands = LS_BANDS[:3] if mode == "tc" else (LS_BANDS[0], LS_BANDS[3])
        raw = np.zeros((len(bands), T, T), np.uint8)
        jobs = []
        for lat0 in range(int(math.floor(S_)), int(math.ceil(N_))):
            for lon0 in range(int(math.floor(W_)), int(math.ceil(E_))):
                cell = _cell_name(lat0, lon0)
                for k, band in enumerate(bands):
                    jobs.append(_band_into(_path(year, cell, band), raw[k], lon_c, lat_c, level_px))
        got = await asyncio.gather(*jobs)
        _stat["bytes"] += sum(got)
        if mode == "tc":
            alpha = (raw > 0).all(0)
            if not alpha.any() or _stretch["v"] is None:
                return None
            out = np.zeros((T, T, 4), np.uint8)
            for k in range(3):
                lo, hi = _stretch["v"][k]
                out[..., k] = np.clip((raw[k].astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            out[..., 3] = np.where(alpha, 255, 0)
            return _encode(out)
        rgb, valid = ndvi_rgb(raw[0], raw[1])
        if not valid.any():
            return None
        out = np.zeros((T, T, 4), np.uint8)
        out[..., :3] = rgb
        out[..., 3] = np.where(valid, 255, 0)
        return _encode(out)

    async def _tile(year, mode, z, x, y):
        """The gain-1 PNG or None, from memory, disk, or Copernicus."""
        key = (year, mode, z, x, y)
        if key in _png:
            return _png[key]
        fp = _fp(year, mode, z, x, y)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                png = f.read() or None
            _png[key] = png
            return png
        t0 = time.time()
        png = await _render(year, mode, z, x, y)
        _remember(year, mode, z, x, y, png)
        _stat["served" if png is not None else "blank"] += 1
        _stat["ms"] += 1000 * (time.time() - t0)
        return png

    async def _prefetch_worker():
        try:
            while _queue:
                if _stat["live"] > 0:
                    await asyncio.sleep(0.2)
                    continue
                key = _queue.pop(0)
                _queued.discard(key)
                year, mode, z, x, y = key
                try:
                    if key not in _png and not os.path.exists(_fp(year, mode, z, x, y)):
                        await _tile(year, mode, z, x, y)
                        _stat["prefetched"] += 1
                except Exception:
                    pass
        finally:
            _worker["task"] = None

    def _prefetch(year, mode, z, x, y):
        for i in range(1, LS_PREFETCH + 1):
            yy = year - i
            if yy not in LS_YEARS:
                break
            key = (yy, mode, z, x, y)
            if key in _png or key in _queued:
                continue
            _queue.append(key)
            _queued.add(key)
        while len(_queue) > 2000:
            _queued.discard(_queue.pop(0))
        if _queue and _worker["task"] is None:
            _worker["task"] = asyncio.get_running_loop().create_task(_prefetch_worker())

    async def ls_tile_png(z, x, y, year, mode="tc"):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's Landsat
        mosaic, true colour (`tc`, B03 B02 B01 under the locked stretch and
        the picture scale) or `ndvi` (B03 and B04 on the fixed Emrld scale);
        None below LS_TILE_MIN_Z, above LS_MAX_Z, or where no cell has data."""
        if z < LS_TILE_MIN_Z or z > LS_MAX_Z or year not in LS_YEARS:
            return None
        if _store is None:
            raise RuntimeError("no CDSE key (CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY)")
        _stat["live"] += 1
        try:
            png = await _tile(year, mode, z, x, y)
        finally:
            _stat["live"] -= 1
        _prefetch(year, mode, z, x, y)
        if png is None:
            return None
        return _gained(png) if mode == "tc" else png

    return ls_raster_stats, ls_ready, ls_set_scale, ls_tile_png


@app.cell
def _(duckdb):
    # ---- DuckDB: the click row and the tables under the map -------------------
    con = duckdb.connect()
    return (con,)


@app.cell
def _(
    ALPHA_FILL,
    ALPHA_QUIET,
    CIVIDIS,
    CT_YEARS,
    DIV_RAMP,
    GREENS,
    GREY,
    LOSS_ABS,
    LOSS_FRAC,
    np,
    pa,
):
    # ---- a FRAME: the window's two ends, the change, the loss year, three fills --
    def _ramp(stops):
        st = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in stops], np.float64)
        r = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(st)), st[:, k]) for k in range(3)], 1)
        return r.round().astype(np.uint8)

    DIV, CIV, GRN = _ramp(DIV_RAMP), _ramp(CIVIDIS), _ramp(GREENS)
    _hex = lambda R: ["#%02x%02x%02x" % tuple(int(v) for v in R[i]) for i in range(0, 256, 17)]
    DIV_HEX, CIV_HEX, GRN_HEX = _hex(DIV), _hex(CIV), _hex(GRN)
    _YI = {y: i for i, y in enumerate(CT_YEARS)}

    def build_frame(tab, y0, y1):
        """From the fold's per-year means: the stock at the to-end, the change
        between the ends, the loss year (the first year the drop from the
        running maximum since y0 passes LOSS_FRAC of it and LOSS_ABS), the
        deepest such drop, what came back since the window's minimum, and the
        uncertainty at both ends. Three fills and their legends."""
        tab = tab.sort_by("cell")
        n = tab.num_rows
        A = np.stack([tab[f"a_{y}"].to_numpy(zero_copy_only=False).astype(np.float32) for y in CT_YEARS], 0) if n else np.zeros((len(CT_YEARS), 0), np.float32)
        u0 = tab["u0"].to_numpy(zero_copy_only=False).astype(np.float32) if n else np.zeros(0, np.float32)
        u1 = tab["u1"].to_numpy(zero_copy_only=False).astype(np.float32) if n else np.zeros(0, np.float32)
        i0, i1 = _YI[y0], _YI[y1]
        a0, a1 = A[i0], A[i1]
        change = a1 - a0
        z = np.abs(change) / np.sqrt(np.maximum(u0 ** 2 + u1 ** 2, 1e-6))
        # the loss year: the running maximum from y0, the drop from it
        win = A[i0:i1 + 1]
        runmax = np.maximum.accumulate(np.nan_to_num(win, nan=-1.0), axis=0)
        drop = runmax - np.nan_to_num(win, nan=np.inf)
        passed = (drop >= LOSS_ABS) & (drop >= LOSS_FRAC * runmax) & np.isfinite(win)
        passed[0] = False
        any_loss = passed.any(0)
        first = passed.argmax(0)
        lossyear = np.where(any_loss, y0 + first, -1).astype(np.int64)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lost = np.where(np.isfinite(win).any(0), np.nanmax(np.where(np.isfinite(win), drop, 0), axis=0), np.nan).astype(np.float32) if n else np.zeros(0, np.float32)
            wmin = np.nanmin(win, axis=0) if n else np.zeros(0, np.float32)
            ymin = np.where(np.isfinite(win).any(0), y0 + np.nanargmin(np.where(np.isfinite(win), win, np.inf), axis=0), -1) if n else np.zeros(0, np.int64)
        recovered = (a1 - wmin).astype(np.float32)
        ushare = np.where(a1 > 0, u1 / np.maximum(a1, 1e-6), np.inf).astype(np.float32)
        scored = np.isfinite(change)
        cells = pa.table({
            "cell": tab["cell"],
            "npx": tab["npx"],
            "stock": pa.array(a1.astype(np.float32)),
            "stock0": pa.array(a0.astype(np.float32)),
            "change": pa.array(change.astype(np.float32)),
            "z": pa.array(z.astype(np.float32)),
            "lossyear": pa.array(lossyear.astype(np.int16)),
            "lost": pa.array(lost),
            "recovered": pa.array(recovered),
            "ymin": pa.array(np.asarray(ymin).astype(np.int16)),
            "unc0": pa.array(u0.astype(np.float32)),
            "unc1": pa.array(u1.astype(np.float32)),
            "ushare": pa.array(ushare),
        })
        cellid = tab["cell"].to_numpy().astype(np.uint64)
        ok = change[scored]
        if len(ok) >= 2:
            p2, p98 = (float(q) for q in np.percentile(ok, [2, 98]))
            lim = max(abs(p2), abs(p98), 1.0)
        else:
            lim = 1.0
        t = np.clip((np.where(scored, change, 0) / lim + 1) / 2, 0, 1)
        # loss is the orange side: the ramp runs blue (gain) .. white .. orange (loss)
        rgb_change = DIV[((1 - t) * 255).round().astype(np.int64)]
        a_change = np.where(scored, ALPHA_QUIET + (ALPHA_FILL - ALPHA_QUIET) * np.clip(z, 0, 1), ALPHA_QUIET).round().astype(np.int64)
        span = max(1, y1 - y0)
        ty = np.clip((lossyear - y0) / span, 0, 1)
        rgb_loss = np.where(any_loss[:, None], CIV[(ty * 255).round().astype(np.int64)], np.array(GREY, np.uint8)).astype(np.uint8)
        a_loss = np.where(any_loss, ALPHA_FILL, ALPHA_QUIET)
        st = a1[np.isfinite(a1)]
        s_hi = max(float(np.percentile(st, 98)), 1.0) if len(st) >= 2 else 1.0
        ts = np.clip(np.nan_to_num(a1, nan=0.0) / s_hi, 0, 1)
        rgb_stock = GRN[(ts * 255).round().astype(np.int64)]
        a_stock = np.where(np.isfinite(a1) & (ushare <= 1.0), ALPHA_FILL, ALPHA_QUIET)

        def fill(kind, hit=None):
            if kind == "change":
                c, a = rgb_change, a_change
            elif kind == "lossyear":
                c, a = rgb_loss, a_loss
            else:
                c, a = rgb_stock, a_stock
            return np.ascontiguousarray(np.concatenate([c, a[:, None].astype(np.uint8)], axis=1)).astype(np.uint8)

        def legend(kind):
            if kind == "change":
                return [{"ramp": list(reversed(DIV_HEX)), "lo": f"{y0} to {y1}: rose {lim:.0f} Mg/ha", "hi": f"fell {lim:.0f}",
                         "title": "biomass at the to-end minus the from-end, symmetric about zero on this view's p2/p98; faint where the change is smaller than the uncertainty of its two ends"}]
            if kind == "lossyear":
                return [{"ramp": CIV_HEX, "lo": f"fell in {y0 + 1}", "hi": f"{y1}",
                         "title": f"the first year the biomass fell {100 * LOSS_FRAC:.0f}% below its running maximum since {y0} (and {LOSS_ABS:.0f} Mg/ha)"},
                        {"name": f"no such fall {y0 + 1} to {y1} ({100 * (1 - any_loss.mean()) if n else 0:.0f}%)", "hex": "#%02x%02x%02x" % GREY}]
            return [{"ramp": GRN_HEX, "lo": f"{y1} stock 0", "hi": f"{s_hi:.0f} Mg/ha",
                     "title": "mean biomass at the to-end, stretched to this view's p98; faint where the uncertainty exceeds the stock"}]

        n_loss = int(any_loss.sum())
        n_sig = int((scored & (z >= 1)).sum())
        med = float(np.nanmedian(change)) if scored.any() else float("nan")
        score = (
            f"CTrees {y0} to {y1}: {n:,} cells · median change {med:+.1f} Mg/ha · {n_sig:,} changed past their uncertainty · {n_loss:,} fell past the threshold"
        )
        return {"cells": cells, "cellid": cellid, "A": A, "y0": y0, "y1": y1, "lim": lim, "s_hi": s_hi,
                "fill": fill, "legend": legend, "score": score, "n_loss": n_loss}

    return (build_frame,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class PairMap(anywidget.AnyWidget):
        """Two maplibre maps in a row, one camera. LEFT: the S2 mosaic as tiles
        the kernel renders (custom messages, PNG bytes back), keyed by source
        (`s2` or `ls`), year and mode.
        RIGHT: an H3HexagonLayer (highPrecision) from cell ids + rgba. Hover on
        either pane: h3-js cell at the frame's res, its ring drawn on BOTH.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `config`
        (JSON), `status` / `panel` / `legend` (strings for the strip), custom
        `tile` replies.
        Browser -> kernel: `view` (JSON lon/lat/zoom + the pane's w/h on every
        moveend), `pick` (JSON: the clicked cell as hex, or null), `ctl`
        (JSON: picture year, picture scale, the window, fill, labels
        request)."""

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
            self.tile_fn = None
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c):
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            try:
                png = await self.tile_fn(c.get("src", "s2"), int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]), c.get("mode", "tc"))
            except Exception as e:
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";

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
            head.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;display:flex;flex-direction:column;gap:.3rem;align-items:flex-start;" +
              "max-width:calc(100% - 72px);background:rgba(255,255,255,.94);color:#1d1d1b;padding:4px 8px;border-radius:6px;" +
              "box-shadow:0 1px 3px rgba(0,0,0,.18);white-space:nowrap;font-variant-numeric:tabular-nums";
            pane.append(mapEl, head);
            return {pane, mapEl, head};
          };
          const L = mkPane("left"), R = mkPane("right");
          row.append(L.pane, R.pane);
          const strip = document.createElement("div");
          strip.style.cssText = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";
          const status = document.createElement("div");
          status.className = "sp-status";
          status.style.cssText = "font:14px ui-sans-serif,system-ui,sans-serif;color:#444;white-space:pre-wrap";
          // two legends, one under each pane: the picture's on the left, the
          // fill's on the right (Stephen, 2026-09-04: "each under their pane")
          const legends = document.createElement("div");
          legends.style.cssText = "display:flex;gap:4px;width:100%";
          const legendL = document.createElement("div");
          legendL.className = "sp-legend-l";
          legendL.style.cssText = "flex:1 1 0;min-width:0;display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          const legend = document.createElement("div");
          legend.className = "sp-legend";
          legend.style.cssText = "flex:1 1 0;min-width:0;display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          legends.append(legendL, legend);
          const panel = document.createElement("div");
          panel.className = "sp-panel";
          panel.style.cssText = "font-size:14px";
          strip.append(legends, panel, status);
          status.hidden = !!cfg.minimal;
          root.append(row, strip);
          el.append(css, root);

          // ---- the controls: the pane headers ------------------------------
          const ACCENT = "#2a5db0";
          const btnCss = font + ";padding:.15rem .55rem;border:0;background:transparent;color:#1d1d1b;cursor:pointer;line-height:1.4;font-variant-numeric:tabular-nums";
          const onCss = (b, on) => { b.style.background = on ? ACCENT : "transparent"; b.style.color = on ? "#fff" : "#1d1d1b"; };
          let s2y = cfg.s2_year, fill = cfg.fill || "change", labelsOn = cfg.labels !== false;
          let picMode = cfg.pic_mode || "tc";  // the picture: true colour or NDVI, both sensors
          let s2scale = Number(cfg.s2_scale) || 1;
          let y0 = cfg.win_from, y1 = cfg.win_to;
          const s2Years = cfg.s2_years || [], picYears = cfg.pic_years || [], lsYears = cfg.ls_years || [];
          const s2First = s2Years.length ? s2Years[0] : 1e9;
          const isLs = (y) => y < s2First;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({act, s2y, s2scale, fill, y0, y1, labels: labelsOn, mode: picMode, n: Date.now()}, extra || {})));
            model.save_changes();
          };
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
          const newRow = (head) => { const r = document.createElement("span"); r.style.cssText = "display:inline-flex;gap:.6rem;align-items:center"; head.appendChild(r); return r; };
          const rowOf = (head) => head.lastElementChild || newRow(head);
          const rowBreak = (head) => { newRow(head); };
          let styleS2 = () => {};
          const fills = (cfg.fills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          const styleFill = mkGroup(R.head, "fill", fills, () => fill, (v) => { fill = v; }, "fill", "sp-fill");
          // the biomass window: a stepped two-handle slider, 2000..2025
          rowBreak(R.head);
          const winYears = cfg.win_years || [];
          const sty = document.createElement("style");
          sty.textContent = [
            ".sp-range{position:relative;width:340px;height:30px}",
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
            ".sp-scale{position:relative;width:120px;height:22px}",
            ".sp-scale input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;-webkit-appearance:none;appearance:none}",
            ".sp-scale input:focus{outline:none}",
            ".sp-scale input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-scale input::-moz-range-track{background:none;height:22px}",
            ".sp-scale input::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-scale .spn{position:absolute;left:8px;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            // the picture year: the scale slider's handle, a long track, the range's ticks
            ".sp-year{height:30px;width:340px}",
            ".sp-year .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-year .tks span{width:0;display:flex;justify-content:center}",
            // the Landsat leg of the track, a shade darker: the other sensor
            ".sp-year .ls{position:absolute;left:8px;top:9px;height:4px;background:rgba(29,29,27,.34);border-radius:2px}",
          ].join("\n");
          el.appendChild(sty);
          // a tick label every fifth year (and the ends); the rest empty
          const tickLabel = (y, i, arr) => (y % 5 === 0 || i === 0 || i === arr.length - 1) ? String(y) : "";
          const winWrap = document.createElement("span");
          winWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const winLab = document.createElement("span"); winLab.textContent = "window";
          winLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const rng = document.createElement("span"); rng.className = "sp-range";
          const trk = document.createElement("span"); trk.className = "trk";
          const spn = document.createElement("span"); spn.className = "spn";
          const tks = document.createElement("span"); tks.className = "tks";
          winYears.forEach((y, i, arr) => { const t = document.createElement("span"); const l = document.createElement("i"); l.style.fontStyle = "normal"; l.textContent = tickLabel(y, i, arr); t.appendChild(l); tks.appendChild(t); });
          const mkRange = () => {
            const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = Math.max(0, winYears.length - 1); r.step = 1;
            r.title = "the biomass window: drag either end (whole years); release to rebuild the frame (no fetch)"; return r;
          };
          const rFrom = mkRange(), rTo = mkRange();
          const winTxt = document.createElement("span");
          winTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:6.5em";
          rng.append(trk, spn, tks, rFrom, rTo);
          winWrap.append(winLab, rng, winTxt);
          rowOf(R.head).appendChild(winWrap);
          const styleWin = () => {
            const i0 = Math.max(0, winYears.indexOf(y0)), i1 = Math.max(0, winYears.indexOf(y1)), n = Math.max(1, winYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = rng.clientWidth - 16;
            spn.style.left = (8 + usable * i0 / n) + "px"; spn.style.width = (usable * (i1 - i0) / n) + "px";
            winTxt.textContent = y0 + " to " + y1;
          };
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(winYears.length - 1, b);
            y0 = winYears[a]; y1 = winYears[b]; styleWin();
          };
          rFrom.addEventListener("input", () => onDrag("from"));
          rTo.addEventListener("input", () => onDrag("to"));
          let winSent = [y0, y1];
          const winRelease = () => { if (y0 !== winSent[0] || y1 !== winSent[1]) { winSent = [y0, y1]; send("win"); } };
          rFrom.addEventListener("change", winRelease);
          rTo.addEventListener("change", winRelease);
          setTimeout(styleWin, 0);
          try { new ResizeObserver(styleWin).observe(rng); } catch (e) {}
          // the picture year: ONE slider, Landsat years then Sentinel-2 years
          const yrWrap = document.createElement("span");
          yrWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const yrLab = document.createElement("span"); yrLab.textContent = "picture";
          yrLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const yr = document.createElement("span"); yr.className = "sp-scale sp-year";
          const yrTrk = document.createElement("span"); yrTrk.className = "trk";
          const yrLs = document.createElement("span"); yrLs.className = "ls";
          const yrSpn = document.createElement("span"); yrSpn.className = "spn";
          const yrTks = document.createElement("span"); yrTks.className = "tks";
          picYears.forEach((y, i, arr) => { const t = document.createElement("span"); const l = document.createElement("i"); l.style.fontStyle = "normal"; l.textContent = tickLabel(y, i, arr); t.appendChild(l); yrTks.appendChild(t); });
          const yri = document.createElement("input"); yri.type = "range"; yri.min = 0; yri.max = Math.max(0, picYears.length - 1); yri.step = 1;
          yri.title = "which picture is drawn: Sentinel-2 2022-2025, Landsat 1997-2021, live tiles both (arrow keys, or [ and ])";
          const yrTxt = document.createElement("span");
          yrTxt.className = "sp-yeartxt";
          yrTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:8em";
          yr.append(yrTrk, yrLs, yrSpn, yrTks, yri);
          yrWrap.append(yrLab, yr, yrTxt);
          rowOf(L.head).appendChild(yrWrap);
          styleS2 = () => {
            const i = Math.max(0, picYears.indexOf(s2y)), n = Math.max(1, picYears.length - 1);
            yri.value = i;
            const usable = yr.clientWidth - 16;
            yrSpn.style.width = Math.max(0, usable * i / n) + "px";
            yrLs.style.width = Math.max(0, usable * Math.max(0, lsYears.length - 1) / n) + "px";
            let t = String(s2y);
            if (isLs(s2y)) t += cfg.ls_ok ? " Landsat" : " Landsat, no key";
            else t += " Sentinel-2";
            yrTxt.textContent = t;
            try { renderLegendL(); } catch (e) {}
          };
          let yrSent = s2y, yrTimer = null;
          const yrRelease = () => { if (yrTimer) { clearTimeout(yrTimer); yrTimer = null; } if (s2y !== yrSent) { yrSent = s2y; send("s2"); } };
          yri.addEventListener("input", () => { s2y = picYears[Number(yri.value)]; styleS2(); update(); if (yrTimer) clearTimeout(yrTimer); yrTimer = setTimeout(yrRelease, 150); });
          yri.addEventListener("change", yrRelease);
          setTimeout(styleS2, 0);
          try { new ResizeObserver(styleS2).observe(yr); } catch (e) {}
          // the picture mode: true colour or NDVI, whichever sensor the slider is
          // on; the NDVI ramp beside it, one fixed scale for every year
          rowBreak(L.head);
          const modes = (cfg.pic_modes || []).map((m) => ({value: m[0], label: m[1]}));
          const styleMode = mkGroup(L.head, "show", modes, () => picMode, (v) => { picMode = v; }, "mode", "sp-mode");
          // the picture's legend, under the left pane: which sensor and year,
          // true colour or the NDVI ramp on its fixed scale
          const renderLegendL = () => {
            legendL.replaceChildren();
            const ls = isLs(s2y);
            const who = ls ? (cfg.ls_ok ? "Landsat " + s2y : "Landsat " + s2y + ", no CDSE key in the environment") : "Sentinel-2 " + s2y;
            const s = document.createElement("span");
            s.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
            if (picMode === "ndvi") {
              const lo = document.createElement("span"); lo.textContent = who + " · NDVI " + (cfg.ndvi_lo != null ? cfg.ndvi_lo : -0.1); lo.style.opacity = ".75";
              const bar = document.createElement("span");
              bar.style.cssText = "display:inline-block;width:11rem;height:12px;border-radius:2px;background:linear-gradient(90deg," + (cfg.ndvi_ramp || []).join(",") + ")";
              const hi = document.createElement("span"); hi.textContent = String(cfg.ndvi_hi != null ? cfg.ndvi_hi : 0.9); hi.style.opacity = ".75";
              s.append(lo, bar, hi);
              s.title = "NDVI = (NIR - red) / (NIR + red), one fixed scale for every year and both sensors; dark low, bright high (Carto Emrld)";
            } else {
              const tx = document.createElement("span"); tx.textContent = who + " · true colour"; tx.style.opacity = ".75";
              s.append(tx);
            }
            legendL.appendChild(s);
          };
          const styleNdvi = () => { renderLegendL(); };
          // the S2 scale
          const SC_MIN = 0.2, SC_MAX = 3;
          const scWrap = document.createElement("span");
          scWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const scLab = document.createElement("span"); scLab.textContent = "scale";
          scLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const scr = document.createElement("span"); scr.className = "sp-scale";
          const scTrk = document.createElement("span"); scTrk.className = "trk";
          const scSpn = document.createElement("span"); scSpn.className = "spn";
          const sc = document.createElement("input"); sc.type = "range"; sc.min = SC_MIN; sc.max = SC_MAX; sc.step = 0.1;
          sc.title = "brightness of the Sentinel-2 mosaic (a gain on the TCI bytes; double-click for 1.0)";
          const scTxt = document.createElement("span");
          scTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4em";
          scr.append(scTrk, scSpn, sc);
          scWrap.append(scLab, scr, scTxt);
          rowOf(L.head).appendChild(scWrap);
          const styleSc = () => {
            sc.value = s2scale;
            scSpn.style.width = Math.max(0, (scr.clientWidth - 16) * (s2scale - SC_MIN) / (SC_MAX - SC_MIN)) + "px";
            scTxt.textContent = s2scale.toFixed(1) + "×";
          };
          let scSent = s2scale, scTimer = null;
          const scRelease = () => { if (scTimer) { clearTimeout(scTimer); scTimer = null; } if (s2scale !== scSent) { scSent = s2scale; send("s2scale"); } };
          const SC_DEBOUNCE_MS = 150;
          sc.addEventListener("input", () => { s2scale = Number(sc.value); styleSc(); if (scTimer) clearTimeout(scTimer); scTimer = setTimeout(scRelease, SC_DEBOUNCE_MS); });
          sc.addEventListener("change", scRelease);
          scr.addEventListener("dblclick", (e) => { e.preventDefault(); s2scale = 1; styleSc(); scRelease(); });
          setTimeout(styleSc, 0);
          try { new ResizeObserver(styleSc).observe(scr); } catch (e) {}
          // the geocoder: Photon, from the browser, flies the left map
          const PHOTON = "https://photon.komoot.io/api/";
          const gcWrap = document.createElement("span");
          gcWrap.style.cssText = "position:relative;display:inline-flex;align-items:center;gap:.4rem";
          const gcLab = document.createElement("span"); gcLab.textContent = "find";
          gcLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const gc = document.createElement("input");
          gc.type = "search"; gc.placeholder = "a place…"; gc.autocomplete = "off"; gc.spellcheck = false;
          gc.title = "Photon geocoder: type, pick a hit (arrows, Enter, or click) and both maps fly there";
          gc.style.cssText = "width:14rem;" + font + ";padding:.15rem .45rem;border:1px solid rgba(29,29,27,.28);border-radius:5px;background:#fff;color:#1d1d1b";
          const gcList = document.createElement("div");
          gcList.className = "sp-hits";
          gcList.style.cssText = "position:absolute;left:0;top:calc(100% + 4px);z-index:9;display:none;min-width:100%;max-width:26rem;" +
            "background:#fff;color:#1d1d1b;border:1px solid rgba(29,29,27,.28);border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.18);overflow:hidden";
          gcWrap.append(gcLab, gc, gcList);
          rowBreak(L.head);
          rowOf(L.head).appendChild(gcWrap);
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const GC_DEBOUNCE_MS = 250, GC_LIMIT = 6;
          const hitName = (f) => {
            const p = f.properties || {};
            const parts = [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null,
              p.county && p.county !== p.city && p.county !== p.name ? p.county : null, p.state, p.country];
            return parts.filter((x) => x).join(", ");
          };
          const hitKind = (f) => { const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(" · "); };
          const gcHide = () => { gcList.style.display = "none"; gcList.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            gcList.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => {
              const row = document.createElement("div");
              row.style.cssText = "padding:.3rem .55rem;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;" +
                (i === gcSel ? "background:" + ACCENT + ";color:#fff" : "");
              const nm = document.createElement("div"); nm.textContent = hitName(f);
              const kd = document.createElement("div"); kd.textContent = hitKind(f);
              kd.style.cssText = "font-size:11px;opacity:" + (i === gcSel ? ".85" : ".6");
              row.append(nm, kd);
              row.onmousedown = (e) => { e.preventDefault(); gcFly(f); };
              row.onmouseenter = () => { gcSel = i; gcShow(); };
              gcList.appendChild(row);
            });
            gcList.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const seq = ++gcSeq;
            const params = new URLSearchParams({q, limit: String(GC_LIMIT), lang: "en"});
            if (mapL) { const c = mapL.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try {
              const r = await fetch(PHOTON + "?" + params.toString());
              const data = await r.json();
              if (seq !== gcSeq) return;
              gcHits = (data.features || []).filter((f) => f.geometry && f.geometry.coordinates);
              gcSel = gcHits.length ? 0 : -1;
              gcShow();
            } catch (e) { if (seq === gcSeq) say("search: " + e.message); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            const w = (L.mapEl.clientWidth || 700);
            let zoom = 10;
            if (ext && ext.length === 4) {
              const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01);
              zoom = Math.log2(360 * (w / 512) / span) - 0.3;
            }
            zoom = Math.max(3.5, Math.min(14, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (mapL) mapL.flyTo({center: [lon, lat], zoom, duration: 2000});
            say("→ " + hitName(f) + " · zoom " + zoom.toFixed(1));
          };
          gc.addEventListener("input", () => { if (gcTimer) clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, GC_DEBOUNCE_MS); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => { setTimeout(gcHide, 120); });
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") {
              e.preventDefault();
              if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]);
              else { if (gcTimer) clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else say("no match: " + gc.value.trim()); }); }
            }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });
          // full screen (the shadow-root walk, docs/12)
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
            styleWin(); styleS2();
          };
          const toggleFull = () => {
            if (isFull()) document.exitFullscreen();
            else root.requestFullscreen().catch((e) => say("fullscreen: " + e.message));
          };
          document.addEventListener("fullscreenchange", () => { setTimeout(paneHeight, 30); });
          window.addEventListener("resize", () => { paneHeight(); });
          const hint = document.createElement("div");
          hint.style.cssText = mono + ";opacity:.55";
          hint.textContent = "keys: [ ] picture year · n true colour / NDVI · ; ' picture scale · 1-3 fill · - = window from · _ + window to · L labels · F full screen · click a hexagon for its story";
          hint.style.color = "#666";
          strip.appendChild(hint);
          hint.hidden = !!cfg.minimal;
          const step = (arr, cur, d) => { const i = arr.indexOf(cur); return arr[Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i) + d))]; };
          root.tabIndex = 0;
          root.addEventListener("pointerup", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            setTimeout(() => { try { root.focus({preventScroll: true}); } catch (err) {} }, 0);
          });
          root.addEventListener("keydown", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            const k = e.key;
            if (k === "[" || k === "]" || k === "ArrowLeft" || k === "ArrowRight") { s2y = step(picYears, s2y, (k === "]" || k === "ArrowRight") ? 1 : -1); styleS2(); update(); yrRelease(); }
            else if (k === ";" || k === "'") { s2scale = Math.round(10 * Math.max(SC_MIN, Math.min(SC_MAX, s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; styleSc(); scRelease(); }
            else if (k >= "1" && k <= "9") { const f = fills[Number(k) - 1]; if (f) { fill = f.value; styleFill(); send("fill"); } }
            else if (k === "-" || k === "=") { const v = step(winYears, y0, k === "=" ? 1 : -1); if (v < y1) { y0 = v; styleWin(); winRelease(); } }
            else if (k === "_" || k === "+") { const v = step(winYears, y1, k === "+" ? 1 : -1); if (v > y0) { y1 = v; styleWin(); winRelease(); } }
            else if (k === "n" || k === "N") { picMode = picMode === "ndvi" ? "tc" : "ndvi"; styleMode(); styleNdvi(); update(); send("mode"); }
            else if (k === "l" || k === "L") { labelsOn = !labelsOn; labels(labelsOn); send("labels"); }
            else if (k === "f" || k === "F") { toggleFull(); }
            else return;
            e.preventDefault();
          });

          const say = (t) => {
            status.textContent = t || "";
            if (cfg.minimal) status.hidden = !/folding|failed|zoom in past|no match|search:|Landsat|no CDSE/.test(t || "");
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

          // ---- the picture tiles, both sensors: ask the kernel ---------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (msg, buffers) => {
            if (!msg) return;
            if (msg.kind !== "tile") return;
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
          const getTileDataFor = (src, year, mode) => ({index, signal}) => new Promise((resolve, reject) => {
            const id = ++tseq;
            tstat.asked++;
            pending.set(id, {resolve, reject});
            model.send({kind: "tile", id, src, year, mode: mode || "tc", x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => {
              pending.delete(id); tstat.abort++;
              const e = new Error("aborted"); e.name = "AbortError"; reject(e);
            });
          });

          // ---- the layers ---------------------------------------------------
          let mapL = null, mapR = null, ovL = null, ovR = null;
          let hover = null;
          const slot = () => cfg.labels_slot || "watername_ocean";
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const outline = (id, h, color, width) => {
            const r = h ? ring(h) : null;
            if (!r) return null;
            return new PathLayer({id, data: [r], getPath: (d) => d, getColor: color,
              widthUnits: "pixels", getWidth: width, widthMinPixels: 1, beforeId: slot()});
          };
          const mkRaster = (src, year, maxZ, extent, visible, mode) => new TileLayer({
            id: src + "-" + year + "-" + (mode || "tc") + (cfg.s2_gen && (mode || "tc") === "tc" ? "-s" + cfg.s2_gen : ""),
            getTileData: getTileDataFor(src, year, mode),
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
            const out = [];
            // one sensor at a time: a Landsat year is its own tile layer (only
            // with a key; without one the pane is the basemap and the legend
            // says why), an S2 year is its tiles
            if (isLs(s2y)) { if (cfg.ls_ok) out.push(mkRaster("ls", s2y, cfg.ls_max_z || 12, null, true, picMode)); }
            else out.push(mkRaster("s2", s2y, 14, null, true, picMode));
            const h = outline("hover-l", hover, [255, 255, 255, 255], 2);
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked-l", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function layersRight() {
            const out = [];
            if (dataObj && hexZoomOk()) out.push(new H3HexagonLayer({
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
          function update() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
          }
          function updateHover() { update(); }

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
            if (v.zoom >= (cfg.hex_zoom || 9)) say("folding hexagons…");
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
              attributionControl: {compact: false},
            });
            mapL = mk(L.mapEl); mapR = mk(R.mapEl);
            mapL.keyboard.disable(); mapR.keyboard.disable();
            mapR.addControl(new maplibregl.FullscreenControl({container: root}), "top-right");
            mapR.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            ovL = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck L: " + (e && e.message ? e.message : e))});
            ovR = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck R: " + (e && e.message ? e.message : e))});
            mapL.addControl(ovL); mapR.addControl(ovR);
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
            const onLoad = () => { ready++; if (ready === 2) { labels(labelsOn); update(); sendView(); } };
            mapL.on("load", onLoad); mapR.on("load", onLoad);
            mapL.on("moveend", sendView); mapR.on("moveend", sendView);
            mapL.on("zoom", () => update());
            mapR.on("zoom", () => update());
            for (const m of [mapL, mapR]) {
              m.on("mousemove", (e) => {
                const h = cellAt(e.lngLat);
                if (h !== hover) { hover = h; updateHover(); }
              });
              m.on("mouseout", () => { if (hover) { hover = null; updateHover(); } });
              m.on("click", (e) => {
                const h = cellAt(e.lngLat);
                model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, n: ++seq}));
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
            if (Number(cfg.s2_scale) && Number(cfg.s2_scale) !== s2scale) { s2scale = Number(cfg.s2_scale); scSent = s2scale; styleSc(); }
            if (cfg.fill && cfg.fill !== fill) { fill = cfg.fill; styleFill(); }
            if (cfg.pic_mode && cfg.pic_mode !== picMode) { picMode = cfg.pic_mode; styleMode(); styleNdvi(); }
            if (cfg.labels !== was.labels) { labelsOn = cfg.labels !== false; labels(labelsOn); }
            styleS2(); styleNote();
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
def _(
    CT_YEARS,
    FILLS,
    FILL_NAMES,
    FILL_SHORT,
    HEX_ZOOM,
    HOME,
    LABELS_SLOT,
    LS_MAX_Z,
    LS_YEARS,
    NDVI_HEX,
    NDVI_HI,
    NDVI_LO,
    PIC_MODES,
    PIC_YEARS,
    PairMap,
    RASTER_TILE,
    S2_SCALE0,
    S2_YEAR0,
    S2_YEARS,
    STRIP_MINIMAL,
    VIEW_H,
    WIN_FROM0,
    WIN_TO0,
    json,
    ls_ready,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    pair = PairMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "fill": FILLS[0],
        "s2_years": list(S2_YEARS), "ls_years": list(LS_YEARS), "pic_years": list(PIC_YEARS),
        "ls_ok": bool(ls_ready()), "ls_max_z": LS_MAX_Z,
        "pic_mode": "tc", "pic_modes": [list(m) for m in PIC_MODES], "ndvi_ramp": NDVI_HEX, "ndvi_lo": NDVI_LO, "ndvi_hi": NDVI_HI,
        "win_from": WIN_FROM0, "win_to": WIN_TO0, "win_years": list(CT_YEARS),
        "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "hex_zoom": HEX_ZOOM,
        "minimal": STRIP_MINIMAL,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2y": S2_YEAR0, "s2scale": S2_SCALE0, "s2gen": 0, "fill": FILLS[0], "labels": True, "mode": "tc",
        "y0": WIN_FROM0, "y1": WIN_TO0,
        "hit": None, "memo": {}, "ct": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
    }
    pair
    return HOLD, pair


@app.cell
def _(
    CELL_KM2,
    CT_YEARS,
    FILLS,
    FILL_NAMES,
    HEX_ZOOM,
    HOLD,
    HOME,
    LS_YEARS,
    PIC_MODES,
    PIC_YEARS,
    SETTLE,
    STRIP_MINIMAL,
    asyncio,
    build_frame,
    con,
    contains,
    ct_fold,
    json,
    ls_raster_stats,
    ls_set_scale,
    ls_tile_png,
    np,
    pad_box,
    pair,
    res_for_view,
    s2_raster_stats,
    s2_set_scale,
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

    async def _tile_fn(src, z, x, y, year, mode="tc"):
        if src == "ls":
            return await ls_tile_png(z, x, y, year, mode)
        return await s2_tile_png(z, x, y, year, mode)

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

    def _ls_line():
        st = ls_raster_stats()
        if not st["served"] and not st["blank"] and not st["bytes"]:
            return ""
        return (f" · Landsat tiles {st['served']:,} served, {st['blank']:,} empty, {st['prefetched']:,} ahead"
                f" · {st['bytes'] / 1e6:.0f} MB from Copernicus this session")

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · zoom in past {HEX_ZOOM:g} for the hexagons (CTrees has no pyramid to draw below it)")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and res_for_view(vsd, box) <= HOLD["res"]:
            if not force:
                _say(HOLD.get("last_status", "") + " · held")
                return
            # a window change inside the held box keeps the held box and res,
            # so the biomass window read is a cache hit and only the two
            # uncertainty ends are new
            box, res = HOLD["box"], HOLD["res"]
        else:
            res = res_for_view(vsd, box)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (y0, y1, res, rbox)
        t0 = time.time()
        _say("folding CTrees, all 26 years…" if STRIP_MINIMAL else f"res {res} · folding CTrees 2000..2025… (wiring run {HOLD['runs']})")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox, y0, y1)
            if bkey not in HOLD["ct"]:
                HOLD["ct"][bkey] = await ct_fold(box, res, y0, y1)
                if len(HOLD["ct"]) > 24:
                    HOLD["ct"].pop(next(iter(HOLD["ct"])))
            tab, s1 = HOLD["ct"][bkey]
            if tab is None or tab.num_rows == 0:
                _say(f"res {res} · {s1}")
                return
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, tab, y0, y1)
            stats = f"res {res} · {s1} · frame {time.time() - t1:.1f} s"
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 24:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        st = s2_raster_stats()
        HOLD["last_status"] = (
            f"{stats} · {fr['score']}"
            + f" · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
            f" · S2 tiles {st['served']:,} served, {st['blank']:,} empty"
            + "".join(f" · S2 {y}: {100 * v:.0f}% of pixels backfilled from the temporal median" for y, v in sorted(st["fill"].items()))
            + _ls_line()
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

    def _f(v, d=1):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    def _spark(series, y0, y1, unc=None):
        """A small inline SVG of the 26-year series, the window shaded."""
        W, H, pad = 260, 56, 4
        ys = np.array(series, np.float64)
        ok = np.isfinite(ys)
        if not ok.any():
            return ""
        top = max(float(np.nanmax(ys)), 1.0)
        n = len(ys)
        xs = [pad + i * (W - 2 * pad) / (n - 1) for i in range(n)]
        pts = " ".join(f"{xs[i]:.1f},{H - pad - (ys[i] / top) * (H - 2 * pad):.1f}" for i in range(n) if ok[i])
        i0, i1 = CT_YEARS.index(y0), CT_YEARS.index(y1)
        band = ""
        if unc is not None:
            us = np.array(unc, np.float64)
            hi = " ".join(f"{xs[i]:.1f},{H - pad - min(1.0, (ys[i] + us[i]) / top) * (H - 2 * pad):.1f}" for i in range(n) if ok[i] and np.isfinite(us[i]))
            lo = " ".join(f"{xs[i]:.1f},{H - pad - max(0.0, (ys[i] - us[i]) / top) * (H - 2 * pad):.1f}" for i in reversed(range(n)) if ok[i] and np.isfinite(us[i]))
            band = f"<polygon points='{hi} {lo}' fill='#2a5db0' opacity='.12'/>"
        return (
            f"<svg width='{W}' height='{H}' style='vertical-align:middle;margin-left:.6rem'>"
            f"<rect x='{xs[i0]:.1f}' y='{pad}' width='{xs[i1] - xs[i0]:.1f}' height='{H - 2 * pad}' fill='#e07b1c' opacity='.13'/>"
            f"{band}<polyline points='{pts}' fill='none' stroke='#2a5db0' stroke-width='1.5'/>"
            f"<text x='{xs[0]:.0f}' y='{H - 1}' font-size='8' fill='#888'>{CT_YEARS[0]}</text>"
            f"<text x='{xs[-1] - 18:.0f}' y='{H - 1}' font-size='8' fill='#888'>{CT_YEARS[-1]}</text>"
            f"<text x='{pad}' y='9' font-size='8' fill='#888'>{top:.0f} Mg/ha</text></svg>"
        )

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
                pair.panel = ""
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT stock0, stock, change, z, lossyear, lost, recovered, ymin, unc0, unc1, ushare "
                "FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                pair.panel = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                s0, s1, ch, z, ly, lost, rec, ymin, u0, u1, ush = r
                y0, y1 = fr["y0"], fr["y1"]
                ci = int(np.searchsorted(fr["cellid"], np.uint64(cell)))
                series = fr["A"][:, ci] if ci < len(fr["cellid"]) else []
                useries = None  # the uncertainty is read for the window's ends only
                if s0 is None or s1 is None or np.isnan(s0) or np.isnan(s1):
                    l1 = "CTrees has no valid biomass here at one end of the window."
                else:
                    how = "down" if ch < 0 else "up"
                    pct = f" ({100 * abs(ch) / s0:.0f}%)" if s0 > 0 else ""
                    l1 = f"CTrees: <b>{s0:.0f} Mg/ha</b> in {y0}, <b>{s1:.0f}</b> in {y1}: {how} <b>{abs(ch):.0f}</b>{pct}."
                if ly is not None and ly >= 0:
                    l2 = f"The biomass fell past the threshold in <b>{ly}</b> (the deepest drop inside the window {_f(lost, 0)} Mg/ha)"
                    l2 += f"; {_f(rec, 0)} Mg/ha came back since its low in {ymin}." if rec is not None and rec > 1 else "; nothing came back."
                else:
                    l2 = f"No fall past the threshold {y0} to {y1}" + (f"; the low was {ymin}, {_f(rec, 0)} Mg/ha below {y1}." if rec is not None and rec > 1 else ".")
                if u1 is not None and not np.isnan(u1):
                    zt = "well past" if z >= 2 else ("past" if z >= 1 else "within")
                    l3 = f"Uncertainty ±{u1:.0f} Mg/ha at {y1} ({100 * min(ush, 9.99):.0f}% of the stock); the change is {zt} it ({_f(z, 1)}×)."
                else:
                    l3 = "No uncertainty here."
                detail = f"{CELL_KM2.get(HOLD['res'], 0):.3f} km²{where}"
                pair.panel = (
                    f"<div style='font-size:14px;line-height:1.5'>{l1}<br>{l2}<br>{l3}{_spark(series, y0, y1, useries)}</div>"
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
            if y in PIC_YEARS and y != HOLD["s2y"]:
                HOLD["s2y"] = y
                _cfg(s2_year=y)
                _say((HOLD.get("last_status") or "") + (f" · Landsat {y}" if y in LS_YEARS else f" · Sentinel-2 {y}"))
            return
        if act == "s2scale":
            try:
                v = float(min(3.0, max(0.2, float(c.get("s2scale", HOLD["s2scale"])))))
            except (TypeError, ValueError):
                return
            # one gain for both sensors' true colour
            if s2_set_scale(v) | ls_set_scale(v):
                HOLD["s2scale"] = v
                HOLD["s2gen"] += 1
                _cfg(s2_scale=v, s2_gen=HOLD["s2gen"])
                _say((HOLD.get("last_status") or "") + f" · picture scale {v:.1f}× · tiles re-served")
            return
        if act == "win":
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in CT_YEARS and b in CT_YEARS and a < b and (a, b) != (HOLD["y0"], HOLD["y1"]):
                HOLD["y0"], HOLD["y1"] = a, b
                _cfg(win_from=a, win_to=b)
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
        if act == "mode":
            m = c.get("mode", "tc")
            if m in dict(PIC_MODES) and m != HOLD["mode"]:
                HOLD["mode"] = m
                _cfg(pic_mode=m)
                _say((HOLD.get("last_status") or "") + f" · picture: {dict(PIC_MODES)[m]}")
            return
        if act == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
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
    settles): `npx` (sampled CTrees pixels), `stock0` / `stock` (Mg/ha at the
    window's two ends), `change`, `z` (the change over the uncertainty of
    its two ends), `lossyear` (-1 none), `lost` (the deepest drop from the
    running maximum), `recovered` (since the window's minimum, in `ymin`),
    `unc0` / `unc1`, `ushare` (uncertainty over stock at the to-end).
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
    by_year = mo.sql(
        """
        SELECT lossyear, count(*) AS cells, round(avg(lost), 1) AS mean_drop, round(avg(stock0), 1) AS mean_stock_from,
               round(avg(stock), 1) AS mean_stock_to
        FROM view_cells GROUP BY lossyear ORDER BY lossyear
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
