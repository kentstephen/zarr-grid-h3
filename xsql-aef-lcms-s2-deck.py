# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "geoarrow-rust-core",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
# ]
# ///
"""Land change, seen or not by AlphaEarth, across CONUS: LCMS Change x AEF x Sentinel-2.

The xsql-aef-lulc-s2-deck.py chassis with the land-cover leg swapped for the
USFS Landscape Change Monitoring System's annual CHANGE product (v2025-11, 30 m,
CONUS, 1999-2023 less 2016), read as plain COGs from source.coop
(`ganzk/lcms/change/LCMS_CONUS_v2025-11_Change_<year>.tif`, EPSG:5071 Conus
Albers, uint8 Level 3 codes 1..16, nodata 255, six overviews that keep the
codes). LCMS is a TRANSITION raster, not a state raster: each pixel says what
happened to it that year (Wildfire, Tree Removal, Insect/Disease/Drought
Stress, ... , Vegetation Successional Growth, or Stable), so the question to
AlphaEarth is no longer "does the embedding back this word" but "did the
embedding MOVE where LCMS says something happened, and stay put where it says
Stable". Four AlphaEarth years are folded per view (Y-2 .. Y+1) and every
cell gets a displacement: 1 - cos between its mean vector in consecutive years,
the largest of the three steps (pre: Y-2 -> Y-1, in: Y-1 -> Y, out: Y -> Y+1).
The steps either side are there because LCMS's annual composite is a
growing-season window, so a late-season disturbance lands in the FOLLOWING
year's map (measured over the Dixie Fire, Jul-Oct 2021: 13% Wildfire in the
2021 layer, 27% in 2022) while the embedding may have moved the year before
the label (with only the in and out steps, 31% of the 2022 Wildfire cells had
moved; the 2021 embedding already carried the burn).

Per view, the cells LCMS calls Stable set the baseline: D0 is the displacement
below which (1 - FA) of them sit, and a cell "moved" when it is above D0. The
agreement is a sigmoid on (disp - D0) for a change cell and its complement for
a stable cell, so every cue the WorldCover build had (color by agreement,
agreement coverage, highlight, boundaries, fill, hide agreeing, threshold)
works unchanged. Two paints are new: the VERDICT (both / LCMS only / AEF only /
neither) and the raw displacement on the ramp; the clusters paint clusters the
DIFFERENCE vectors (Y-1 -> Y), so a cluster is a kind of change, not a kind of
ground. Earth Genome's Sentinel-2 yearly mosaic (2022-2024 only) rides along
as before: raster, underlay and indices per cell, NBR being the one that speaks
to fire.

Run: uv run marimo edit xsql-aef-lcms-s2-deck.py   (or --sandbox)

Attribution: LCMS is the USDA Forest Service's (public domain). "The AlphaEarth
Foundations Satellite Embedding dataset is produced by Google and Google
DeepMind." (CC-BY 4.0.)
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
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/x-sql-marimo/blob/main/xsql-aef-lcms-s2-deck.py)

    # Land change, seen or not by AlphaEarth, across CONUS

    Fly anywhere in the lower 48. When the map settles, the ground in view is
    folded to H3 at the resolution the zoom deserves (res 5 to 10): the
    **USFS LCMS Change** layer for the year (30 m, 16 change classes; per
    hexagon the share of pixels carrying a change code and which change it
    mostly is), the **AlphaEarth Foundations embedding** for the year, the two before it and
    the one after (the mean 64-vector per hexagon per year) and **Earth
    Genome's Sentinel-2 yearly mosaic** as mean spectral indices per hexagon.

    LCMS says what *happened* to a pixel that year, so the embedding is asked a
    different question than in the land-cover build: did it **move**? Each
    hexagon's displacement is 1 - cos between its vectors in consecutive years,
    the largest of three steps: into the year before, into the year, out of it (LCMS's
    growing-season composite books late-season disturbance in the following year's
    map, so the embedding often moves a year before the label). The hexagons LCMS calls Stable set the view's baseline: **D0**
    is the displacement below which 95% of them sit, and a cell **moved** when
    it is above D0. The **agreement** is how far above D0 a change cell sits
    (or how far below, a stable one).

    - **LCMS raster** / **Sentinel-2 raster**: the two rasters as tiles the
      kernel renders; either can stay UNDER a hexagon paint (`raster under`).
    - **LCMS H3** paint: each hexagon in the color of its main change class (or
      Stable), with the two agreement toggles: **color by agreement** (a
      cool-to-warm ramp) and **agreement coverage** (each hexagon shrunken where
      the embedding does not back LCMS's word). **highlight disagreement**
      reverses either.
    - **verdict H3** paint: the 2x2. Both say change; LCMS only (a change the
      embedding did not see, or one it will see next year); AEF only (the
      embedding moved where LCMS says Stable); neither.
    - **AEF shift H3** paint: the raw displacement on the ramp, stretched to the
      view.
    - **Sentinel-2 H3** paint: one index (NDVI, NDWI, NDBI, NBR, MNDWI) per
      hexagon on the ramp. NBR is the one that speaks to fire.
    - **AEF Δ clusters H3** paint: the year-to-year DIFFERENCE vectors clustered
      (k-means), the legend saying what each cluster is made of in LCMS terms:
      a cluster is a kind of change, not a kind of ground.
    - **boundaries** toggle (with the threshold slider): outlines every patch of
      touching cells below the threshold **that share a class**; **fill**
      paints each patch by the average of its cells; **hide agreeing** draws
      only the doubtful cells.

    Click a hexagon for its story; click a legend chip to isolate a class,
    verdict or cluster; *analyze* tables the view per change class with the
    share the embedding saw, and per verdict. D0 and the clusters are
    recomputed per view (local, honest, colors shift).

    | leg | data | engine |
    |---|---|---|
    | change | USFS LCMS Change v2025-11 (`ganzk/lcms/change` on source.coop; COGs, 30 m, EPSG:5071, 1999-2023 less 2016, CONUS) | obstore + async-geotiff (the overview nearest the pitch), pyproj, DataFusion fold (h3 UDF) |
    | embeddings, four years | `tge-labs/aef` COGs' overviews (80..2560 m, per UTM tile); `tge-labs/aef-mosaic` (10 m Zarr) past res 10 | obstore + async-geotiff, pyproj per tile, one DataFusion fold per year |
    | spectral indices | Earth Genome `sentinel2-yearly-mosaics` (STAC per box; band COGs on source.coop, EPSG:3857 + pyramid; 2022-2024 only) | obstore + async-geotiff, DataFusion fold (h3 UDF, indices as `avg()` expressions) |
    | score | join on cell, displacement, the stable baseline D0, sigmoid, k-means on the differences | numpy; DuckDB for the tables |
    | boundaries | the low cells grouped by *touching + same class*, dissolved per patch | DuckDB h3 (`h3_grid_disk`, `h3_cells_to_multi_polygon_wkb`) + spatial (`ST_Dump`, `ST_Boundary`), GeoArrow to deck |

    Next leg, not built: MTBS burn severity per year as res-10 H3 parquet
    (`cboettig/fire/mtbs-severity-1984-2024-conus` on source.coop) joins on the
    cell id and would referee the Wildfire class from a third side.
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # THE YEAR: the LCMS Change layer in view. Everything else is keyed off it:
    # AlphaEarth is folded for YEAR + each offset in AEF_WINDOW (the step INTO
    # the year and the step OUT of it; the out leg is the answer to LCMS's
    # composite lag, see the module docstring), the Sentinel-2 mosaic is the
    # same year where one exists (2022-2024 only; earlier LCMS years get the
    # nearest mosaic and the strip says so).
    YEAR = 2022
    # offsets from YEAR. Every CONSECUTIVE pair in the window is one step and a
    # cell's shift is the largest step: Y-2 -> Y-1 (`pre`: LCMS books a
    # late-season event in the NEXT year's map, so the embedding may have moved
    # the year before the label; measured over the Dixie Fire, burned Jul-Oct
    # 2021, labelled mostly in LCMS 2022: with (-1, 0, 1) only 31% of the
    # Wildfire cells moved, the 2021 embedding already carrying the burn),
    # Y-1 -> Y (`in`), Y -> Y+1 (`out`). Fewer years is a cheaper read (the AEF
    # reads run concurrently, so the wall time barely moves) and a blinder one.
    AEF_WINDOW = (-2, -1, 0, 1)
    LCMS_YEARS = tuple(y for y in range(1999, 2024) if y != 2016)  # what the bucket holds
    AEF_YEARS_AVAILABLE = range(2017, 2026)
    S2_YEARS_AVAILABLE = range(2022, 2025)
    YEAR_S2 = min(max(YEAR, S2_YEARS_AVAILABLE.start), S2_YEARS_AVAILABLE.stop - 1)

    # The zoom -> H3 ladder (the nlcd-zoom notebook's): BASE_RES at ZOOM0, one step
    # finer every PER_RES zoom units, clamped, then coarsened until the view's
    # expected cell count fits CELL_BUDGET.
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    # res 10 (~17 pixels of 30 m per cell) is the 30 m product's floor; res 11
    # would be 2.4 pixels a cell
    MIN_RES, MAX_RES = 5, 10
    CELL_BUDGET = 150_000
    # LCMS is read at the COG level whose pixel gives about this many pixels per
    # cell (30 m native .. 1920 m, the pyramid's range); LC_MAX_PX caps one
    # fold's window. NOTE (measured 2026-08-31 over the Dixie Fire): the
    # overviews keep the class codes and a CONTIGUOUS change (wildfire, 13% of
    # the box) holds its share at every level, but SPECKLE change (drought
    # stress, 17% at 30 m) erodes toward Stable as the level coarsens (13% at
    # 120 m, 9% at 480 m, 6% at 1920 m). Zoomed out, LCMS under-counts the
    # salt-and-pepper classes; the fold reads the finest level the budget allows.
    LC_PX_PER_CELL = 30
    LC_MAX_PX = 12_000_000
    # Which AlphaEarth source and level each res reads. Mosaic from MOSAIC_MIN_RES
    # up (native 10 m; unreached with MAX_RES 10); below that the COG overview
    # index (0 = 20 m, 1 = 40 m, 2 = 80 m, 3 = 160 m, 4 = 320 m, 5 = 640 m,
    # 6 = 1280 m, 7 = 2560 m), picked for ~15-50 overview pixels per cell.
    MOSAIC_MIN_RES = 11
    # ONE LEVEL COARSER than the WorldCover build for res 5-9 (2026-08-31, after
    # Stephen's first flight, "the buttons are extremely unresponsive"): four
    # years of 64-band int8 windows were ~200 MB per fold, 10-17 s over a home
    # link (a single year's opens + reads measured 0.9 s + 2 s; the rest is
    # bytes). One level coarser is 4x fewer bytes at ~7 px per cell (res 8 at
    # 320 m), and the displacement compares the SAME pixels across years, so
    # the coarser mean changes nothing the score depends on. res 10 keeps 40 m
    # (80 m would be 2.4 px per cell).
    AEF_LEVEL_FOR_RES = {5: 7, 6: 7, 7: 5, 8: 4, 9: 3, 10: 1}
    AEF_MAX_FILES = 2500  # more files than this and the view gets LCMS only
    # Earth Genome's Sentinel-2 yearly mosaic (source.coop earthgenome/
    # earthindeximagery): the SPECTRAL witness beside AlphaEarth. One STAC search
    # per fold box (the `datetime` filter does not constrain these items, so the
    # year is enforced on the item id), the band COGs on the same store as LCMS
    # and AEF. EPSG:3857 with a six-level pyramid on the Web Mercator grid (L0
    # 9.55 m = z14 .. L5 306 m = z9), all bands one grid, nodata 0 (record in
    # docs/s2-mosaic-notes.md). Indices are computed IN THE FOLD SQL from the
    # band means' pixels, never from TCI (a display product).
    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    S2_BANDS = ("b03", "b04", "b08", "b11", "b12")
    # (name, a, b): avg((a - b) / (a + b)) per cell. NDVI vegetation, NDWI water
    # (McFeeters), NDBI built-up, NBR burn / bare, MNDWI water against built.
    S2_INDICES = (
        ("ndvi", "b08", "b04"),
        ("ndwi", "b03", "b08"),
        ("ndbi", "b11", "b08"),
        ("nbr", "b08", "b12"),
        ("mndwi", "b03", "b11"),
    )
    S2_IDX_WHAT = {
        "ndvi": "vegetation", "ndwi": "water", "ndbi": "built-up",
        "nbr": "unburnt / vegetated (low = burn or bare)", "mndwi": "water vs built",
    }
    # pyramid level per res, ~20-55 px per cell (L0 9.55 m, L1 19, L2 38, L3
    # 76, L4 153, L5 306); res 5-7 all read L5, there is nothing coarser
    # (one level coarser than the WorldCover build for res 8-10 too, same
    # reason as AEF_LEVEL_FOR_RES: the S2 read was the next floor at 6-11 s)
    S2_LEVEL_FOR_RES = {5: 5, 6: 5, 7: 5, 8: 5, 9: 4, 10: 2, 11: 0, 12: 0}
    S2_MAX_ITEMS = 12  # MGRS footprints (~147 km) under one box; over it, no S2
    S2_DN_OFFSET = 1000  # the C1 baseline's radiometric offset, subtracted before every index
    S2_TILE_MIN_Z, S2_TCI_MAX_Z = 9, 14
    S2_SCALE = 1.0
    RASTER_UNDER_SRC = "nlcd"  # the underlay: "nlcd" (the LCMS raster, internal key) or "s2"

    VIEW_W, VIEW_H = 1400, 720
    PAD = 1.3
    SETTLE = 0.35  # seconds the camera must rest before a fold
    HEX_ZOOM = 9.0  # below it the map is the raster as tiles; from it up the hexagons fold live
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    RASTER_UNDER = False
    RASTER_UNDER_OPACITY = 0.6
    HEX_OPACITY = 1.0
    # home: the northern Sierra (the Dixie Fire, 2021, the largest single fire
    # in California's record; LCMS books most of it in 2022), zoomed to the
    # first hexagon rung
    HOME = {"longitude": -120.95, "latitude": 40.15, "zoom": 9.2}

    # ---- the score ------------------------------------------------------------
    # a cell SAYS CHANGE when at least this share of its LCMS pixels carry a
    # change code (1..14: any disturbance or successional growth); below it the
    # cell is Stable for every purpose here (paint, baseline, verdict)
    CHG_MIN = 0.05
    # the baseline: D0 is the displacement quantile (1 - FA) of the view's stable
    # cells, so FA of them are "moved" by construction (the false-alarm rate)
    FA = 0.05
    # the sigmoid's width on (disp - D0) is the stable cells' robust spread
    # (1.4826 x MAD) times this
    TAU_MAD = 1.0
    MIN_STABLE_CELLS = 30  # fewer stable cells than this and the view is unscored
    K_CLUSTERS = 10
    CLUSTER_HEX = ["#0072B2", "#E69F00", "#56B4E9", "#F0E442", "#CC79A7",
                   "#009E73", "#D55E00", "#999999", "#7B4EA3", "#6B3F1D"]
    ALPHA_MIN, ALPHA_MAX = 30, 235
    COV_MIN = 0.30
    COV_FLAT = 1.00
    ALPHA_FLAT = 190
    # the paints the strip's "agreement coverage" and "color by agreement" toggles
    # are offered on
    ACOV_PAINTS = ("nlcd", "s2", "verdict")
    AGREE_CMAP = "viridis"
    RAMPS = {
        "viridis": "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725",
        "cividis": "00224e00285b002e6a0533711c396f293f6e33446d3c4a6c45506c4d556c555b6d5c616e6467706b6d72727274787877807f78888578908b78979177a09875a89e73b0a571b9ab6dc2b369cbb965d3c05fdcc859e6d051efd748f8df3cfee838",
    }
    ALPHA_RAMP = 225
    DIM_ALPHA = 22
    EDGE_THR = 0.5
    EDGE_MIN_CELLS = 3
    EDGE_ALPHA = 235
    EDGE_WIDTH = 2  # px

    # ---- LCMS Change on source.coop -------------------------------------------
    # `ganzk/lcms/change/LCMS_CONUS_v2025-11_Change_<year>.tif`: 527 MB COGs,
    # 154,180 x 97,279 px at 30 m, EPSG:5071 (NAD83(HARN) Conus Albers), uint8
    # palette codes 1..16, nodata 255, 256 px blocks, deflate, overviews 2..64x
    # that keep the codes. The bucket's zarr pyramids (`pyramids/`) are the same
    # data as float32 with MIN-resampled overviews (a coarse pixel shows the
    # LOWEST code present), useful for "did anything happen here", not for a
    # majority; the COG is the cheaper native read and is what is used.
    LCMS_PREFIX = "ganzk/lcms/change"
    LCMS_NAME = "LCMS_CONUS_v2025-11_Change_{year}.tif"
    LCMS_NODATA = 255
    LCMS_NPA = 16  # Non-Processing Area Mask: treated as no data
    LCMS_STABLE = 15
    LCMS_GROWTH = 14
    LCMS_CRS = "EPSG:5071"
    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-lcms")

    # LCMS Change Level 3 (the 16 codes in the v2025-11 COGs, matched against
    # the embedded colormap and the USFS Levels guidance). NOT the product's
    # colors: three of those are reds (prescribed fire #A10018, wildfire #D54309,
    # southern pine beetle #A64C28) and Stephen has trouble seeing red. The set
    # here separates on blue / orange / purple / yellow and on lightness, with
    # Stable a light grey so change stands out of it. Fire is orange (the warm
    # leg a protanope keeps), tree removal yellow-green, the insect/disease
    # family purples, water blues, the rest by lightness.
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
    # the 2x2 verdict per cell (the `verdict` paint and its legend); codes 200+
    # in the legend's selection space (classes are 1..16, clusters 100+k)
    VERDICTS = {
        0: ("neither", (222, 222, 222)),
        1: ("both say change", (0, 114, 178)),
        2: ("LCMS only", (230, 159, 0)),
        3: ("AEF only", (204, 121, 167)),
    }
    return (
        AEF_INDEX_URL,
        AEF_LEVEL_FOR_RES,
        AEF_MAX_FILES,
        AEF_NODATA,
        AEF_PREFIX,
        AEF_RES,
        AEF_WINDOW,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS_AVAILABLE,
        ACOV_PAINTS,
        AGREE_CMAP,
        ALPHA_FLAT,
        ALPHA_MAX,
        ALPHA_MIN,
        ALPHA_RAMP,
        BASE_RES,
        CACHE_DIR,
        CELL_BUDGET,
        CHG_MIN,
        CLASSES,
        CLUSTER_HEX,
        COV_FLAT,
        COV_MIN,
        DIM_ALPHA,
        EDGE_ALPHA,
        EDGE_MIN_CELLS,
        EDGE_THR,
        EDGE_WIDTH,
        FA,
        HEX_OPACITY,
        HEX_ZOOM,
        HOME,
        K_CLUSTERS,
        LABELS_SLOT,
        LCMS_CRS,
        LCMS_GROWTH,
        LCMS_NAME,
        LCMS_NODATA,
        LCMS_NPA,
        LCMS_PREFIX,
        LCMS_STABLE,
        LCMS_YEARS,
        LC_MAX_PX,
        LC_PX_PER_CELL,
        MAX_RES,
        MIN_RES,
        MIN_STABLE_CELLS,
        MOSAIC_MIN_RES,
        PAD,
        PER_RES,
        RAMPS,
        RASTER_TILE,
        RASTER_UNDER,
        RASTER_UNDER_OPACITY,
        RASTER_UNDER_SRC,
        S2_BANDS,
        S2_COLLECTION,
        S2_DN_OFFSET,
        S2_IDX_WHAT,
        S2_INDICES,
        S2_LEVEL_FOR_RES,
        S2_MAX_ITEMS,
        S2_SCALE,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEARS_AVAILABLE,
        SETTLE,
        TAU_MAD,
        VERDICTS,
        VIEW_H,
        VIEW_W,
        YEAR,
        YEAR_S2,
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
    _CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) from the view; the widget reports
        its canvas size (`w`, `h`) with every move, the constants are the seed."""
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
        return (
            max(-179.9, b[0] - dx),
            max(-85.0, b[1] - dy),
            min(179.9, b[2] + dx),
            min(85.0, b[3] + dy),
        )

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box, dres=0):
        """The ladder's res for this zoom (+ the strip's offset), coarsened until the
        box fits CELL_BUDGET."""
        r = max(MIN_RES, min(MAX_RES, BASE_RES + dres + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / _CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return (
            outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3]
        )

    return box_km2, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule). One context, both folds.
    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(
                coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())
            ),
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
    LCMS_YEARS,
    LC_MAX_PX,
    LC_PX_PER_CELL,
    RASTER_TILE,
    S3Store,
    Transformer,
    Window,
    YEAR,
    asyncio,
    ctx,
    io,
    math,
    np,
    time,
    xr,
):
    # ---- LCMS Change: one CONUS COG per year on source.coop --------------------
    # One file covers CONUS (154,180 x 97,279 px at 30 m in Conus Albers), so a
    # read is one window of one level: the box's corners go to Albers through
    # pyproj (the box edges sampled, as the AEF COG reader does, since a lon/lat
    # rectangle is not a rectangle in Albers), the window is the pixel bounds of
    # that, and the pixel centres come back to lon/lat for the h3 UDF. The fold
    # reads the level nearest the pitch the res deserves (LC_PX_PER_CELL pixels
    # per cell); the raster paint the level nearest the deck tile's own pixel.
    if YEAR not in LCMS_YEARS:
        raise ValueError(f"LCMS has no {YEAR} layer (2016 is missing from the bucket; 1999-2023 otherwise)")
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _fwd = Transformer.from_crs("EPSG:4326", LCMS_CRS, always_xy=True)
    _inv = Transformer.from_crs(LCMS_CRS, "EPSG:4326", always_xy=True)
    _open = {}  # year -> GeoTIFF (headers only)
    _sem = asyncio.Semaphore(16)
    _win = {}  # (year, level, r0, r1, c0, c1) -> uint8 window
    _stats = {"served": 0, "blank": 0, "ms": 0.0, "reads": 0}
    _fold_lock = asyncio.Lock()  # one DataFusion table name (`lc`); folds can overlap
    _LEVELS = 7  # native + six overviews: 30, 60, 120, 240, 480, 960, 1920 m
    _cmap = np.zeros((256, 4), np.uint8)
    for _code, (_nm, _rgb) in CLASSES.items():
        _cmap[_code, :3] = _rgb
        _cmap[_code, 3] = 255
    _cmap[LCMS_NPA, 3] = 0  # the mask and nodata are transparent in the raster
    _cmap[LCMS_NODATA, 3] = 0
    _CELL_M2 = {5: 252.9e6, 6: 36.13e6, 7: 5.161e6, 8: 0.7373e6, 9: 0.1053e6, 10: 15050.0, 11: 2150.0, 12: 307.1}
    # the product's extent in lon/lat (EPSG:5071's area of use, padded): the
    # raster TileLayer's `extent`, so deck never asks for a tile off CONUS
    lc_bounds = (-125.5, 23.5, -66.0, 50.0)

    async def _get(year):
        if year not in _open:
            async with _sem:
                _open[year] = await GeoTIFF.open(f"{LCMS_PREFIX}/{LCMS_NAME.format(year=year)}", store=_store)
        return _open[year]

    def _grid(g, k):
        """(pixel size, x origin, y origin, width, height) of level k: the COG's
        affine is north-up (e < 0) with the origin at the top left."""
        lv = g if k == 0 else g.overviews[k - 1]
        H, W = lv.shape
        t = g.transform
        return lv, t.a * (g.width / W), t.c, t.f, W, H

    async def lc_window(year, k, box):
        """uint8 (h, w) of level k over the lon/lat box, plus the Albers pixel
        size and the x/y of the window's top-left corner. The window is the
        pixel bounds of the box's edges in Albers (20 points a side)."""
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
            _stats["reads"] += 1
            _win[key] = a
            if len(_win) > 600:
                _win.pop(next(iter(_win)))
        return a, px, x0 + c0 * px, y0 - r0 * px

    _png_cache = {}

    async def lc_tile_png(z, x, y):
        """RGBA PNG bytes for Web Mercator tile (z, x, y) of the year's LCMS
        Change, or None where every pixel is nodata / mask. The level is the one
        whose pixel is nearest the deck tile's own (in metres at the tile's
        latitude)."""
        key = (z, x, y)
        if key in _png_cache:
            return _png_cache[key]
        T = RASTER_TILE
        n = 2 ** z
        t0 = time.time()
        lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        if lon1 < lc_bounds[0] or lon0 > lc_bounds[2] or lat1 < lc_bounds[1] or lat0 > lc_bounds[3]:
            _stats["blank"] += 1
            _png_cache[key] = None
            return None
        m_tile = 2 * math.pi * 6378137.0 / (n * T) * math.cos(math.radians((lat0 + lat1) / 2))
        k = max(0, min(_LEVELS - 1, int(round(math.log2(max(m_tile, 30.0) / 30.0)))))
        got = await lc_window(YEAR, k, (lon0, lat0, lon1, lat1))
        if got is None:
            _stats["blank"] += 1
            _png_cache[key] = None
            return None
        arr, px, wx0, wy0 = got
        # the output pixel centres in lon/lat -> Albers -> nearest source pixel
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
            _stats["blank"] += 1
            _png_cache[key] = None
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png_cache[key] = buf.getvalue()
        _stats["served"] += 1
        _stats["ms"] += 1000 * (time.time() - t0)
        if len(_png_cache) > 4000:
            _png_cache.pop(next(iter(_png_cache)))
        return _png_cache[key]

    def lc_raster_stats():
        return dict(_stats, cached=len(_png_cache), blocks=len(_win))

    def lc_raster_clear():
        n_png, n_blk = len(_png_cache), len(_win)
        _png_cache.clear()
        _win.clear()
        _stats.update(served=0, blank=0, ms=0.0, reads=0)
        return n_png, n_blk

    async def lc_fold(box, res, year=YEAR):
        """Per res cell over the box: the majority code over every processed
        pixel (`maj`), the share of pixels carrying a change code (`p_chg`, codes
        1..14: any disturbance or growth) and a disturbance code (`p_dist`,
        1..13), and the majority AMONG the change pixels (`chg`; Stable when
        there are none). Nodata and the Non-Processing mask are left out.
        Returns (arrow table, stats string)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        pitch = min(1920.0, max(30.0, math.sqrt(_CELL_M2[res] / LC_PX_PER_CELL)))
        k = max(0, min(_LEVELS - 1, int(round(math.log2(pitch / 30.0)))))
        got = None
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
                       CAST(sum(CASE WHEN cls < {LCMS_GROWTH} THEN n ELSE 0 END) AS DOUBLE) / sum(n) AS p_dist,
                       CAST(max(n) AS DOUBLE) / sum(n) AS purity
                FROM c GROUP BY cell
            """).to_arrow_table()
        return out, (
            f"LCMS {year} {w:,}x{h:,} px ({30 * 2 ** k:.0f} m) read {tr - t0:.1f} s · albers {t1 - tr:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return lc_bounds, lc_fold, lc_raster_clear, lc_raster_stats, lc_tile_png


@app.cell
async def _(
    AEF_INDEX_URL,
    AEF_LEVEL_FOR_RES,
    AEF_MAX_FILES,
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_WINDOW,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS_AVAILABLE,
    CACHE_DIR,
    GeoTIFF,
    MOSAIC_MIN_RES,
    ObjectStore,
    S3Store,
    Transformer,
    Window,
    YEAR,
    asyncio,
    ctx,
    duckdb,
    math,
    np,
    os,
    pa,
    pq,
    time,
    xr,
):
    # ---- AlphaEarth: two sources, one fold, THREE YEARS -------------------------
    # The WorldCover build's cell with the year made a parameter: `aef_fold(box,
    # res, year)` for each year in AEF_YEARS (YEAR + AEF_WINDOW, clipped to what
    # exists), each with its own COG index slice and its own mosaic time index.
    # The frame joins the years on the cell and takes the displacements.
    AEF_YEARS = tuple(y for y in (YEAR + d for d in AEF_WINDOW) if y in AEF_YEARS_AVAILABLE)
    if YEAR not in AEF_YEARS:
        raise ValueError(f"AlphaEarth has no {YEAR} (2017-2025)")
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _mstore = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, prefix=AEF_PREFIX
    )
    _ds = xr.open_zarr(ObjectStore(_mstore, read_only=True), chunks=None, consolidated=False)
    _ti = {y: int(np.where(_ds.time.values == y)[0][0]) for y in AEF_YEARS}

    # The COG index per year, cached as parquet under tmp (the full index is
    # 302k rows over HTTP, ~10 s; a year's slice, worldwide, is ~34k rows).
    os.makedirs(CACHE_DIR, exist_ok=True)
    _IDX, _PATHS, _CRS = {}, {}, {}
    aef_index = {}
    for _y in AEF_YEARS:
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
        aef_index[_y] = pq.read_table(_idx_path)
        _IDX[_y] = {k: aef_index[_y][k].to_numpy() for k in aef_index[_y].column_names if k not in ("path", "crs")}
        _PATHS[_y] = aef_index[_y]["path"].to_pylist()
        _CRS[_y] = aef_index[_y]["crs"].to_pylist()

    _open = {}  # path -> GeoTIFF (headers only)
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
        """One file's overview window over the box: (int8 (64, h, w), lon, lat) or None.
        Rows and columns go through the file's AFFINE TRANSFORM, not its bounds:
        these COGs are stored SOUTH-UP (transform e = +10, origin at the south
        edge), and a north-up assumption mirrors every tile within its 82 km."""
        g = await _get(_PATHS[year][i])
        ov = g.overviews[li]
        H, W = ov.shape
        t = g.transform
        sx, sy = t.a * (g.width / W), t.e * (g.height / H)  # signed overview pixel sizes
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

    _DEQ = ", ".join(
        f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64)
    )
    _fold_lock = asyncio.Lock()  # one DataFusion table name, folds of several years in flight

    async def _fold_rows(res, box, cols, lat, lon):
        """cols: int8 (64, n); lat/lon (n,). One 1-D Dataset, one fold."""
        W_, S_, E_, N_ = box
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)}
            | {"lat": (("i",), lat), "lon": (("i",), lon)},
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

    async def aef_fold(box, res, year=YEAR):
        """Mean AlphaEarth vector per res cell over the box for one year, from the
        mosaic (res >= MOSAIC_MIN_RES) or the COG overviews. Returns (arrow table
        or None, stats)."""
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
            return out, (
                f"AEF {year} mosaic {emb.shape[2]:,}x{emb.shape[1]:,} px "
                f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
            )
        li = AEF_LEVEL_FOR_RES[res]
        ix = _IDX[year]
        hit = np.where(
            (ix["wgs84_east"] > W_) & (ix["wgs84_west"] < E_)
            & (ix["wgs84_north"] > S_) & (ix["wgs84_south"] < N_)
        )[0]
        if len(hit) == 0:
            return None, f"AEF {year}: no COG tiles under the view"
        if len(hit) > AEF_MAX_FILES:
            return None, f"AEF {year}: {len(hit):,} tiles under the view (> {AEF_MAX_FILES:,}); zoom in for AlphaEarth"
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
            f"AEF {year} ov{li} ({10 * 2 ** (li + 1)} m) {len(parts):,} files {cols.shape[1] / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return AEF_YEARS, aef_fold, aef_index


@app.cell
def _(
    GeoTIFF,
    S2_BANDS,
    Image,
    RASTER_TILE,
    S2_COLLECTION,
    S2_DN_OFFSET,
    S2_INDICES,
    S2_LEVEL_FOR_RES,
    S2_MAX_ITEMS,
    S2_SCALE,
    S2_STAC,
    S2_TCI_MAX_Z,
    S2_TILE_MIN_Z,
    S3Store,
    Window,
    YEAR_S2,
    asyncio,
    ctx,
    io,
    json,
    math,
    np,
    time,
    urllib,
    xr,
):
    # ---- Sentinel-2 (Earth Genome yearly mosaic): the spectral witness ---------
    # Same store, same async_geotiff window reads as NLCD and AEF, simpler
    # arithmetic (the COGs are 3857, so a lon/lat box is a closed-form Mercator
    # window; no pyproj). The bands cross into DataFusion as one 1-D Dataset and
    # every index is an avg() of a per-pixel expression in the fold SQL: the
    # H3 UDF in the GROUP BY (repo rule), the index in the SELECT.
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _R = 6378137.0
    _items = {}  # item id -> {band: path on the store}
    _boxes = {}  # rounded box -> the year's item ids under it
    _open = {}  # path -> GeoTIFF (headers only)
    _sem = asyncio.Semaphore(32)
    _fold_lock = asyncio.Lock()  # one DataFusion table name (`s2`); folds can overlap

    def _stac(box):
        body = json.dumps({"collections": [S2_COLLECTION], "bbox": list(box), "limit": 100}).encode()
        req = urllib.request.Request(S2_STAC, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["features"]

    async def _s2_items(box):
        key = tuple(round(v, 2) for v in box)
        if key not in _boxes:
            loop = asyncio.get_running_loop()
            feats = await loop.run_in_executor(None, _stac, box)
            ids = []
            for f in feats:
                # the year lives in the id (`10SFJ_2024-01-01_2025-01-01`); the
                # STAC datetime filter returns other years for the same footprint
                if not f["id"].endswith(f"{YEAR_S2}-01-01_{YEAR_S2 + 1}-01-01"):
                    continue
                _items[f["id"]] = {
                    b: f["assets"][b.upper()]["href"].split("source.coop/")[1] for b in S2_BANDS
                } | {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox")}
                ids.append(f["id"])
            _boxes[key] = ids
        return _boxes[key]

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                _open[rel] = await GeoTIFF.open(rel, store=_store)
        return _open[rel]

    def _merc(lon, lat):
        return _R * math.radians(lon), _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def _merc_inv(x, y):
        return np.degrees(x / _R), np.degrees(2 * np.arctan(np.exp(y / _R)) - np.pi / 2)

    async def _read_band(rel, target_px, box):
        """One band's window over the box at the pyramid level nearest target_px:
        (uint16 (h, w), left, top, px) in 3857, or None if the box misses the
        file. The bands are THREE grids per item (10 m: B02-B04/B08 on 15360²;
        20 m: B05-B07/B8A/B11/B12 on 8192² with their own origin; 60 m: B01/B09),
        so the caller samples each onto the reference band's pixel centres."""
        g = await _get(rel)
        levels = [g, *g.overviews]
        L, _B, R_, T = g.bounds
        sizes = [(R_ - L) / lv.shape[1] for lv in levels]
        li = int(np.argmin([abs(math.log(px / target_px)) for px in sizes]))
        lv, px = levels[li], sizes[li]
        H, W = lv.shape
        x0, y0 = _merc(box[0], box[1])
        x1, y1 = _merc(box[2], box[3])
        c0, c1 = max(0, int((x0 - L) / px)), min(W, int(math.ceil((x1 - L) / px)))
        r0, r1 = max(0, int((T - y1) / px)), min(H, int(math.ceil((T - y0) / px)))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(r1 - r0, c1 - c0)
        return a, L + c0 * px, T - r0 * px, px

    async def _read_item(iid, li, box):
        """Every band of one item sampled (nearest) onto the reference band's
        level-li pixel centres: {band: 1-D uint16, lat, lon, px}, or None."""
        g = await _get(_items[iid][S2_BANDS[0]])
        L, _B, R_, _T = g.bounds
        target_px = (R_ - L) / [g, *g.overviews][li].shape[1]
        parts = await asyncio.gather(*(_read_band(_items[iid][b], target_px, box) for b in S2_BANDS))
        if any(p is None for p in parts):
            return None
        ref, rl, rt, rpx = parts[0]
        xs = rl + (np.arange(ref.shape[1]) + 0.5) * rpx
        ys = rt - (np.arange(ref.shape[0]) + 0.5) * rpx
        out = {S2_BANDS[0]: ref.ravel()}
        for b, (a, bl, bt, bpx) in zip(S2_BANDS[1:], parts[1:]):
            cols = np.clip(((xs - bl) / bpx).astype(np.int64), 0, a.shape[1] - 1)
            rows = np.clip(((bt - ys) / bpx).astype(np.int64), 0, a.shape[0] - 1)
            out[b] = a[rows[:, None], cols[None, :]].ravel()
        X, Y = np.meshgrid(xs, ys)
        lon, lat = _merc_inv(X, Y)
        return out | {"lat": lat.ravel(), "lon": lon.ravel(), "px": rpx}

    def _refl(b):
        return f"greatest({b} - {S2_DN_OFFSET}, 1)"

    _IDX_SQL = ", ".join(
        f"avg(({_refl(a)} - {_refl(b)}) / ({_refl(a)} + {_refl(b)})) AS {nm}" for nm, a, b in S2_INDICES
    )
    _VALID = " AND ".join(f"{b} > 0" for b in S2_BANDS)  # 0 is nodata in every band

    async def s2_fold(box, res):
        """Mean spectral indices per res cell over the box, from the pyramid level
        the res deserves. Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        try:
            ids = await _s2_items(box)
        except Exception as e:
            return None, f"S2: STAC failed ({e})"
        if not ids:
            return None, "S2: no mosaic under the view"
        if len(ids) > S2_MAX_ITEMS:
            return None, f"S2: {len(ids)} mosaic tiles under the view (> {S2_MAX_ITEMS}); zoom in for Sentinel-2"
        li = S2_LEVEL_FOR_RES[res]
        parts = [p for p in await asyncio.gather(*(_read_item(i, li, box) for i in ids)) if p is not None]
        if not parts:
            return None, "S2: nothing read"
        ds1 = xr.Dataset(
            {b: (("i",), np.concatenate([p[b] for p in parts]).astype(np.float32)) for b in S2_BANDS}
            | {k: (("i",), np.concatenate([p[k] for p in parts])) for k in ("lat", "lon")},
        )
        ds1 = ds1.assign_coords(i=np.arange(ds1.sizes["i"]))
        npx = ds1.sizes["i"]
        t1 = time.time()
        async with _fold_lock:
            try:
                ctx.deregister_table("s2")
            except Exception:
                pass
            ctx.from_dataset("s2", ds1, chunks={"i": 262_144})
            out = ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS ns2, {_IDX_SQL}
                FROM s2
                WHERE {_VALID}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()
        return out, (
            f"S2 L{li} ({parts[0]['px']:.0f} m) {len(parts)} tiles {npx / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    # ---- the TCI tiles (the NLCD tile renderer's shape): STAC once per z9
    # ancestor tile, every footprint under the tile composited in numpy (black =
    # nodata -> alpha 0; the first footprint to paint a pixel wins), one PNG.
    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    async def _items_for_tile(z, x, y):
        d = z - S2_TILE_MIN_Z
        ids = await _s2_items(_tile_ll(S2_TILE_MIN_Z, x >> d, y >> d))
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        out = []
        for i in ids:
            b = _items[i].get("bbox")
            if not b or (b[0] < E_ and b[2] > W_ and b[1] < N_ and b[3] > S_):
                out.append(i)
        return out

    _png = {}   # (z, x, y, scale) -> PNG bytes
    _arr = {}   # (z, x, y) -> the composited RGBA tile, before the gain
    _gain = {"v": float(S2_SCALE)}
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}

    def _encode(key, out):
        """The gain (the strip's `scale`) on the composited bytes, then PNG."""
        g = _gain["v"]
        rgba = out if g == 1.0 else np.concatenate(
            [np.clip(out[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8), out[..., 3:]], axis=2)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png[key] = buf.getvalue()
        if len(_png) > 4000:
            _png.pop(next(iter(_png)))
        return _png[key]

    async def s2_tile_png(z, x, y):
        """PNG bytes for Web Mercator tile (z, x, y) of the TCI mosaic, or None
        (nothing there: below S2_TILE_MIN_Z, or no footprint under the tile)."""
        key = (z, x, y, _gain["v"])
        if key in _png:
            return _png[key]
        if (z, x, y) in _arr:
            # composited already at another scale: re-encode, no read
            return _encode(key, _arr[(z, x, y)]) if _arr[(z, x, y)] is not None else None
        if z < S2_TILE_MIN_Z or z > S2_TCI_MAX_Z:
            _tstat["blank"] += 1
            return None
        ids = await _items_for_tile(z, x, y)
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
        li = S2_TCI_MAX_Z - z
        for iid in ids:
            g = await _get(_items[iid]["tci"])
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
            rgb = a[:, np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]]
            rgb = rgb.transpose(1, 2, 0)
            valid = okr[:, None] & okc[None, :] & (rgb.sum(2) > 0) & (out[..., 3] == 0)
            out[valid, :3] = rgb[valid]
            out[valid, 3] = 255
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            _arr[(z, x, y)] = None
            return None
        _arr[(z, x, y)] = out
        if len(_arr) > 2000:
            _arr.pop(next(iter(_arr)))
        png = _encode(key, out)
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    def s2_set_scale(v):
        """The strip's `scale`: the gain the next S2 tiles are encoded with.
        Returns True when it changed (the caller then re-asks deck for the tiles)."""
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def s2_raster_stats():
        return dict(_tstat, cached=len(_png), scale=_gain["v"])

    def s2_raster_clear():
        n = len(_png)
        _png.clear()
        _arr.clear()
        _tstat.update(served=0, blank=0, ms=0.0)
        return n

    return s2_fold, s2_raster_clear, s2_raster_stats, s2_set_scale, s2_tile_png


@app.cell
def _(
    ACOV_PAINTS,
    AEF_YEARS,
    AGREE_CMAP,
    ALPHA_FLAT,
    ALPHA_MAX,
    ALPHA_MIN,
    ALPHA_RAMP,
    CHG_MIN,
    CLASSES,
    CLUSTER_HEX,
    COV_FLAT,
    COV_MIN,
    DIM_ALPHA,
    FA,
    K_CLUSTERS,
    LCMS_STABLE,
    MIN_STABLE_CELLS,
    RAMPS,
    S2_INDICES,
    TAU_MAD,
    VERDICTS,
    YEAR,
    duckdb,
    io,
    np,
    pa,
    time,
):
    # ---- a FRAME: displacement, the stable baseline, verdicts, clusters, colors ---
    S2_IDX = [nm for nm, _a, _b in S2_INDICES]
    import pyarrow.ipc as pa_ipc
    from geoarrow.rust.core import (
        from_wkb as ga_from_wkb,
        linestring as ga_linestring,
        polygon as ga_polygon,
    )
    from arro3.core import Array as ArroArray, Table as ArroTable
    _PAL = np.array([tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in CLUSTER_HEX], np.uint8)
    _hx = RAMPS[AGREE_CMAP]
    _stops = np.array([[int(_hx[i + j:i + j + 2], 16) for j in (0, 2, 4)] for i in range(0, len(_hx), 6)], np.float64)
    _RAMP = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    RAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in _RAMP[i]) for i in range(0, 256, 17)]
    _VPAL = np.array([VERDICTS[k][1] for k in range(4)], np.uint8)
    _GREY = np.array([128, 128, 128], np.uint8)
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial")
    _E = [f"e{i:02d}" for i in range(64)]
    # the named steps (pre: Y-2 -> Y-1, in: Y-1 -> Y, out: Y -> Y+1); any other
    # consecutive pair in the window is folded into the max unnamed
    STEP_NAMES = {(YEAR - 2, YEAR - 1): "pre", (YEAR - 1, YEAR): "in", (YEAR, YEAR + 1): "out"}

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

    def build_frame(lc_cells, aef_by_year, s2_cells=None):
        """Join the LCMS fold with each year's AlphaEarth fold (LEFT: a cell keeps
        its LCMS word with or without an embedding) and the S2 indices, then:
        the displacement per cell (1 - cos between consecutive years, every
        step in the window, the largest kept), the view's stable baseline D0,
        the agreement, the 2x2 verdict, k-means on the difference vectors."""
        import time as _time
        _tt = {"t": _time.time()}
        _lap = {}

        def lap(name):
            now = _time.time()
            _lap[name] = now - _tt["t"]
            _tt["t"] = now

        # its own cursor: build_frame runs in the executor thread while the
        # hover handler queries the same connection from the kernel thread
        # (Xiong'an, 2026-09-02: "Attempting to execute an unsuccessful or
        # closed pending query result"); a cursor is a separate connection
        # to the same database, so the two no longer cancel each other
        cx = con.cursor()
        cx.register("lc_cells", lc_cells)
        years = [y for y in AEF_YEARS if aef_by_year.get(y) is not None]
        sel = ["l.*"]
        joins = []
        for y in years:
            cx.register(f"aef_{y}", aef_by_year[y])
            sel += [f"a{y}.{e} AS {e}_{y}" for e in _E]
            joins.append(f"LEFT JOIN aef_{y} a{y} USING (cell)")
        j = cx.execute(f"SELECT {', '.join(sel)} FROM lc_cells l {' '.join(joins)} ORDER BY cell").arrow().read_all()
        has_aef = len(years) >= 2
        has_s2 = s2_cells is not None and s2_cells.num_rows > 0
        if has_s2:
            cx.register("s2_cells", s2_cells)
            cx.register("j_cells", j)
            j = cx.execute(
                f"SELECT j_cells.*, {', '.join(S2_IDX)} FROM j_cells LEFT JOIN s2_cells USING (cell) ORDER BY cell"
            ).arrow().read_all()
        n = j.num_rows
        lap("join")
        maj = j["maj"].to_numpy().astype(np.int64)
        chg = j["chg"].to_numpy().astype(np.int64)
        p_chg = j["p_chg"].to_numpy().astype(np.float32)
        p_dist = j["p_dist"].to_numpy().astype(np.float32)
        says = p_chg >= CHG_MIN
        # the cell's class for every purpose here: its main change when it says
        # change, Stable otherwise
        cls = np.where(says, chg, LCMS_STABLE).astype(np.int64)

        def _V(y):
            if y not in years:
                return None
            V = np.stack([j[f"{e}_{y}"].to_numpy(zero_copy_only=False) for e in _E], axis=1).astype(np.float32)
            nrm = np.linalg.norm(V, axis=1)
            V = V / np.maximum(nrm, 1e-9)[:, None]
            V[~np.isfinite(nrm) | (nrm == 0)] = np.nan
            return V

        steps = {nm: np.full(n, np.nan, np.float32) for nm in ("pre", "in", "out")}
        disp = np.full(n, np.nan, np.float32)
        D = None  # the difference vectors of the `in` step (else the first step), for the clusters
        if has_aef and n > 0:
            Vs = {y: _V(y) for y in years}
            for ya, yb in zip(years[:-1], years[1:]):
                if yb != ya + 1:
                    continue
                d = (1.0 - np.einsum("ij,ij->i", Vs[ya], Vs[yb])).astype(np.float32)
                nm = STEP_NAMES.get((ya, yb))
                if nm:
                    steps[nm] = d
                with np.errstate(invalid="ignore"):
                    disp = np.fmax(disp, d)  # NaN only where every step is NaN
                if D is None or nm == "in":
                    D = Vs[yb] - Vs[ya]
            lap("disp")
        scored = ~np.isnan(disp)
        stable_ok = scored & ~says
        if stable_ok.sum() >= MIN_STABLE_CELLS:
            ds = disp[stable_ok].astype(np.float64)
            D0 = float(np.quantile(ds, 1 - FA))
            med = float(np.median(ds))
            tau = max(1e-4, TAU_MAD * 1.4826 * float(np.median(np.abs(ds - med))))
            z = (disp.astype(np.float64) - D0) / tau
            moved = scored & (disp > D0)
            agree = np.where(scored, np.where(says, _sig(z), 1 - _sig(z)), np.nan).astype(np.float32)
        else:
            D0, tau = float("nan"), float("nan")
            moved = np.zeros(n, bool)
            agree = np.full(n, np.nan, np.float32)
        # the 2x2: 0 neither, 1 both, 2 LCMS only, 3 AEF only; unscored cells
        # (no embedding) are -1
        verdict = np.where(~scored, -1, np.where(says & moved, 1, np.where(says, 2, np.where(moved, 3, 0)))).astype(np.int64)
        lap("score")
        # spherical k-means on the DIFFERENCE vectors (what kind of change), 12
        # Lloyd steps; cells with no difference (an edge year missing) get -1
        clu = np.full(n, -1, np.int64)
        if D is not None:
            okD = np.isfinite(D).all(axis=1)
            Dn = D[okD]
            nrm = np.linalg.norm(Dn, axis=1)
            keep = nrm > 1e-6
            Dn = Dn[keep] / nrm[keep][:, None]
            m = Dn.shape[0]
            if m >= K_CLUSTERS:
                k = K_CLUSTERS
                rng = np.random.default_rng(0)
                C = Dn[rng.integers(m)][None, :]
                for _ in range(1, k):
                    d = np.clip(1 - (Dn @ C.T).max(1), 1e-12, None).astype(np.float64)
                    C = np.vstack([C, Dn[rng.choice(m, p=d / d.sum())]])
                lab = np.zeros(m, np.int64)
                for _ in range(12):
                    new = (Dn @ C.T).argmax(1)
                    if (new == lab).all():
                        break
                    lab = new
                    for kk in range(k):
                        if (lab == kk).any():
                            C[kk] = Dn[lab == kk].mean(0)
                    C /= np.maximum(np.linalg.norm(C, axis=1), 1e-9)[:, None]
                lab = (Dn @ C.T).argmax(1)
                order = np.argsort(-np.bincount(lab, minlength=k))
                lab = np.argsort(order)[lab]
                idx_ok = np.where(okD)[0][keep]
                clu[idx_ok] = lab
            lap("kmeans")

        cells = pa.table({
            "cell": j["cell"],
            "cls": pa.array(cls.astype(np.uint8)),
            "name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in cls]),
            "maj": pa.array(maj.astype(np.uint8)),
            "maj_name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in maj]),
            "p_chg": pa.array(p_chg),
            "p_dist": pa.array(p_dist),
            "purity": j["purity"],
            "disp": pa.array(disp.astype(np.float32)),
            "disp_pre": pa.array(steps["pre"]),
            "disp_in": pa.array(steps["in"]),
            "disp_out": pa.array(steps["out"]),
            "moved": pa.array(moved),
            "agree": pa.array(agree),
            "verdict": pa.array(verdict.astype(np.int8)),
            "verdict_name": pa.array([VERDICTS[int(v)][0] if v >= 0 else "unscored" for v in verdict]),
            "cluster": pa.array(clu.astype(np.int16)),
        } | {
            nm: (j[nm].cast(pa.float32()) if has_s2 else pa.nulls(n, pa.float32()))
            for nm in S2_IDX
        })
        idxv = {nm: cells[nm].to_numpy(zero_copy_only=False).astype(np.float32) for nm in S2_IDX} if has_s2 else {}
        idxv["disp"] = disp.astype(np.float32)
        _stretch = {}

        def stretch(idx):
            if idx not in _stretch:
                v = idxv.get(idx)
                ok = v[~np.isnan(v)] if v is not None else np.zeros(0)
                if len(ok) >= 2:
                    lo, hi = (float(q) for q in np.percentile(ok, [2, 98]))
                    if hi <= lo:
                        hi = lo + 1e-6
                else:
                    lo, hi = 0.0, 1.0
                _stretch[idx] = (lo, hi)
            return _stretch[idx]

        lap("table")
        cov = np.where(np.isnan(agree), 1.0, COV_MIN + (1 - COV_MIN) * np.clip(agree, 0, 1)).astype(np.float32)
        cov_flat = np.full(n, COV_FLAT, np.float32)
        cov_inv = np.where(np.isnan(agree), COV_MIN, COV_MIN + (1 - COV_MIN) * (1 - np.clip(agree, 0, 1))).astype(np.float32)
        cellid = cells["cell"].to_numpy().astype(np.uint64)
        lap("hex")
        rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
        rgb_clu = np.where((clu >= 0)[:, None], _PAL[np.clip(clu, 0, len(_PAL) - 1) % len(_PAL)], _GREY).astype(np.uint8)
        rgb_ver = np.where((verdict >= 0)[:, None], _VPAL[np.clip(verdict, 0, 3)], _GREY).astype(np.uint8)
        _ai = np.where(np.isnan(agree), 0, np.clip(agree, 0, 1) * 255).round().astype(np.int64)
        _unscored = np.isnan(agree)[:, None]
        rgb_ramp = np.where(_unscored, 128, _RAMP[_ai]).astype(np.uint8)
        rgb_ramp_inv = np.where(_unscored, 128, _RAMP[255 - _ai]).astype(np.uint8)
        quiet = verdict == 0  # neither: Stable, and the embedding did not move; not drawn

        def fill(paint, sel, hit=None, inv=False, ramp=False, acov=False, idx="ndvi", hide=False, thr=0.5):
            """(N, 4) uint8 rgba for a paint: the widget's getFillColor attribute.
            Paints: `nlcd` (the LCMS class, internal key kept from the chassis),
            `verdict` (the 2x2), `disp` (the displacement on the ramp, stretched
            to the view), `s2` (an index on the ramp), `clusters` (the difference
            vectors' k-means). `ramp` swaps the class or verdict colors for the
            agreement ramp; `inv` reverses it; `hide` drops the cells at or above
            `thr` (unscored cells stay)."""
            _cue = paint in ACOV_PAINTS
            if paint == "clusters":
                c, key = rgb_clu, np.where(clu >= 0, 100 + clu, -1)
            elif paint == "verdict":
                key = np.where(verdict >= 0, 200 + verdict, -1)
                c = (rgb_ramp_inv if inv else rgb_ramp) if ramp else rgb_ver
            elif paint in ("s2", "disp"):
                v = idxv.get("disp" if paint == "disp" else idx)
                if v is None:
                    v = np.full(len(cls), np.nan, np.float32)
                lo, hi = stretch("disp" if paint == "disp" else idx)
                _ok = ~np.isnan(v)
                _t = np.clip((np.where(_ok, v, lo) - lo) / (hi - lo), 0, 1)
                c, key = np.where(_ok[:, None], _RAMP[(_t * 255).round().astype(np.int64)], 128).astype(np.uint8), cls
            elif _cue and ramp:
                c, key = (rgb_ramp_inv if inv else rgb_ramp), cls
            else:
                c, key = rgb, cls
            if (_cue and ramp) or paint in ("s2", "disp"):
                a = np.full(len(cls), ALPHA_RAMP, np.uint8)
            else:
                a = np.full(len(cls), ALPHA_FLAT, np.uint8)
            if sel:
                a = np.where(np.isin(key, list(sel)), a, DIM_ALPHA).astype(np.uint8)
            if hide:
                a = np.where(agree >= thr, 0, a).astype(np.uint8)  # NaN >= thr is False
            # A HEXAGON DRAWS ONLY WHERE THERE IS SOMETHING TO SAY (Stephen,
            # 2026-08-31: "the h3 cells should only show if there's a disagreement
            # ... or there's agreement. It shouldn't show where there's a
            # nonissue"): the cells where LCMS says Stable AND the embedding did
            # not move (verdict 0, "neither") are not drawn on any hexagon paint.
            # They stay in the frame (they set D0, they count, they still pick).
            # Unscored cells (no embedding) are drawn: nothing is known there.
            a = np.where(quiet, 0, a).astype(np.uint8)
            rgba = np.ascontiguousarray(np.concatenate([c, a[:, None]], axis=1)).astype(np.uint8)
            if hit is not None:
                rgba[cellid == hit] = (255, 255, 255, 255)
            return rgba

        def coverage(paint, inv=False, acov=False):
            if acov and paint in ACOV_PAINTS:
                return cov_inv if inv else cov
            return cov_flat

        lap("colors")
        a_ok = agree[~np.isnan(agree)]
        n_says, n_moved, n_quiet = int(says.sum()), int(moved.sum()), int((verdict == 0).sum())
        score = (
            f"{n:,} cells, {n - n_quiet:,} drawn (the rest neither) · {100 * n_says / max(1, n):.0f}% say change (LCMS) · "
            f"{100 * n_moved / max(1, int(scored.sum())):.0f}% moved (AEF, D0 {D0:.3f}) · "
            f"agreement p50 {np.median(a_ok):.2f} · {(a_ok < 0.5).mean() * 100:.0f}% below 0.5"
            if len(a_ok) else f"{n:,} cells · LCMS only ({n_says:,} say change)"
        ) + " (" + " ".join(f"{k} {v:.1f}" for k, v in _lap.items()) + ")"
        return {
            "cells": cells, "cellid": cellid, "fill": fill, "coverage": coverage,
            "cls": cls, "clu": clu, "agree": agree, "verdict": verdict, "disp": disp,
            "says": says, "moved": moved, "D0": D0, "tau": tau,
            "has_aef": has_aef, "has_s2": has_s2, "score": score, "stretch": stretch,
            "years": years,
        }

    def label_components(a, b, n):
        """Connected components over undirected edges (a, b) among n nodes, in
        numpy: min-label hooking + pointer jumping until stable."""
        lab = np.arange(n)
        while True:
            m = lab.copy()
            np.minimum.at(m, a, lab[b])
            np.minimum.at(m, b, lab[a])
            m = m[m]
            while True:
                mm = m[m]
                if np.array_equal(mm, m):
                    break
                m = mm
            if np.array_equal(m, lab):
                return lab
            lab = m

    def edges_for(frame, thr, min_cells, alpha, ramp=False, inv=False, fill=False):
        """Boundaries of the clusters of cells with agreement < thr THAT SHARE A
        CLASS (the LCMS change class, or Stable), dissolved per patch by DuckDB's
        h3 extension; `fill` adds the patches themselves, each painted by the
        average of its cells. The WorldCover build's function unchanged but for
        the class table. Memoised on the frame per (thr, min_cells, alpha, ramp,
        inv, fill)."""
        key = (round(float(thr), 3), int(min_cells), int(alpha),
               bool(ramp), bool(inv), bool(fill))
        cache = frame.setdefault("edges", {})
        if key in cache:
            return cache[key]
        t0 = time.time()
        agree = frame["agree"]
        n_low = int(np.sum(agree < thr))
        out = {"ipc": b"", "fipc": b"", "n_low": n_low, "blobs": 0, "max_km2": 0.0,
               "rings": 0, "polys": 0, "ms": 0}
        if n_low:
            con.register("edge_cells", frame["cells"])
            low = con.sql(
                "SELECT cell, cls, agree, row_number() OVER (ORDER BY cell) - 1 AS i "
                "FROM edge_cells WHERE agree < $thr",
                params={"thr": float(thr)},
            ).arrow().read_all()
            con.register("edge_low", low)
            e = con.sql("""
                WITH nb AS (SELECT i, cls, UNNEST(h3_grid_disk(cell, 1)) AS ncell FROM edge_low)
                SELECT nb.i AS a, l2.i AS b FROM nb JOIN edge_low l2
                  ON nb.ncell = l2.cell AND nb.cls = l2.cls
                WHERE nb.i < l2.i
            """).arrow().read_all()
            lab = label_components(e["a"].to_numpy(), e["b"].to_numpy(), low.num_rows)
            con.register("edge_blob", pa.table({"i": np.arange(low.num_rows), "blob": lab}))
            con.sql("""
                CREATE OR REPLACE TEMP TABLE edge_poly AS
                WITH g AS (
                  SELECT b.blob, any_value(l.cls) AS cls, count(*) AS ncell,
                         avg(l.agree) AS agree,
                         sum(h3_cell_area(l.cell, 'km^2')) AS km2,
                         ST_GeomFromWKB(h3_cells_to_multi_polygon_wkb(list(l.cell))) AS geom
                  FROM edge_low l JOIN edge_blob b USING (i)
                  GROUP BY b.blob HAVING count(*) >= $min_cells),
                s AS (SELECT count(*) AS blobs, max(km2) AS max_km2 FROM g)
                SELECT g.blob, g.cls, g.ncell, g.agree, g.km2, s.blobs, s.max_km2,
                       UNNEST(ST_Dump(g.geom)).geom AS poly
                FROM g, s
            """, params={"min_cells": int(min_cells)})
            r = con.sql("""
                WITH q AS (
                  SELECT blob, cls, ncell, km2, blobs, max_km2,
                         UNNEST(ST_Dump(ST_Boundary(poly))).geom AS ring
                  FROM edge_poly)
                SELECT blob, cls, ncell, km2, blobs, max_km2, ST_AsWKB(ring) AS wkb
                FROM q ORDER BY ncell DESC
            """).arrow().read_all()
            if r.num_rows:
                geom = ga_from_wkb(
                    r["wkb"].combine_chunks().cast(pa.binary()),
                    to_type=ga_linestring("xy", coord_type="interleaved", crs="EPSG:4326"),
                )
                cls = r["cls"].to_numpy().astype(np.int64)
                rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
                rgba = np.concatenate([rgb, np.full((len(cls), 1), alpha, np.uint8)], axis=1).ravel()
                color = pa.FixedSizeListArray.from_arrays(pa.array(rgba, pa.uint8()), 4)
                tbl = pa.table(ArroTable.from_arrays(
                    [ArroArray.from_arrow(geom), ArroArray.from_arrow(color),
                     ArroArray.from_arrow(r["cls"].combine_chunks()), ArroArray.from_arrow(r["km2"].combine_chunks())],
                    names=["geometry", "color", "cls", "km2"],
                )).combine_chunks()
                sink = io.BytesIO()
                with pa_ipc.new_stream(sink, tbl.schema) as w:
                    w.write_table(tbl)
                out.update(ipc=sink.getvalue(), rings=int(r.num_rows),
                           blobs=int(r["blobs"][0].as_py()), max_km2=float(r["max_km2"][0].as_py()))
            if fill:
                f = con.sql(
                    "SELECT cls, agree, ncell, km2, ST_AsWKB(poly) AS wkb "
                    "FROM edge_poly ORDER BY ncell DESC"
                ).arrow().read_all()
                if f.num_rows:
                    fgeom = ga_from_wkb(
                        f["wkb"].combine_chunks().cast(pa.binary()),
                        to_type=ga_polygon("xy", coord_type="interleaved", crs="EPSG:4326"),
                    )
                    fcls = f["cls"].to_numpy().astype(np.int64)
                    fa = np.clip(np.nan_to_num(
                        np.asarray(f["agree"].to_pylist(), np.float64), nan=0.0), 0, 1)
                    if inv:
                        fa = 1 - fa
                    frgb = (
                        _RAMP[np.round(fa * 255).astype(np.int64)]
                        if ramp else
                        np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in fcls], np.uint8)
                    ).astype(np.uint8)
                    falpha = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * fa).round().astype(np.uint8)
                    frgba = np.concatenate([frgb, falpha[:, None]], axis=1).ravel()
                    fcolor = pa.FixedSizeListArray.from_arrays(pa.array(frgba, pa.uint8()), 4)
                    ftbl = pa.table(ArroTable.from_arrays(
                        [ArroArray.from_arrow(fgeom), ArroArray.from_arrow(fcolor),
                         ArroArray.from_arrow(f["cls"].combine_chunks()),
                         ArroArray.from_arrow(f["km2"].combine_chunks())],
                        names=["geometry", "color", "cls", "km2"],
                    )).combine_chunks()
                    fsink = io.BytesIO()
                    with pa_ipc.new_stream(fsink, ftbl.schema) as w:
                        w.write_table(ftbl)
                    out.update(fipc=fsink.getvalue(), polys=int(f.num_rows), fpoly=f)
        out["ms"] = int(1000 * (time.time() - t0))
        cache[key] = out
        return out

    def legend_for(frame, paint, ramp=False, inv=False, idx="ndvi"):
        cls, clu, agree, verdict = frame["cls"], frame["clu"], frame["agree"], frame["verdict"]
        tot = max(1, len(cls))
        items = []
        if paint == "s2":
            lo, hi = frame["stretch"](idx)
            items.append({
                "ramp": RAMP_HEX, "cmap": AGREE_CMAP,
                "lo": f"{idx.upper()} {lo:.2f}", "hi": f"{hi:.2f}",
                "title": f"{AGREE_CMAP}: mean {idx.upper()} per cell, stretched to this view's p2-p98"
                         + ("" if frame["has_s2"] else " (no Sentinel-2 in this view)"),
            })
        if paint == "disp":
            lo, hi = frame["stretch"]("disp")
            d0 = frame["D0"]
            items.append({
                "ramp": RAMP_HEX, "cmap": AGREE_CMAP,
                "lo": f"shift {lo:.3f}", "hi": f"{hi:.3f}",
                "title": f"{AGREE_CMAP}: 1 - cos between the cell's AlphaEarth vectors in consecutive years (the largest step in the window around {YEAR}), stretched to this view's p2-p98"
                         + (f"; D0 (the stable baseline's {100 * (1 - FA):.0f}th percentile) is {d0:.3f}" if not np.isnan(d0) else ""),
            })
        if paint in ACOV_PAINTS and ramp and frame["has_aef"]:
            items.append({
                "ramp": RAMP_HEX, "cmap": AGREE_CMAP,
                "lo": "agreement" if inv else "disagreement",
                "hi": "disagreement" if inv else "agreement",
            })
        if paint == "clusters" and frame["has_aef"]:
            for k in range(int(clu.max()) + 1 if len(clu) and clu.max() >= 0 else 0):
                m = clu == k
                if not m.any():
                    continue
                cc, cn = np.unique(cls[m], return_counts=True)
                top = sorted(zip(cn, cc), reverse=True)[:3]
                mix = ", ".join(f"{100 * nn / m.sum():.0f}% {CLASSES.get(int(c), ('?',))[0]}" for nn, c in top)
                a = agree[m]
                a = a[~np.isnan(a)]
                d = frame["disp"][m]
                d = d[~np.isnan(d)]
                items.append({
                    "code": 100 + k, "name": f"Δ cluster {k}", "hex": CLUSTER_HEX[k % len(CLUSTER_HEX)],
                    "pct": round(100 * int(m.sum()) / tot, 1),
                    "p50": f"{np.median(a):.2f}" if len(a) else "none",
                    "note": mix + (f" · shift p50 {np.median(d):.3f}" if len(d) else ""),
                })
        elif paint == "verdict":
            for v in (1, 2, 3, 0):
                m = verdict == v
                if not m.any():
                    continue
                a = agree[m]
                a = a[~np.isnan(a)]
                items.append({
                    "code": 200 + v, "name": VERDICTS[v][0], "hex": "#%02x%02x%02x" % VERDICTS[v][1],
                    "pct": round(100 * int(m.sum()) / tot, 1),
                    "p50": f"{np.median(a):.2f}" if len(a) else "none",
                    "note": "",
                })
            for it in items:
                if it["code"] == 200:
                    it["note"] = "(not drawn)"
            if (verdict < 0).any():
                items.append({"code": -1, "name": "unscored", "hex": "#808080",
                              "pct": round(100 * int((verdict < 0).sum()) / tot, 1), "p50": "none", "note": "(no embedding)"})
        else:
            codes, nn = np.unique(cls, return_counts=True)
            for code, cnt in sorted(zip(codes, nn), key=lambda t: -t[1]):
                if int(code) not in CLASSES:
                    continue
                a = agree[cls == code]
                a = a[~np.isnan(a)]
                mv = frame["moved"][cls == code]
                sc = ~np.isnan(frame["disp"][cls == code])
                q = frame["verdict"][cls == code] == 0
                items.append({
                    "code": int(code), "name": CLASSES[int(code)][0],
                    "hex": "#%02x%02x%02x" % CLASSES[int(code)][1],
                    "pct": round(100 * int(cnt) / tot, 1),
                    "p50": f"{np.median(a):.2f}" if len(a) else "none",
                    "note": (f"{100 * mv.sum() / max(1, sc.sum()):.0f}% moved" if sc.any() else "(unscored)")
                            + (f" · {100 * q.sum() / cnt:.0f}% not drawn (neither)" if q.any() else ""),
                })
        return items

    return S2_IDX, build_frame, con, edges_for, legend_for


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """The strip under the map (the WorldCover build's strip by copy; two paints added,
        trimmed): paint buttons, pickable legend, panel, status; the one element
        docks into the map's fullscreen. Clicks are the map widget's own (deck
        picking), not captured here."""

        ctl = traitlets.Unicode("").tag(sync=True)
        dres = traitlets.Unicode("0").tag(sync=True)  # kernel -> browser: the offset in force
        thr0 = traitlets.Unicode("0.5").tag(sync=True)  # kernel -> browser: the threshold slider's seed
        runder0 = traitlets.Unicode("0").tag(sync=True)  # kernel -> browser: raster-under seed
        ropac0 = traitlets.Unicode("0.6").tag(sync=True)  # kernel -> browser: its opacity seed
        hopac0 = traitlets.Unicode("1").tag(sync=True)  # kernel -> browser: the hexagons' opacity seed
        status = traitlets.Unicode("").tag(sync=True)
        usrc0 = traitlets.Unicode("nlcd").tag(sync=True)  # seed: the underlay raster
        s2scale0 = traitlets.Unicode("1").tag(sync=True)  # seed: the S2 mosaic's scale (gain)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;" +
            "font:13px ui-sans-serif,system-ui,sans-serif;padding:.35rem 0 0;" +
            "user-select:none;width:100%";
          const btnCss =
            "font:13px ui-sans-serif,system-ui,sans-serif;cursor:pointer;white-space:nowrap;" +
            "padding:.2rem .6rem;border-radius:5px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          // The paint buttons are VISIBILITY (Stephen): one layer on the map at a time;
          // click another and the map goes to that layer; click the one that is on
          // and it disappears. No stacking. Hiding keeps the fold: the kernel flips
          // the widget's visibility and the frame stays, so coming back is instant.
          let paint = "nlcd";
          const sel = new Set();
          let seq = 0;
          let edgesOn = false;
          let fillOn = false;  // the boundaries FILLED, each patch averaged from its cells
          let hideOn = false;  // the boundaries' inverse: cells at/above the threshold not drawn
          // the NLCD raster under the hexagons, and its opacity (declared here, not
          // beside their widgets, so `send` never reads them in their dead zone)
          let runder = model.get("runder0") === "1";
          let ropac = parseFloat(model.get("ropac0") || "0.6");
          let hopac = parseFloat(model.get("hopac0") || "1");
          let open = true;  // the strip's own collapse (client side; no ctl, no run)
          let paintRop = () => {};  // assigned with its widgets below; stylePaint calls it
          let idx = "ndvi";  // the Sentinel-2 H3 paint's index
          let usrc = model.get("usrc0") || "nlcd";  // which raster the underlay is
          let s2scale = parseFloat(model.get("s2scale0") || "1");  // the S2 mosaic's gain
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, paint: paint, sel: Array.from(sel), inv: inv.checked, acol: acol,
              edges: edgesOn, bfill: fillOn, hide: hideOn, thr: parseFloat(thr.value),
              runder: runder, ropac: ropac, hopac: hopac, acov: acov,
              idx: idx, usrc: usrc, s2scale: s2scale,
              n: ++seq }, extra || {})));
            model.save_changes();
          };
          const onCss = (b, on) => {
            b.style.borderColor = on ? "#2b6cb0" : "rgba(127,127,127,.45)";
            b.style.fontWeight = on ? "600" : "400";
          };
          const paintBox = document.createElement("span");
          paintBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const pl = document.createElement("span");
          pl.textContent = "layer";
          const mkPaint = (key, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => { paint = paint === key ? null : key; sel.clear(); stylePaint(); send("set"); renderLegend(); };
            return [key, b];
          };
          const paintBtns = [
            mkPaint("raster", "LCMS raster", "the year's USFS LCMS Change layer (30 m, 16 change classes) as tiles the kernel renders from the COG; click again to hide"),
            mkPaint("s2raster", "S2 raster", "Earth Genome's Sentinel-2 yearly mosaic (true color) as tiles the kernel renders; nothing below zoom 9; click again to hide"),
            mkPaint("nlcd", "LCMS H3", "each hexagon in the color of its main change class (Stable when under 5% of its pixels changed); the two agreement toggles color or size the cells by whether AlphaEarth moved where LCMS says it should have; click again to hide"),
            mkPaint("verdict", "verdict H3", "the 2x2 per hexagon: both say change, LCMS only, AEF only, neither; click again to hide"),
            mkPaint("disp", "AEF shift H3", "the AlphaEarth displacement itself (1 - cos between consecutive years, the largest step in the window) on a ramp stretched to the view; click again to hide"),
            mkPaint("s2", "S2 H3", "the mosaic folded to the hexagons as a spectral index (pick which); NBR is the one that speaks to fire; click again to hide"),
            mkPaint("clusters", "AEF Δ clusters H3", "the year-to-year DIFFERENCE vectors clustered (k-means): a cluster is a kind of change, not a kind of ground; click again to hide"),
          ];
          // the index the Sentinel-2 H3 paint draws (live only on that paint)
          const idxSel = document.createElement("select");
          [["ndvi", "NDVI"], ["ndwi", "NDWI"], ["ndbi", "NDBI"], ["nbr", "NBR"], ["mndwi", "MNDWI"]].forEach(([v, t]) => {
            const o = document.createElement("option"); o.value = v; o.textContent = t; idxSel.appendChild(o);
          });
          idxSel.value = idx;
          idxSel.title = "Sentinel-2 H3: which index colors the hexagons (NDVI vegetation, NDWI water, NDBI built-up, NBR burn / bare, MNDWI water vs built)";
          idxSel.style.cssText = "font:12px ui-sans-serif,system-ui,sans-serif;padding:.05rem .2rem;border-radius:4px;border:1px solid rgba(127,127,127,.45);background:transparent;color:inherit";
          idxSel.addEventListener("change", () => { idx = idxSel.value; send("set"); });
          const styleIdx = () => { idxSel.style.display = paint === "s2" ? "" : "none"; };
          const invLab = document.createElement("label");
          invLab.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const inv = document.createElement("input");
          inv.type = "checkbox"; inv.checked = false;
          invLab.appendChild(inv); invLab.appendChild(document.createTextNode("highlight disagreement"));
          invLab.title = "the agreement toggles reversed: the least-backed cells solid and full size, the agreeing ones faint and small";
          inv.addEventListener("change", () => send("set"));
          // color by agreement: the agreement paint's hexagons on a cool-to-warm
          // ramp (cool = disagreement) instead of NLCD's colors; the highlight
          // checkbox reverses the ramp. Coverage still follows agreement.
          let acol = false;
          const acB = document.createElement("button");
          acB.textContent = "color by agreement"; acB.style.cssText = btnCss;
          acB.title = "LCMS H3 / verdict H3: color the hexagons by agreement (cool = disagreement, warm = agreement) instead of the classes; highlight disagreement reverses the ramp";
          const styleAc = () => { onCss(acB, acol); acB.style.display = (paint === "nlcd" || paint === "verdict") ? "" : "none"; };
          acB.onclick = () => { acol = !acol; styleAc(); send("set"); };
          // agreement coverage: NLCD H3's cells SIZED (and faded) by how well
          // AlphaEarth backs their class. With "color by agreement" beside it these
          // are the two agreement cues, and the agreement paint they came from is
          // gone (Stephen, 2026-08-27). NLCD H3 today; AlphaEarth clusters may join
          // them later, which is this gate and the kernel's ACOV_PAINTS.
          let acov = false;
          const cvB = document.createElement("button");
          cvB.textContent = "agreement coverage"; cvB.style.cssText = btnCss;
          cvB.title = "scale every hexagon by how well AlphaEarth backs LCMS's word there (moved where it says change, still where it says Stable); highlight disagreement reverses it";
          const styleCv = () => {
            onCss(cvB, acov);
            const on = paint === "nlcd" || paint === "s2" || paint === "verdict";
            cvB.style.display = on ? "" : "none";
            invLab.style.display = on ? "inline-flex" : "none";   // it reverses these cues
          };
          cvB.onclick = () => { acov = !acov; styleCv(); send("set"); };
          let styleEdRef = () => {};  // assigned with the boundaries widgets below
          const stylePaint = () => {
            paintBtns.forEach(([k, b]) => onCss(b, k === paint));
            styleAc(); styleCv(); styleIdx(); paintRop(); styleEdRef();
          };
          stylePaint();
          const btnsWithIdx = [];
          paintBtns.forEach(([k, b]) => { btnsWithIdx.push(b); if (k === "s2") btnsWithIdx.push(idxSel); });
          paintBox.append(pl, ...btnsWithIdx, invLab, acB, cvB);
          // NLCD raster UNDER the hexagons, with its opacity: normally the raster
          // hides wherever the hexagons draw, so a hexagon paint says nothing about
          // the pixels it came from; ticked, the tiles stay under them (Stephen).
          // The slider commits on change, never input.
          const underBox = document.createElement("span");
          underBox.style.cssText = "display:inline-flex;gap:.35rem;align-items:center";
          const undLab = document.createElement("label");
          undLab.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const und = document.createElement("input");
          und.type = "checkbox"; und.checked = runder;
          undLab.appendChild(und); undLab.appendChild(document.createTextNode("raster under"));
          undLab.title = "keep a raster's tiles under the hexagons instead of hiding them where they draw (and show it wide, below the hexagon zoom)";
          // which raster: NLCD or the Sentinel-2 mosaic
          const usrcSel = document.createElement("select");
          [["nlcd", "LCMS"], ["s2", "S2"]].forEach(([v, t]) => {
            const o = document.createElement("option"); o.value = v; o.textContent = t; usrcSel.appendChild(o);
          });
          usrcSel.value = usrc;
          usrcSel.title = "the raster kept under the hexagons";
          usrcSel.style.cssText = "font:12px ui-sans-serif,system-ui,sans-serif;padding:.05rem .2rem;border-radius:4px;border:1px solid rgba(127,127,127,.45);background:transparent;color:inherit";
          usrcSel.addEventListener("change", () => { usrc = usrcSel.value; paintRop(); send("vis"); });
          const opLab = document.createElement("span"); opLab.textContent = "opacity";
          const rop = document.createElement("input");
          rop.type = "range"; rop.min = "0"; rop.max = "1"; rop.step = "0.05";
          rop.value = String(ropac);
          rop.style.cssText = "width:6rem;vertical-align:middle";
          rop.title = "the raster's opacity under the hexagons";
          const ropV = document.createElement("span");
          ropV.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4rem";
          // the hexagons' own opacity: a deck layer `opacity` on whichever hexagon
          // layer is drawing, multiplied onto the alpha the paint already carries
          const hexLab = document.createElement("span");
          hexLab.textContent = "hex opacity";
          hexLab.style.cssText = "margin-left:.5rem";
          const hop = document.createElement("input");
          hop.type = "range"; hop.min = "0.05"; hop.max = "1"; hop.step = "0.05";
          hop.value = String(hopac);
          hop.style.cssText = "width:6rem;vertical-align:middle";
          hop.title = "the hexagons' opacity, over the raster or the basemap";
          const hopV = document.createElement("span");
          hopV.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4rem";
          // `scale`: the Sentinel-2 mosaic's brightness (a gain on the TCI bytes,
          // applied in the kernel where the tiles are composited, so a commit drops
          // the S2 tiles and asks deck for them again: NOT a free config flip, so
          // it commits on change only). It ONLY APPEARS while S2 is what is being
          // drawn: the `S2 raster` paint, or `raster under` ticked with S2 picked
          // (Stephen: "a slider for brightness call it scale for the s2 mosaic
          // that only appears when it's clicked and the same for in underlay").
          const scBox = document.createElement("span");
          scBox.style.cssText = "display:none;gap:.35rem;align-items:center";
          scBox.dataset.aefS2scale = "1";
          const scLab = document.createElement("span"); scLab.textContent = "scale";
          const sc = document.createElement("input");
          sc.type = "range"; sc.min = "0.2"; sc.max = "3"; sc.step = "0.1";
          sc.value = String(s2scale);
          sc.style.cssText = "width:6rem;vertical-align:middle";
          sc.title = "brightness of the Sentinel-2 mosaic (a gain on the TCI bytes; the tiles are re-served)";
          const scV = document.createElement("span");
          scV.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4rem";
          const paintSc = () => { scV.textContent = parseFloat(sc.value).toFixed(1) + "×"; };
          paintSc();
          sc.addEventListener("input", paintSc);
          sc.addEventListener("change", () => { s2scale = parseFloat(sc.value); send("s2scale"); });
          scBox.append(scLab, sc, scV);
          // CONDITIONAL CONTROLS APPEAR ONLY WHEN THEY APPLY (Stephen: "the sliders
          // and buttons that are conditional should pop up only when they are
          // needed ... cleaner and saves space"): nothing is dimmed any more, it
          // is there or it is not. Under a hexagon paint: raster under (+ its
          // source and `opacity` once ticked) and hex opacity; `scale` whenever
          // the S2 mosaic is what is drawn (its paint, or the S2 underlay); the
          // whole row hides when nothing in it applies.
          const show = (el, on, disp) => { el.style.display = on ? (disp || "") : "none"; };
          paintRop = () => {
            ropV.textContent = parseFloat(rop.value).toFixed(2);
            hopV.textContent = parseFloat(hop.value).toFixed(2);
            const hexPaint = !!paint && paint !== "raster" && paint !== "s2raster";
            const s2Drawn = paint === "s2raster" || (hexPaint && und.checked && usrc === "s2");
            show(undLab, hexPaint, "inline-flex"); show(usrcSel, hexPaint);
            [opLab, rop, ropV].forEach((e) => show(e, hexPaint && und.checked));
            show(scBox, s2Drawn, "inline-flex");
            [hexLab, hop, hopV].forEach((e) => show(e, hexPaint));
            show(underBox, hexPaint || s2Drawn, "inline-flex");
          };
          paintRop();
          und.addEventListener("change", () => { runder = und.checked; paintRop(); send("vis"); });
          rop.addEventListener("input", paintRop);
          rop.addEventListener("change", () => { ropac = parseFloat(rop.value); if (und.checked) send("vis"); });
          hop.addEventListener("input", paintRop);
          hop.addEventListener("change", () => { hopac = parseFloat(hop.value); send("vis"); });
          // order: raster under · source · opacity · scale · hex opacity (scale
          // beside the opacity it dims with, Stephen: "after hex opacity is confusing")
          underBox.append(undLab, usrcSel, opLab, rop, ropV, scBox, hexLab, hop, hopV);
          // boundaries around the patches of low-agreement cells OF ONE NLCD CLASS
          // (grouped and dissolved in the kernel by DuckDB's h3 extension), with
          // the agreement threshold that defines "low": a slider that commits on
          // change (never input: every commit is a dissolve and a send)
          const edgeBox = document.createElement("span");
          edgeBox.style.cssText = "display:inline-flex;gap:.35rem;align-items:center";
          const edB = document.createElement("button");
          edB.textContent = "boundaries"; edB.style.cssText = btnCss;
          edB.title = "outline every patch of touching cells that are below the threshold AND share a change class (or Stable), in that class's color; click again to hide";
          const thr = document.createElement("input");
          thr.type = "range"; thr.min = "0.05"; thr.max = "0.95"; thr.step = "0.05";
          thr.value = String(model.get("thr0") || "0.5");
          thr.style.cssText = "width:7rem;vertical-align:middle";
          thr.title = "agreement below this is inside a boundary; with hide agreeing on, cells at or above it are not drawn";
          const thrV = document.createElement("span");
          thrV.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.6rem";
          const paintThr = () => { thrV.textContent = "< " + parseFloat(thr.value).toFixed(2); };
          paintThr();
          thr.addEventListener("input", paintThr);
          thr.addEventListener("change", () => { if (edgesOn || hideOn) send("set"); });
          // fill: the same patches painted in, each one the AVERAGE of its own cells
          // (color and opacity), under the rings. Only with the boundaries on.
          const flB = document.createElement("button");
          flB.textContent = "fill"; flB.style.cssText = btnCss;
          flB.title = "fill each boundary with the average of the cells inside it: its class color (or the agreement ramp) at the alpha that mean agreement earns";
          // hide agreeing: the boundaries' INVERSE (Stephen, 2026-08-28). The cells
          // at or above the same threshold are not drawn (alpha 0 in the kernel's
          // colors), so the doubtful cells stand alone over the raster or the
          // basemap; the hidden cells still pick (the click is h3-js on the
          // coordinate, not the pixel). Shares the threshold slider; on its own
          // (boundaries off) the slider is still shown for it.
          const hdB = document.createElement("button");
          hdB.textContent = "hide agreeing"; hdB.style.cssText = btnCss;
          hdB.title = "do not draw the cells at or above the threshold, so the raster shows through where AlphaEarth and LCMS agree; they still pick. Click again to draw them";
          const styleEd = () => {
            onCss(edB, edgesOn);
            onCss(flB, fillOn);
            onCss(hdB, hideOn);
            // fill only once the boundaries are on; the threshold when either
            // consumer is on; hide only where the hexagons draw
            const hexPaint = !!paint && paint !== "raster" && paint !== "s2raster";
            flB.style.display = edgesOn ? "" : "none";
            hdB.style.display = hexPaint ? "" : "none";
            [thr, thrV].forEach((e) => { e.style.display = (edgesOn || (hideOn && hexPaint)) ? "" : "none"; });
            // the boundaries are their own layer with their own toggle: offered
            // under every paint (and with none), not only where the hexagons draw;
            // res applies wherever a fold runs, which is a hexagon paint OR the
            // boundaries on
            edgeBox.style.display = "inline-flex";
            resBox.style.display = (hexPaint || edgesOn) ? "inline-flex" : "none";
          };
          // (styleEd runs first once resBox exists below: it hides both boxes)
          edB.onclick = () => { edgesOn = !edgesOn; styleEd(); send("set"); };
          flB.onclick = () => { fillOn = !fillOn; styleEd(); send("set"); };
          hdB.onclick = () => { hideOn = !hideOn; styleEd(); send("set"); };
          edgeBox.append(edB, flB, hdB, thr, thrV);
          // res: the offset from the ladder (-2..+2). + refolds the CURRENT view one
          // step finer (zooming in never does on its own); the offset resets when
          // the camera leaves the served box, and the kernel echoes it back.
          const resBox = document.createElement("span");
          resBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const rl = document.createElement("span"); rl.textContent = "res";
          const rv = document.createElement("span");
          rv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums;min-width:1.6rem;text-align:center";
          const mkRes = (d, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => {
              const cur = parseInt(model.get("dres") || "0", 10);
              const nxt = Math.max(-2, Math.min(2, cur + d));
              if (nxt !== cur) send("dres", { dres: nxt });
            };
            return b;
          };
          const rMinus = mkRes(-1, "−", "refold this view one step coarser");
          const rPlus = mkRes(+1, "+", "refold this view one step finer (7x the cells, and the read)");
          const paintR = () => {
            const v = parseInt(model.get("dres") || "0", 10);
            rv.textContent = (v > 0 ? "+" : "") + v;
          };
          model.on("change:dres", paintR);
          paintR();
          resBox.append(rl, rMinus, rv, rPlus);
          styleEdRef = styleEd;
          styleEd();
          const legendBox = document.createElement("div");
          legendBox.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;" +
            "gap:.15rem .7rem;flex:1 1 100%;min-width:14rem;font-size:13px";
          const renderLegend = () => {
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); }
            catch (e) { items = []; }
            legendBox.innerHTML = "";
            if (sel.size) {
              const x = document.createElement("button");
              x.textContent = "× all";
              x.style.cssText =
                "font:11px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;border:1px solid " +
                "#2b6cb0;background:transparent;color:inherit";
              x.onclick = () => { sel.clear(); send("set"); renderLegend(); };
              legendBox.appendChild(x);
            }
            items.forEach((it) => {
              if (it.ramp) {
                // the agreement ramp bar with its end labels
                const r = document.createElement("span");
                r.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;font:12px ui-sans-serif,system-ui,sans-serif";
                r.title = it.title || (it.cmap + ": color by agreement");
                r.innerHTML =
                  '<span style="opacity:.75">' + it.lo + '</span>' +
                  '<span style="display:inline-block;width:9rem;height:10px;border-radius:2px;' +
                  "background:linear-gradient(to right," + it.ramp.join(",") + ')"></span>' +
                  '<span style="opacity:.75">' + it.hi + '</span>';
                legendBox.appendChild(r);
                return;
              }
              const b = document.createElement("button");
              const on = sel.has(it.code);
              b.style.cssText =
                "display:inline-flex;align-items:center;gap:.3rem;" +
                "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;background:transparent;" +
                "color:inherit;border:1px solid " +
                (on ? "#2b6cb0" : "transparent") + (on ? ";font-weight:600" : "");
              b.title = it.pct + "% of cells · agreement p50 " + it.p50;
              b.innerHTML =
                '<span style="width:10px;height:10px;border-radius:2px;' +
                "background:" + it.hex + ';display:inline-block"></span>' +
                it.name + (it.note ? ' <span style="opacity:.6">' + it.note + "</span>" : "");
              b.onclick = () => {
                if (sel.has(it.code)) sel.delete(it.code); else sel.add(it.code);
                send("set"); renderLegend();
              };
              legendBox.appendChild(b);
            });
          };
          model.on("change:legend", renderLegend);
          renderLegend();
          // analyze what's in view (the crops notebook's button): the kernel fills
          // the panel with the view's summary; × clear empties it
          const anBox = document.createElement("span");
          anBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const anB = document.createElement("button");
          anB.textContent = "analyze what's in view"; anB.style.cssText = btnCss;
          anB.title = "per change class in view: share, area, how many cells AlphaEarth saw move, shift and agreement, NBR split by moved / still; the verdicts; the Δ clusters' make-up";
          anB.onclick = () => send("analyze");
          const clB = document.createElement("button");
          clB.textContent = "× clear"; clB.style.cssText = btnCss;
          clB.onclick = () => send("clear");
          // refresh (cdl-aef-deck's button): drop every cached NLCD tile and COG
          // block, rebuild the raster layer under a new id so deck asks for its
          // tiles again, and refold the view. The way back when the raster (or the
          // fold) is not what it should be.
          const rfB = document.createElement("button");
          rfB.textContent = "refresh"; rfB.style.cssText = btnCss;
          rfB.title = "re-read the raster tiles and refold this view (drops every rendered tile)";
          rfB.onclick = () => send("refresh");
          anBox.append(anB, clB, rfB);
          let labelsOn = true;
          const lbB = document.createElement("button");
          lbB.textContent = "labels"; lbB.style.cssText = btnCss; lbB.title = "place labels over the map, on or off";
          const styleLb = () => { lbB.style.borderColor = labelsOn ? "#2b6cb0" : "rgba(127,127,127,.45)"; lbB.style.fontWeight = labelsOn ? "600" : "400"; };
          styleLb();
          lbB.onclick = () => { labelsOn = !labelsOn; styleLb(); send("labels", { labels: labelsOn }); };
          anBox.append(lbB);
          // the search field: Photon (cdl-ftw's), Enter geocodes camera-biased on the
          // kernel and the first hit flies the map; the fold follows the moveend
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place…";
          search.title = "Photon geocoder: Enter flies to the first hit";
          search.style.cssText =
            "width:11rem;font:13px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) { e.preventDefault(); send("search", { q: q }); }
          });
          anBox.append(search);
          // collapse (top right of the strip) / expand (bottom right of the map,
          // just above the Carto credit), the cdl-ftw-zarr-marimo strip's by copy.
          // Client-side only: no ctl, so no kernel run, no repaint and no re-fold.
          const sqCss =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "width:1.5rem;height:1.5rem;line-height:1;padding:0;border-radius:5px;" +
            "border:1px solid rgba(127,127,127,.45);color:inherit;opacity:.6";
          const colB = document.createElement("button");
          colB.textContent = "\u25be"; colB.title = "hide the controls";
          colB.style.cssText = sqCss + ";margin-left:auto;flex:0 0 auto;background:transparent";
          const expB = document.createElement("button");
          expB.textContent = "\u25b4"; expB.title = "show the controls";
          expB.dataset.aefExpand = "1";
          expB.className = "maplibregl-ctrl";   // the map's own controls, not the canvas
          expB.style.cssText =
            sqCss + ";position:absolute;right:8px;bottom:52px;z-index:6;display:none;" +
            "background:#fff;color:#222;border-color:rgba(0,0,0,.2);opacity:1;" +
            "box-shadow:0 0 0 2px rgba(0,0,0,.1)";
          // the expand arrow belongs to the MAP. The map is another widget and may
          // not be in the DOM yet, so poll briefly for its container; failing that,
          // the page's own bottom right corner.
          const deepFind = (sel) => {
            const walk = (r) => {
              for (const n of r.querySelectorAll("*")) {
                if (n.matches && n.matches(sel)) return n;
                if (n.shadowRoot) { const h = walk(n.shadowRoot); if (h) return h; }
              }
              return null;
            };
            return walk(document);
          };
          let tries = 0;
          const dock = () => {
            const m = deepFind(".maplibregl-map");
            if (m) { m.appendChild(expB); return true; }
            if (++tries > 60) {
              expB.style.position = "fixed";
              expB.style.right = ".9rem";
              expB.style.bottom = ".9rem";
              expB.style.zIndex = "60";
              document.body.appendChild(expB);
              return true;
            }
            return false;
          };
          if (!dock()) { const iv = setInterval(() => { if (dock()) clearInterval(iv); }, 400); }
          const setOpen = (v) => {
            open = v;
            wrap.style.display = v ? "" : "none";   // the strip goes entirely
            expB.style.display = v ? "none" : "block";
          };
          colB.onclick = () => setOpen(false);
          expB.onclick = () => setOpen(true);
          [colB, expB].forEach((b) => {
            b.onmouseenter = () => { b.style.opacity = "1"; };
            b.onmouseleave = () => { b.style.opacity = b === expB ? "1" : ".6"; };
          });
          // row one is the paints, with the collapse button hard right ON THAT LINE
          // (its own row, so a wrap in the controls below never carries it down);
          // the opacities and the rest wrap under it
          const topRow = document.createElement("div");
          topRow.style.cssText =
            "display:flex;flex-wrap:nowrap;align-items:center;gap:.6rem 1rem;flex:1 1 100%";
          topRow.append(paintBox, colB);
          box.append(topRow, underBox, edgeBox, resBox, anBox, legendBox);
          const panel = document.createElement("div");
          panel.style.cssText = "font:13.5px ui-sans-serif,system-ui,sans-serif;padding:.25rem 0";
          const status = document.createElement("div");
          status.style.cssText =
            "font:13px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.2rem 0;min-height:1.2em;white-space:pre-line";
          const wrap = document.createElement("div");
          wrap.style.cssText = "width:100%;box-sizing:border-box";
          wrap.dataset.aefStrip = "1";
          wrap.append(box, panel, status);
          const killOld = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("[data-aef-strip]").forEach((w) => {
              if (w !== wrap) { w.dataset.dead = "1"; w.remove(); }
            });
            root.querySelectorAll("[data-aef-expand]").forEach((b) => {
              if (b !== expB) b.remove();
            });
            root.querySelectorAll("*").forEach((n) => { if (n.shadowRoot) killOld(n.shadowRoot); });
          };
          killOld(document);
          el.appendChild(wrap);
          setOpen(open);
          const realFs = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement)
              fe = fe.shadowRoot.fullscreenElement;
            return fe;
          };
          const onFs = () => {
            if (wrap.dataset.dead || !el.isConnected) {
              wrap.remove();
              document.removeEventListener("fullscreenchange", onFs);
              return;
            }
            const fe = realFs();
            if (fe && fe !== el && !el.contains(fe)) {
              if (getComputedStyle(fe).position === "static") fe.style.position = "relative";
              wrap.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;z-index:30;" +
                "background:rgba(255,255,255,.94);color:#111;box-sizing:border-box;" +
                "padding:.6rem 1.4rem .7rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "width:100%;box-sizing:border-box";
              el.appendChild(wrap);
            }
            setOpen(open);   // the cssText rewrites above drop a collapsed strip's display
          };
          document.addEventListener("fullscreenchange", onFs);
          const paintS = () => { status.textContent = model.get("status") || ""; };
          model.on("change:status", paintS);
          paintS();
          const paintP = () => { panel.innerHTML = model.get("panel") || ""; };
          model.on("change:panel", paintP);
          paintP();
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            wrap.remove();
            expB.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class DeckMap(anywidget.AnyWidget):
        """The map: maplibre (Carto Positron, interleaved) with deck.gl 9.3.10 from
        esm.sh drawing INSIDE it under the label layers, the HRRR counties film's
        chassis. Two deck layers with real ids: `nlcd`, a TileLayer whose tiles the
        kernel renders on request (anywidget custom messages, PNG bytes back), and
        `hexes`, an H3HexagonLayer subclass whose high-precision polygon path
        scales each cell's own ring by that cell's coverage (`scaledRing`), so
        every paint draws from cell ids + rgba + float32 coverage and nothing is
        tessellated in the kernel.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `cov` (f32),
        `config` (JSON: height, home, raster mode, labels, hex_zoom, extent).
        Browser -> kernel: `view` (JSON lon/lat/zoom + canvas w/h on every
        moveend) and `pick` (JSON: the clicked cell as a hex string, or null;
        deck's GPU pick when it answers, else h3-js on the click's lon/lat)."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        cov = traitlets.Bytes(b"").tag(sync=True)
        # low-agreement boundaries: one GeoArrow IPC stream (geoarrow.linestring,
        # interleaved coords; the counties film's transport), drawn by a
        # GeoArrowPathLayer under `config.edges`
        edges = traitlets.Bytes(b"").tag(sync=True)
        # the same boundaries FILLED: geoarrow.polygon (holes kept), one color per
        # blob from the average of its cells, a GeoArrowSolidPolygonLayer under the
        # rings and under `config.bfill`. Never pickable (picking stays on cells).
        fills = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (z, x, y) -> PNG bytes; the wiring sets it
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError as e:
                # no loop in the comm handler: an ERROR, not an empty tile (an
                # empty tile is cached by deck and never asked for again)
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c):
            """One tile for the widget's TileLayer. A FAILURE IS REPORTED AS AN
            ERROR, never as an empty tile: deck caches an empty tile as loaded and
            the area stays blank for the life of the tileset, with nothing said
            anywhere. `empty` is only for a tile that is legitimately outside
            NLCD."""
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            try:
                png = await self.tile_fn(int(c["z"]), int(c["x"]), int(c["y"]), c.get("src", "nlcd"))
            except Exception as e:
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        // every deck import pins the same versions AND the same ?deps= per package
        // (esm.sh hashes a module by its deps list), so the whole graph resolves to
        // ONE @deck.gl/core; apache-arrow rides along for the GeoArrow layers. The
        // strings are the HRRR counties film's (crawled: one core, one luma set).
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {PolygonLayer, BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {GeoArrowPathLayer, GeoArrowSolidPolygonLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/aggregation-layers@9.3.10,@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";
        import {latLngToCell, getResolution, cellToBoundary, cellToLatLng} from "https://esm.sh/h3-js@4.5.0";

        const STYLES = {
          labels: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
          nolabels: "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
        };

        // A cell's OWN boundary (h3-js), every vertex scaled about the cell's
        // centre by that cell's coverage: exact per-cell geometry AND a per-cell
        // size. deck's H3HexagonLayer only offers one of the two: its low-precision
        // path (an instanced column mesh, one coverage attribute per layer, which
        // the kepler.gl-style shader patch used to make per cell) draws every
        // cell with the CENTRE cell's mesh, so cells drift towards the edges of a
        // wide view and the `overfill` that closed the gap was a lattice of
        // doubled edges wherever translucent cells overlapped (Stephen's screenshot,
        // 2026-08-27, uniform across the view and across res: overfill, not drift).
        // This is deck's own high-precision polygon path with the per-cell scale
        // put where deck applies its single `coverage`. Flat rings, [lng, lat].
        function scaledRing(hex, cov) {
          const ring = cellToBoundary(hex, true);   // closed, GeoJSON [lng, lat], CCW
          const c = cellToLatLng(hex);              // [lat, lng]
          const clng = c[1], clat = c[0];
          const out = new Float64Array(ring.length * 2);
          let k = 0;
          for (const v of ring) {
            let dl = v[0] - clng;
            if (dl > 180) dl -= 360; else if (dl < -180) dl += 360;   // across the antimeridian
            out[k++] = clng + dl * cov;
            out[k++] = clat + (v[1] - clat) * cov;
          }
          return out;
        }
        class CoverageH3Layer extends H3HexagonLayer {
          _renderPolygonLayer() {
            const {data, getHexagon, getCoverage, updateTriggers} = this.props;
            const Sub = this.getSubLayerClass("hexagon-cell-hifi", PolygonLayer);
            const fwd = this._getForwardProps();
            fwd.updateTriggers.getPolygon = [updateTriggers.getHexagon, updateTriggers.getCoverage];
            return new Sub(fwd, this.getSubLayerProps({id: "hexagon-cell-hifi", updateTriggers: fwd.updateTriggers}), {
              data, _normalize: false, _windingOrder: "CCW", positionFormat: "XY",
              getPolygon: (o, oi) => scaledRing(getHexagon(o, oi), getCoverage(o, oi)),
            });
          }
        }
        CoverageH3Layer.layerName = "CoverageH3Layer";
        CoverageH3Layer.defaultProps = {...H3HexagonLayer.defaultProps, getCoverage: {type: "accessor", value: 1}};

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        function copyOf(u8) {  // an aligned private copy (DataView slices are not aligned)
          return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
        }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const root = document.createElement("div");
          root.style.cssText = "position:relative;width:100%";
          const mapEl = document.createElement("div");
          mapEl.style.cssText = "width:100%;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
          const note = document.createElement("div");
          note.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;font:11px ui-monospace,Menlo,monospace;" +
            "color:#333;background:rgba(255,255,255,.85);padding:2px 6px;border-radius:3px;pointer-events:none;display:none";
          note.dataset.aefNote = "1";
          root.append(mapEl, note);
          el.append(css, root);
          const say = (t) => { note.textContent = t; note.style.display = t ? "block" : "none"; };

          let hexes = [], N = 0, colors = null, cov = null, seq = 0, map = null, overlay = null;
          let hexIndex = new Map(), res = -1;
          // The bytes are COPIED the moment a trait changes: read later (even 0 ms
          // later, from a timer) the DataView marimo handed over is no longer
          // readable and loadCells silently left N = 0 (measured: the change event
          // saw 46,440 bytes, the deferred read saw nothing, a manual reload worked).
          const raw = {cells: null, colors: null, cov: null, edges: null, fills: null};
          const grab = (k) => {
            try { const u8 = bytesOf(model.get(k)); raw[k] = u8 && u8.length ? copyOf(u8) : null; }
            catch (e) { raw[k] = null; say("grab " + k + ": " + e.message); }
          };
          let dataObj = null;  // one object per (cells, colors, cov) triple: identity is deck's change signal
          let edgeTable = null;  // the boundaries: an arrow Table with a geoarrow.linestring column
          let fillTable = null;  // the same blobs filled: a geoarrow.polygon column
          function loadEdges() {
            const buf = raw.edges;
            if (!buf || !buf.byteLength) { edgeTable = null; return; }
            try { edgeTable = arrow.tableFromIPC(new Uint8Array(buf)); }
            catch (e) { edgeTable = null; say("edges: " + e.message); }
          }
          function loadFills() {
            const buf = raw.fills;
            if (!buf || !buf.byteLength) { fillTable = null; return; }
            try { fillTable = arrow.tableFromIPC(new Uint8Array(buf)); }
            catch (e) { fillTable = null; say("fills: " + e.message); }
          }

          function loadCells() {
            const buf = raw.cells;
            if (!buf || !buf.byteLength) { hexes = []; N = 0; hexIndex = new Map(); res = -1; return; }
            const ids = new BigUint64Array(buf);
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
          }
          function loadAttrs() {
            const c8 = raw.colors, v8 = raw.cov;
            colors = c8 && c8.byteLength === N * 4 ? new Uint8Array(c8) : null;
            cov = v8 && v8.byteLength === N * 4 ? new Float32Array(v8) : null;
            // no overfill any more: both hexagon layers draw each cell's own ring
            // (the coverage one scaled per cell), so cells meet exactly and there is
            // nothing to close. One object per frame: its identity is deck's change signal.
            dataObj = N && colors && cov ? {length: N} : null;
          }

          // NLCD tiles: ask the kernel, get a PNG back on the custom-message channel.
          // A tile that FAILS rejects (deck retries it and `onTileError` says so);
          // only a tile the kernel calls `empty` (outside NLCD) resolves to null.
          // Resolving null on a failure or an abort is what silently blanks an area
          // for good: deck stores it as a loaded tile with no data.
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (msg, buffers) => {
            if (msg && msg.kind === "fly" && map) {
              // the geocoder's hit: maplibre flies, moveend sends the view, the kernel folds
              map.flyTo({center: [msg.lon, msg.lat], zoom: msg.zoom, duration: msg.duration || 2000});
              return;
            }
            if (!msg || msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.err) {
              tstat.err++;
              say((msg.src === "s2" ? "Sentinel-2" : "LCMS") + " tile: " + msg.err);
              p.reject(new Error(msg.err));
              return;
            }
            if (msg.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(
              (b) => { tstat.got++; p.resolve(b); },
              (e) => { tstat.err++; p.reject(e instanceof Error ? e : new Error("decode")); });
          });
          // one fetcher per raster SOURCE ("nlcd", "s2"): the kernel renders
          // whichever the message names
          const getTileDataFor = (src) => ({index, signal}) => new Promise((resolve, reject) => {
            const id = ++tseq;
            tstat.asked++;
            pending.set(id, {resolve, reject});
            model.send({kind: "tile", id, src, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => {
              pending.delete(id);
              tstat.abort++;
              const e = new Error("aborted"); e.name = "AbortError"; reject(e);
            });
          });

          // the raster shows where its switch is on and the hexagons are not drawn
          // (their switch off, or no frame, or below hex_zoom); the hexagons show
          // where their switch is on. Both are `visible` flips: the tiles and the
          // frame stay in the browser, nothing is refetched or refolded.
          // `raster_under` (the strip's checkbox) keeps the tiles on WHERE THE
          // HEXAGONS DRAW as well, at `raster_opacity`; the raster layer is pushed
          // first, so it stays under them.
          const hexesDrawn = () => cfg.show_hexes !== false && !!dataObj && !!map && map.getZoom() >= (cfg.hex_zoom || 9);
          const rasterUnder = () => cfg.raster_under === true;
          const rasterOn = () => cfg.show_raster !== false && (!hexesDrawn() || rasterUnder());
          // WHICH raster draws (the kernel decides: the raster paint that is on,
          // or the underlay pick while a hexagon paint is on)
          // hexagons drawn: the underlay pick. Not drawn (below hex_zoom, or a raster
          // paint): the LAST RASTER CLICKED (`wide_src`, LULC by default; Stephen:
          // "zoom out goes back to last raster clicked or default lulc"), and never
          // S2 below its pyramid's floor, where its tiles are empty by construction
          // ("no s2 raster here").
          const rasterSrc = () => {
            let src = cfg.raster_src || "nlcd";
            if (!hexesDrawn() && cfg.show_hexes !== false) src = cfg.wide_src || "nlcd";
            if (src === "s2" && map && map.getZoom() < (cfg.s2_min_z || 9)) src = "nlcd";
            return src;
          };
          const rasterOpacity = () => (hexesDrawn() && rasterUnder()
            ? (cfg.raster_opacity == null ? 0.6 : cfg.raster_opacity) : 1);
          // the hexagons' own opacity (the strip's second slider): deck multiplies
          // it onto the alpha the paint already carries. The boundaries and the
          // picked cell's outline keep their own, so a faint paint stays readable.
          const hexOpacity = () => (cfg.hex_opacity == null ? 1 : cfg.hex_opacity);

          function layers() {
            const out = [];
            // two kernel-served rasters, NLCD and the Sentinel-2 mosaic, one visible
            // at a time (rasterSrc); both pushed first so they sit under the hexagons
            const mkRaster = (src, maxZ, extent) => new TileLayer({
              // the id carries the refresh generation: a bump is a NEW layer to
              // deck, so its tile cache goes with the old one and every tile in
              // view is asked for again (the strip's `refresh`). The S2 layer adds
              // its own generation, bumped by the `scale` slider (a kernel-side
              // gain on the TCI bytes), so a scale change re-asks for S2 tiles only.
              id: src + (cfg.tilegen ? "-" + cfg.tilegen : "") + (src === "s2" && cfg.s2gen ? "-s" + cfg.s2gen : ""),
              getTileData: getTileDataFor(src),
              onTileError: (e) => { if (!e || e.name !== "AbortError") say(src + " tile: " + ((e && e.message) || e)); },
              tileSize: cfg.tile || 256,
              minZoom: 0, maxZoom: maxZ,
              extent: extent,
              visible: rasterOn() && rasterSrc() === src,
              opacity: rasterOpacity(),
              // "best-available" (deck's default), NOT "no-overlap": no-overlap
              // hides a parent tile as soon as ONE of its children has loaded, so
              // every child still in flight (or never coming, e.g. one whose
              // request was aborted by the next camera move) is a HOLE in the
              // raster. best-available keeps the coarser tile under the gap, so the
              // worst case is blurry rather than missing. The cost is a transient
              // double-blend where parent and child overlap while the child loads,
              // visible only under the raster-under slider.
              refinementStrategy: "best-available",
              beforeId: cfg.labels_slot || "watername_ocean",
              renderSubLayers: (p) => {
                if (!p.data) return null;
                const {west, south, east, north} = p.tile.bbox;
                return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
              },
            });
            out.push(mkRaster("nlcd", 13, cfg.extent || null));
            // the S2 pyramid ends at z14 (10 m); below z9 the kernel answers empty
            out.push(mkRaster("s2", 14, null));
            // Two hexagon layers, one on the map at a time. The FLAT paints (NLCD,
            // clusters) need no coverage, so they take the stock H3HexagonLayer on
            // its highPrecision path: every cell's own boundary, tessellated in the
            // browser, no shared mesh, no drift, no gaps (Stephen: "a separate h3
            // hexagon layer for those two"). Agreement keeps the coverage column.
            if (dataObj && cfg.flat) out.push(new H3HexagonLayer({
              id: "hexes-flat",
              data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              opacity: hexOpacity(),
              visible: cfg.show_hexes !== false,
              pickable: true,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            // the coverage paints: the same exact rings, each scaled by its cell's
            // coverage (highPrecision routes deck to _renderPolygonLayer, overridden
            // above; nothing here is the shared column mesh any more)
            if (dataObj && !cfg.flat) out.push(new CoverageH3Layer({
              id: "hexes",
              data: dataObj,
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              getCoverage: (_, {index}) => cov[index],
              updateTriggers: {getFillColor: [dataObj], getCoverage: [dataObj], getHexagon: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              opacity: hexOpacity(),
              visible: cfg.show_hexes !== false,
              pickable: true,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            // boundaries of the low-agreement patches, one per NLCD class (grouped
            // and dissolved in the kernel by DuckDB's h3 extension, each ring
            // carrying its class's color): one PathLayer over every closed ring. The
            // BOUNDARIES TOGGLE OWNS THEM (Stephen, 2026-08-27 night): they draw
            // whatever the paint, hexagons shown or not, over a raster paint too;
            // the kernel keeps folding for them while they are on, so they follow
            // the camera, and it clears them below hex_zoom (the "old rings at zoom
            // 9.7" report was rings of a frame the camera had left, not this).
            const edgeZoomOk = () => !!map && map.getZoom() >= (cfg.hex_zoom || 9);
            // the filled blobs go UNDER their own rings (and over the hexagons):
            // one flat color per patch, averaged from its cells in the kernel
            if (cfg.edges && cfg.bfill && fillTable && fillTable.numRows && edgeZoomOk()) out.push(new GeoArrowSolidPolygonLayer({
              id: "fills",
              data: fillTable,
              getPolygon: fillTable.getChild("geometry"),
              getFillColor: fillTable.getChild("color"),
              filled: true, extruded: false,
              opacity: hexOpacity(),   // the same slider as the hexagons (Stephen)
              pickable: true,   // a click inside a patch names the patch (kernel-side contains test) AND the cell
              _validate: false,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            if (cfg.edges && edgeTable && edgeTable.numRows && edgeZoomOk()) out.push(new GeoArrowPathLayer({
              id: "edges",
              data: edgeTable,
              getPath: edgeTable.getChild("geometry"),
              getColor: edgeTable.getChild("color"),
              widthUnits: "pixels", getWidth: cfg.edge_width || 2, widthMinPixels: 1,
              jointRounded: true,
              _validate: false,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            // the picked cell: its own color stays, a gold outline from its boundary
            if (cfg.hit && hexIndex.has(cfg.hit)) {
              let ring = null;
              try { ring = cellToBoundary(cfg.hit, true); } catch (e) { ring = null; }
              if (ring) out.push(new PathLayer({
                id: "picked",
                data: [ring],
                getPath: (d) => d,
                getColor: [255, 200, 40, 255],
                widthUnits: "pixels", getWidth: 3, widthMinPixels: 2,
                beforeId: cfg.labels_slot || "watername_ocean",
              }));
            }
            return out;
          }
          function update() { if (overlay) overlay.setProps({layers: layers()}); }

          function labels(on) {
            if (!map || !map.isStyleLoaded()) return;
            const st = map.getStyle();
            if (!st || !st.layers) return;
            st.layers.forEach((l) => {
              if (l.layout && l.layout["text-field"] !== undefined)
                map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
            });
          }

          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            model.set("view", JSON.stringify({
              longitude: c.lng, latitude: c.lat, zoom: map.getZoom(),
              w: mapEl.clientWidth, h: mapEl.clientHeight, n: ++seq,
            }));
            model.save_changes();
          }

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            map = new maplibregl.Map({
              container: mapEl, style: STYLES.labels,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: true},
            });
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            map.addControl(new maplibregl.FullscreenControl(), "top-right");
            overlay = new MapboxOverlay({
              interleaved: true,
              layers: [],
              onClick: (info) => {
                // deck's GPU pick first; when it returns nothing (it did on every
                // click here, as in the counties film: interleaved inside marimo's
                // shadow DOM), the click's lon/lat -> h3-js cell at the frame's res
                let cell = info && info.layer && (info.layer.id === "hexes" || info.layer.id === "hexes-flat") && info.index >= 0 ? hexes[info.index] : null;
                let how = "gpu";
                if (!cell && info && info.coordinate && res >= 0) {
                  try { const h = latLngToCell(info.coordinate[1], info.coordinate[0], res); if (hexIndex.has(h)) cell = h; how = "h3"; }
                  catch (e) { how = "h3: " + e.message; }
                }
                if (cfg.debug) say("pick " + how + " " + cell);
                model.set("pick", JSON.stringify({cell, lon: info && info.coordinate ? info.coordinate[0] : null,
                  lat: info && info.coordinate ? info.coordinate[1] : null, n: ++seq}));
                model.save_changes();
              },
              onError: (e) => say("deck: " + (e && e.message ? e.message : e)),
            });
            map.addControl(overlay);
            window.__aefTiles = tstat;  // {asked, got, empty, err, abort}: the raster's own tally
          window.__aefMap = () => map;  // the drive harness flies the camera through it
          window.__aefLayers = () => layers().map(l => l.id + ":" + l.constructor.layerName + (l.props.visible === false ? " (hidden)" : "") + " op=" + l.props.opacity);
          if (cfg.debug) window.__aef = {overlay, map, model, reload, get N() { return N; }, get colors() { return colors; }, get cov() { return cov; }, get dataObj() { return dataObj; }, get edgeTable() { return edgeTable; }, get raw() { return raw; }, get cfg() { return cfg; }};
            map.on("load", () => { labels(cfg.labels !== false); update(); sendView(); });
            map.on("moveend", sendView);
            map.on("zoom", () => update());
            map.on("error", (e) => { if (e && e.error && e.error.message) say("map: " + e.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            document.addEventListener("fullscreenchange", () => { setTimeout(() => { try { map.resize(); } catch (e) {} }, 50); });
          }

          let pendingLoad = null;
          let needCells = false;
          const flush = () => {  // cells/colors/cov land as three trait changes: rebuild once
            pendingLoad = null;
            try { if (needCells) loadCells(); needCells = false; loadAttrs(); loadEdges(); loadFills(); update(); }
            catch (e) { say("load: " + e.message); console.error(e); }
          };
          const reload = () => { needCells = true; if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          const reattr = () => { if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          model.on("change:cells", () => { grab("cells"); reload(); });
          model.on("change:colors", () => { grab("colors"); reattr(); });
          model.on("change:cov", () => { grab("cov"); reattr(); });
          model.on("change:edges", () => { grab("edges"); reattr(); });
          model.on("change:fills", () => { grab("fills"); reattr(); });
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (cfg.height && cfg.height !== was.height && !document.fullscreenElement) mapEl.style.height = cfg.height + "px";
            if (cfg.labels !== was.labels) labels(cfg.labels !== false);
            update();
          });
          try { grab("cells"); grab("colors"); grab("cov"); grab("edges"); grab("fills"); loadCells(); loadAttrs(); loadEdges(); loadFills(); boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (DeckMap,)


@app.cell
def _(
    DeckMap,
    EDGE_THR,
    HOME,
    LABELS_SLOT,
    RASTER_TILE,
    HEX_OPACITY,
    RASTER_UNDER,
    RASTER_UNDER_OPACITY,
    RASTER_UNDER_SRC,
    S2_SCALE,
    json,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter -----------------
    deck = DeckMap(config=json.dumps({
        "height": 720, "home": dict(HOME), "show_raster": True, "show_hexes": True, "labels": True,
        "hex_zoom": 9.0, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "raster_under": RASTER_UNDER, "raster_opacity": RASTER_UNDER_OPACITY,
        "hex_opacity": HEX_OPACITY, "s2gen": 0,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "task": None, "loop": None,
        "paint": "nlcd", "show_raster": True, "show_hexes": True,
        "sel": set(), "hit": None, "memo": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "dres": 0,  # the strip's res offset; a statement about the box it was set on
        "inv": False,  # reversed alpha: disagreeing cells solid
        "acol": False,  # color by agreement (the ramp) instead of NLCD's colors
        "acov": False,  # agreement coverage on a flat paint (ACOV_PAINTS: NLCD H3)
        "labels": True,
        "tilegen": 0,  # the strip's refresh: a bump rebuilds the widget's TileLayer
        "s2gen": 0,  # the S2 `scale` slider: a bump rebuilds the S2 TileLayer only
        "s2scale": S2_SCALE,  # the gain the S2 tiles are encoded with
        "last_raster": "nlcd",  # the raster paint clicked last: what shows wide, below HEX_ZOOM
        # the NLCD raster kept UNDER the hexagons, and its opacity there
        "runder": RASTER_UNDER, "ropac": RASTER_UNDER_OPACITY,
        "hopac": HEX_OPACITY,  # the hexagons' own opacity (a deck layer opacity)
        "usrc": RASTER_UNDER_SRC,  # which raster the underlay is: "nlcd" | "s2"
        "idx": "ndvi",  # the Sentinel-2 H3 paint's index
        "edges": False, "thr": EDGE_THR,  # the low-agreement boundaries and their threshold
        "bfill": False,  # the boundaries filled, each patch averaged from its cells
        "hide": False,  # the boundaries' inverse: cells at/above thr not drawn (still pickable)
        "edges_sent": None,  # (frame, thr, colors) the widget holds
        "pending_force": False,  # the queued serve is a forced one (res, refresh)
        "runs": 0,  # how often the wiring cell has run
    }
    deck  # the cell's LAST statement: what marimo displays
    return HOLD, deck


@app.cell
def _(EDGE_THR, HEX_OPACITY, HudControls, RASTER_UNDER, RASTER_UNDER_OPACITY, RASTER_UNDER_SRC, S2_SCALE, mo):
    hud = mo.ui.anywidget(HudControls(
        thr0=str(EDGE_THR),
        runder0="1" if RASTER_UNDER else "0",
        ropac0=str(RASTER_UNDER_OPACITY),
        hopac0=str(HEX_OPACITY),
        usrc0=RASTER_UNDER_SRC,
        s2scale0=str(S2_SCALE),
    ))
    hud
    return (hud,)


@app.cell
def _(
    ACOV_PAINTS,
    AEF_YEARS,
    CHG_MIN,
    CLASSES,
    CLUSTER_HEX,
    EDGE_ALPHA,
    EDGE_MIN_CELLS,
    EDGE_WIDTH,
    FA,
    HEX_ZOOM,
    HOLD,
    HOME,
    S2_IDX,
    S2_IDX_WHAT,
    S2_TILE_MIN_Z,
    SETTLE,
    VERDICTS,
    VIEW_W,
    YEAR,
    YEAR_S2,
    aef_fold,
    asyncio,
    build_frame,
    con,
    contains,
    deck,
    edges_for,
    hud,
    json,
    lc_bounds,
    lc_fold,
    lc_raster_clear,
    lc_raster_stats,
    lc_tile_png,
    legend_for,
    math,
    np,
    pad_box,
    res_for_view,
    s2_fold,
    s2_raster_clear,
    s2_raster_stats,
    s2_set_scale,
    s2_tile_png,
    time,
    traceback,
    urllib,
    view_to_bbox,
):
    # ---- wiring: the camera loop and the strip. Re-runs freely (un-observes its
    # old handlers first); the map cell never re-runs.
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] = HOLD.get("runs", 0) + 1  # how often this cell has run (the status says)
    async def _tile_fn(z, x, y, src="nlcd"):
        try:
            if src == "s2":
                return await s2_tile_png(z, x, y)
            return await lc_tile_png(z, x, y)
        except Exception as e:
            HOLD["tile_errs"] = HOLD.get("tile_errs", 0) + 1
            HOLD["tile_err"] = f"{type(e).__name__}: {e}"
            raise

    deck.tile_fn = _tile_fn
    _HEX_PAINTS = ("nlcd", "verdict", "disp", "s2", "clusters")
    _RASTER_WHAT = {
        "raster": f"LCMS {YEAR} change raster (its own tiles)",
        "s2raster": f"Sentinel-2 raster (the {YEAR_S2} mosaic's tiles; nothing below zoom {S2_TILE_MIN_Z})",
    }

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _say_lines(head):
        _say("\n".join(
            [head] + [HOLD[k] for k in ("edge_note", "hide_note", "raster_note") if HOLD.get(k)]))

    def _cfg(**kw):
        c = json.loads(deck.config or "{}")
        c.update(kw)
        deck.config = json.dumps(c)

    def _show():
        HOLD["show_raster"] = HOLD["paint"] is not None
        HOLD["show_hexes"] = HOLD["paint"] in _HEX_PAINTS
        _src = {"raster": "nlcd", "s2raster": "s2"}.get(
            HOLD["paint"], HOLD["usrc"] if HOLD["runder"] else "nlcd")
        _flat = HOLD["paint"] in _HEX_PAINTS and not (
            HOLD["acov"] and HOLD["paint"] in ACOV_PAINTS)
        if HOLD["paint"] in ("raster", "s2raster"):
            HOLD["last_raster"] = _src
        _cfg(show_raster=HOLD["show_raster"], show_hexes=HOLD["show_hexes"],
             flat=_flat, raster_src=_src, wide_src=HOLD["last_raster"], s2_min_z=S2_TILE_MIN_Z,
             raster_under=HOLD["runder"], raster_opacity=HOLD["ropac"],
             hex_opacity=HOLD["hopac"])

    _show()
    _cfg(extent=list(lc_bounds), hex_zoom=HEX_ZOOM, edges=HOLD["edges"],
         bfill=HOLD["bfill"], edge_width=EDGE_WIDTH)

    def _hexes_off(msg):
        if HOLD["sent"] is not None or HOLD["edges_sent"] is not None:
            with deck.hold_sync():
                deck.cells, deck.colors, deck.cov = b"", b"", b""
                deck.edges, deck.fills = b"", b""
            HOLD["sent"], HOLD["edges_sent"] = None, None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        HOLD["edge_note"] = ""
        try:
            hud.widget.legend = "[]"
            hud.widget.panel = ""
        except Exception:
            pass
        _say_lines(msg)

    def _say_dres():
        try:
            hud.widget.dres = str(HOLD["dres"])
        except Exception:
            pass

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

    def _paint():
        fr = HOLD["frame"]
        if fr is None:
            return False
        rgba = fr["fill"](HOLD["paint"], HOLD["sel"], None, HOLD["inv"], HOLD["acol"], HOLD["acov"], HOLD["idx"],
                          HOLD["hide"], HOLD["thr"])
        cov = fr["coverage"](HOLD["paint"], HOLD["inv"], HOLD["acov"])
        if HOLD["hide"] and HOLD["show_hexes"]:
            _n_hid = int(np.sum(fr["agree"] >= HOLD["thr"]))
            HOLD["hide_note"] = (
                f"hidden: {_n_hid:,} cells at or above {HOLD['thr']:.2f} · "
                f"{len(fr['cellid']) - _n_hid:,} drawn (hidden cells still pick)")
        else:
            HOLD["hide_note"] = ""
        ed = None
        if HOLD["edges"]:
            ed = edges_for(fr, HOLD["thr"], EDGE_MIN_CELLS, EDGE_ALPHA,
                           HOLD["acol"], HOLD["inv"], HOLD["bfill"])
            HOLD["edge_note"] = (
                f"boundaries: {ed['n_low']:,} cells below {HOLD['thr']:.2f} · {ed['blobs']:,} same-class blobs ≥ {EDGE_MIN_CELLS} cells"
                + (f" · largest {ed['max_km2']:,.1f} km²" if ed["blobs"] else "")
                + f" · {ed['rings']:,} rings"
                + (f" · {ed['polys']:,} filled" if HOLD["bfill"] else "")
                + f" · {ed['ms']} ms"
            )
        else:
            HOLD["edge_note"] = ""
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None,
             edges=HOLD["edges"], bfill=HOLD["bfill"])
        with deck.hold_sync():
            if HOLD["sent"] is not fr:
                deck.cells = fr["cellid"].astype("<u8").tobytes()
                HOLD["sent"] = fr
            deck.colors = rgba.tobytes()
            deck.cov = cov.astype("<f4").tobytes()
            _ekey = (id(fr), round(HOLD["thr"], 3), HOLD["bfill"], HOLD["acol"], HOLD["inv"])
            if ed is not None and HOLD["edges_sent"] != _ekey:
                deck.edges = ed["ipc"]
                deck.fills = ed["fipc"]
                HOLD["edges_sent"] = _ekey
        try:
            hud.widget.legend = json.dumps(
                legend_for(fr, HOLD["paint"], HOLD["acol"], HOLD["inv"], HOLD["idx"])
                if HOLD["show_hexes"] else [])
        except Exception:
            pass
        return True

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · the raster as its own tiles · zoom in past {HEX_ZOOM:g} for the hexagons")
            return
        if not (HOLD["show_hexes"] or HOLD["edges"]):
            what = _RASTER_WHAT.get(HOLD["paint"], "nothing on")
            _say_lines(f"zoom {vsd['zoom']:.1f} · {what} · pick a hexagon paint (or boundaries) for the fold")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and not force:
            ladder = res_for_view(vsd, box, HOLD["dres"])
            if ladder >= HOLD["res"]:
                note = f" · finer available (res {ladder}: press res +)" if ladder > HOLD["res"] else ""
                _say(HOLD.get("last_status", "") + " · held" + note)
                return
        if not inside and HOLD["box"] is not None:
            HOLD["dres"] = 0
            _say_dres()
        res = res_for_view(vsd, box, HOLD["dres"])
        key = (res, tuple(round(v, 3) for v in box))
        t0 = time.time()
        _say(f"res {res} · folding LCMS {YEAR}, AlphaEarth {', '.join(str(y) for y in AEF_YEARS)}, Sentinel-2 {YEAR_S2}… (wiring run {HOLD['runs']})")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            got = await asyncio.gather(
                lc_fold(box, res), s2_fold(box, res), *(aef_fold(box, res, y) for y in AEF_YEARS)
            )
            (nl, s1), (sp, s3) = got[0], got[1]
            aef_by_year = {y: g[0] for y, g in zip(AEF_YEARS, got[2:])}
            s2s = " · ".join(g[1] for g in got[2:])
            if nl is None or nl.num_rows == 0:
                _say(f"res {res} · {s1}")
                return
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, nl, aef_by_year, sp)
            stats = f"res {res} · {s1} · {s2s} · {s3} · frame {time.time() - t1:.1f} s"
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        HOLD["last_status"] = f"{stats} · {fr['score']} · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
        _say_lines(HOLD["last_status"])

    async def refresh(vs, force=False, settle=True):
        """ONE serve at a time (the fix for Stephen's first flight, 2026-08-31:
        "the buttons are extremely unresponsive ... stuck on Sentinel-2"): every
        request, from the camera, a paint button, res or refresh, comes through
        here. A serve in flight makes the request the single PENDING one (the
        latest wins; a forced one stays forced) and it runs when the serve
        finishes. Before, a button pressed during a fold spawned a second whole
        fold beside the first (four AEF reads each, contending for the same
        DataFusion table names), and the kernel fell behind every click."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            HOLD["pending_force"] = HOLD.get("pending_force", False) or force
            return
        HOLD["busy"] = True
        try:
            while True:
                if settle:
                    await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    force, HOLD["pending_force"] = HOLD.get("pending_force", False), False
                    settle = True
                    continue
                await _serve(vs, force)
                vs = HOLD["pending"]
                if vs is None:
                    return
                force, HOLD["pending"], HOLD["pending_force"] = HOLD.get("pending_force", False), None, False
                settle = False
        except Exception as exc:
            _say(f"failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"], HOLD["pending_force"] = False, None, False

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _request(vs=None, force=False):
        """Ask for a serve of `vs` (the camera's last view by default) through the
        one-at-a-time loop; never a bare `_serve`."""
        if vs is None:
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
            deck.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    deck.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    _CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307}
    _td = "padding:.1rem .6rem .1rem 0;white-space:nowrap"
    _th = "padding:.1rem .6rem .1rem 0;text-align:left;opacity:.6;font-weight:500"

    def _chip(rgb):
        return f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:rgb{tuple(rgb)};margin-right:.35rem;vertical-align:-1px'></span>"

    def _f(v, d=2):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    def _analyze_html(fr):
        """The view's summary for the strip's panel: per class (the cell's main
        change, or Stable), the share of cells and area, how many the embedding
        saw move, the displacement and agreement medians, and NBR split by
        moved / still; then the 2x2 of verdicts; then the difference clusters'
        LCMS make-up."""
        cls, verdict = fr["cls"], fr["verdict"]
        n = max(1, len(cls))
        km2 = _CELL_KM2.get(HOLD["res"], 0.0)
        head = (
            f"<b>res {HOLD['res']}</b> · {n:,} cells · {n * km2:,.0f} km² · LCMS {YEAR} · "
            f"AlphaEarth {', '.join(str(y) for y in fr['years'])}"
            + (f" · D0 {fr['D0']:.3f} (the stable cells' {100 * (1 - FA):.0f}th percentile of shift), τ {fr['tau']:.4f}"
               if fr["has_aef"] and not np.isnan(fr["D0"]) else " · unscored (no embedding, or too few stable cells)")
        )
        con.register("cur_cells", fr["cells"])
        rows = con.execute(f"""
            SELECT cls, name, count(*) AS cells,
                   avg(p_chg), avg(p_dist),
                   count(*) FILTER (WHERE disp IS NOT NULL) AS scored,
                   count(*) FILTER (WHERE moved) AS moved,
                   median(disp), median(disp_pre), median(disp_in), median(disp_out),
                   median(agree), avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END) FILTER (WHERE agree IS NOT NULL),
                   median(nbr) FILTER (WHERE moved), median(nbr) FILTER (WHERE NOT moved),
                   median(ndvi) FILTER (WHERE moved), median(ndvi) FILTER (WHERE NOT moved)
            FROM cur_cells GROUP BY cls, name ORDER BY cells DESC
        """).fetchall()
        trs = []
        for (c, nm, cnt, pc, pd, sc, mv, dm, dp, di, do, ag, blw, nbm, nbs, ndm, nds) in rows:
            rgb = CLASSES.get(int(c), ("?", (128, 128, 128)))[1]
            trs.append(
                f"<tr><td style='{_td}'>{_chip(rgb)}{nm}</td>"
                f"<td style='{_td};text-align:right'>{100 * cnt / n:.1f}%</td>"
                f"<td style='{_td};text-align:right'>{cnt * km2:,.0f} km²</td>"
                f"<td style='{_td};text-align:right'>{_f(pc)}</td>"
                + (f"<td style='{_td};text-align:right'>{100 * mv / sc:.0f}%</td>"
                   f"<td style='{_td};text-align:right'>{_f(dm, 3)} <span style='opacity:.6'>({_f(dp, 3)} pre, {_f(di, 3)} in, {_f(do, 3)} out)</span></td>"
                   f"<td style='{_td};text-align:right'>{_f(ag)}</td>"
                   f"<td style='{_td};text-align:right'>{_f(100 * blw if blw is not None else None, 0)}%</td>"
                   if sc else f"<td style='{_td}' colspan=4><span style='opacity:.6'>unscored</span></td>")
                + (f"<td style='{_td};text-align:right'>{_f(nbm)} / {_f(nbs)}</td>"
                   f"<td style='{_td};text-align:right'>{_f(ndm)} / {_f(nds)}</td>" if fr["has_s2"] else "")
                + "</tr>"
            )
        table = (
            f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr>"
            f"<th style='{_th}'>LCMS {YEAR}</th><th style='{_th}'>of cells</th><th style='{_th}'>area</th>"
            f"<th style='{_th}' title='mean share of the cells'' pixels carrying a change code (1..14)'>changed px</th>"
            f"<th style='{_th}' title='cells whose AlphaEarth shift is above D0'>AEF moved</th>"
            f"<th style='{_th}' title='1 - cos between the cell''s vectors in consecutive years, the largest step (pre: {YEAR - 2} to {YEAR - 1}, in: {YEAR - 1} to {YEAR}, out: {YEAR} to {YEAR + 1})'>shift p50</th>"
            f"<th style='{_th}'>agreement p50</th><th style='{_th}'>below 0.5</th>"
            + (f"<th style='{_th}' title='Sentinel-2 {YEAR_S2}: median of the cells AlphaEarth saw move / of the cells it did not; low NBR is burn or bare'>NBR moved / still</th>"
               f"<th style='{_th}' title='same split'>NDVI moved / still</th>" if fr["has_s2"] else "")
            + "</tr>" + "".join(trs) + "</table>"
        )
        vt = ""
        if fr["has_aef"]:
            items = []
            for v in (1, 2, 3, 0):
                m = verdict == v
                if not m.any():
                    continue
                items.append(
                    f"<tr><td style='{_td}'>{_chip(VERDICTS[v][1])}{VERDICTS[v][0]}</td>"
                    f"<td style='{_td};text-align:right'>{100 * m.sum() / n:.1f}%</td>"
                    f"<td style='{_td};text-align:right'>{m.sum() * km2:,.0f} km²</td></tr>")
            vt = (
                f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr>"
                f"<th style='{_th}' title='LCMS says change: at least {100 * CHG_MIN:.0f}% of the cell''s pixels carry a change code; AEF says change: the shift is above D0'>verdict</th>"
                f"<th style='{_th}'>of cells</th><th style='{_th}'>area</th></tr>" + "".join(items) + "</table>"
            )
        clus = ""
        clu = fr["clu"]
        if fr["has_aef"] and len(clu) and clu.max() >= 0:
            items = []
            for k in range(int(clu.max()) + 1):
                m = clu == k
                if not m.any():
                    continue
                cc, cn = np.unique(cls[m], return_counts=True)
                top = sorted(zip(cn, cc), reverse=True)[:3]
                mix = ", ".join(f"{100 * nn / m.sum():.0f}% {CLASSES.get(int(c), ('?',))[0]}" for nn, c in top)
                d = fr["disp"][m]
                d = d[~np.isnan(d)]
                chip = f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{CLUSTER_HEX[k % len(CLUSTER_HEX)]};margin-right:.35rem;vertical-align:-1px'></span>"
                items.append(f"<tr><td style='{_td}'>{chip}Δ cluster {k}</td><td style='{_td};text-align:right'>{100 * m.sum() / n:.1f}%</td>"
                             f"<td style='{_td};text-align:right'>{_f(float(np.median(d)) if len(d) else None, 3)}</td><td style='{_td};opacity:.75'>{mix}</td></tr>")
            clus = (
                f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr><th style='{_th}'>AlphaEarth Δ cluster</th><th style='{_th}'>of cells</th><th style='{_th}'>shift p50</th><th style='{_th}'>made of (LCMS)</th></tr>"
                + "".join(items) + "</table>"
            )
        return head + table + vt + clus

    def _selection_panel(fr):
        if not HOLD["sel"]:
            return ""
        con.register("cur_cells", fr["cells"])
        if HOLD["paint"] == "clusters":
            rows = con.execute("""
                SELECT 'Δ cluster ' || cluster, count(*), round(median(agree), 2),
                       round(100 * avg(CASE WHEN moved THEN 1 ELSE 0 END), 0), mode(name), round(median(disp), 3)
                FROM cur_cells WHERE cluster IN (SELECT UNNEST(?)) GROUP BY cluster ORDER BY 2 DESC
            """, [[k - 100 for k in HOLD["sel"]]]).fetchall()
        elif HOLD["paint"] == "verdict":
            rows = con.execute("""
                SELECT verdict_name, count(*), round(median(agree), 2),
                       round(100 * avg(CASE WHEN moved THEN 1 ELSE 0 END), 0), mode(name), round(median(disp), 3)
                FROM cur_cells WHERE verdict IN (SELECT UNNEST(?)) GROUP BY verdict_name ORDER BY 2 DESC
            """, [[k - 200 for k in HOLD["sel"] if k >= 200]]).fetchall()
        else:
            rows = con.execute("""
                SELECT name, count(*), round(median(agree), 2),
                       round(100 * avg(CASE WHEN moved THEN 1 ELSE 0 END), 0), mode(verdict_name), round(median(disp), 3)
                FROM cur_cells WHERE cls IN (SELECT UNNEST(?)) GROUP BY name ORDER BY 2 DESC
            """, [list(HOLD["sel"])]).fetchall()
        return " · ".join(
            f"<b>{nm}</b>: {cnt:,} cells"
            + (f", agreement p50 {p50:.2f}, {pct:.0f}% moved" if p50 is not None else "")
            + (f", shift p50 {dd:.3f}" if dd is not None else "")
            + (f", mostly <i>{mo_}</i>" if mo_ else "")
            for nm, cnt, p50, pct, mo_, dd in rows
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
                hud.widget.panel = _selection_panel(fr)
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT name, maj_name, p_chg, p_dist, disp, disp_pre, disp_in, disp_out, moved, agree, verdict_name, cluster, purity "
                "FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            patch = ""
            if HOLD["edges"] and HOLD["bfill"] and p.get("lon") is not None:
                try:
                    ed = edges_for(fr, HOLD["thr"], EDGE_MIN_CELLS, EDGE_ALPHA, HOLD["acol"], HOLD["inv"], True)
                    fp = ed.get("fpoly")
                    if fp is not None and fp.num_rows:
                        con.register("cur_fills", fp)
                        pr = con.execute("""
                            SELECT cls, ncell, km2, agree FROM cur_fills
                            WHERE ST_Contains(ST_GeomFromWKB(wkb), ST_Point(?, ?)) LIMIT 1
                        """, [float(p["lon"]), float(p["lat"])]).fetchone()
                        if pr is not None:
                            pcls, pn, pkm2, pag = pr
                            patch = (
                                f"<b>patch</b>: {CLASSES.get(int(pcls), ('?',))[0]}, {int(pn):,} cells, "
                                f"{pkm2:.2f} km², mean agreement {pag:.2f} · "
                            )
                except Exception as e:
                    patch = f"<span style='opacity:.7'>patch: {e}</span> · "
            spec = ""
            if fr["has_s2"] and r is not None:
                cols = ", ".join(f"c.{nm}, median(v.{nm})" for nm in S2_IDX)
                sr = con.execute(f"""
                    SELECT {cols} FROM cur_cells c JOIN cur_cells v ON v.cls = c.cls
                    WHERE c.cell = ? GROUP BY ALL
                """, [cell]).fetchone()
                if sr is not None and sr[0] is not None and not np.isnan(sr[0]):
                    spec = f"; Sentinel-2 {YEAR_S2} " + ", ".join(
                        f"{nm.upper()} <span style='opacity:.6'>({S2_IDX_WHAT.get(nm, '')})</span> {sr[2 * k]:.2f} "
                        f"<span style='opacity:.6'>(class here {sr[2 * k + 1]:.2f})</span>"
                        for k, nm in enumerate(S2_IDX)
                    )
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                hud.widget.panel = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                nm, majn, pc, pd, dsp, dp, di, do, mv, ag, vn, ck, pur = r
                scored = ag is not None and not np.isnan(ag)
                hud.widget.panel = (
                    patch
                    + f"<b>{nm}</b>{where}: {100 * pc:.0f}% of pixels changed ({100 * pd:.0f}% disturbed), majority {majn} ({pur:.2f})"
                    + (f"; AlphaEarth shift {_f(dsp, 3)} <span style='opacity:.6'>({_f(dp, 3)} pre, {_f(di, 3)} in, {_f(do, 3)} out; D0 {fr['D0']:.3f})</span>, "
                       f"{'moved' if mv else 'did not move'}, agreement {ag:.2f}, verdict <i>{vn}</i>"
                       + (f", Δ cluster {ck}" if ck is not None and ck >= 0 else "")
                       if scored else "; AlphaEarth unscored")
                    + spec
                )
        except Exception as e:
            hud.widget.panel = f"<span style='opacity:.7'>click: {e}</span>"
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            deck.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    deck.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    def _photon_first(query, vs):
        params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            params["lon"] = round(vs["longitude"], 4)
            params["lat"] = round(vs["latitude"], 4)
        url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "x-sql-marimo aef lcms deck notebook"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        feats = data.get("features") or []
        if not feats:
            return None
        f = feats[0]
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        name = ", ".join(str(v) for v in (p.get("name"), p.get("city"), p.get("state")) if v) or query
        return name, float(lon), float(lat), p.get("extent")

    async def _search(q):
        vs = _vsd(HOLD.get("vs"))
        try:
            hit = await asyncio.get_running_loop().run_in_executor(None, _photon_first, q, vs)
        except Exception as e:
            _say(f"search error: {type(e).__name__}: {e}")
            return
        if hit is None:
            _say(f"no match: {q}")
            return
        name, lon, lat, ext = hit
        w = vs.get("w") or VIEW_W
        if ext and len(ext) == 4:
            span = max(abs(ext[2] - ext[0]), abs(ext[1] - ext[3]) * 2, 0.01)
            zoom = math.log2(360.0 * (w / 512) / span) - 0.3
        else:
            zoom = 10.0
        zoom = max(3.5, min(13.5, zoom))
        deck.send({"kind": "fly", "lon": lon, "lat": lat, "zoom": zoom, "duration": 2000})
        _say(f"→ {name} · zoom {zoom:.1f}")

    _PAINT_NAME = {"raster": "LCMS raster", "s2raster": "S2 raster", "nlcd": "LCMS H3",
                   "verdict": "verdict H3", "disp": "AEF shift H3",
                   "s2": "S2 H3", "clusters": "AEF Δ clusters H3", None: "nothing"}

    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        _was = HOLD["paint"]
        HOLD["paint"] = c.get("paint", "nlcd")
        HOLD["sel"] = {int(x) for x in c.get("sel", [])}
        HOLD["inv"] = bool(c.get("inv", False))
        HOLD["acol"] = bool(c.get("acol", False))
        _cov_was = HOLD["acov"]
        HOLD["acov"] = bool(c.get("acov", False))
        _r_was = (HOLD["runder"], HOLD["ropac"], HOLD["hopac"], HOLD["usrc"])
        HOLD["runder"] = bool(c.get("runder", False))
        HOLD["usrc"] = "s2" if c.get("usrc") == "s2" else "nlcd"
        HOLD["idx"] = c.get("idx") if c.get("idx") in S2_IDX else HOLD["idx"]
        for _k, _lo in (("ropac", 0.0), ("hopac", 0.05)):
            try:
                HOLD[_k] = min(1.0, max(_lo, float(c.get(_k, HOLD[_k]))))
            except (TypeError, ValueError):
                pass
        if (HOLD["runder"], HOLD["ropac"], HOLD["hopac"], HOLD["usrc"]) != _r_was:
            _show()
        if c.get("act") == "vis":
            return
        if c.get("act") == "s2scale":
            try:
                HOLD["s2scale"] = float(c.get("s2scale", HOLD["s2scale"]))
            except (TypeError, ValueError):
                return
            if s2_set_scale(HOLD["s2scale"]):
                HOLD["s2gen"] += 1
                _cfg(s2gen=HOLD["s2gen"])
                HOLD["raster_note"] = f"Sentinel-2 scale {HOLD['s2scale']:.1f}× · S2 tiles re-served"
                _say_lines(hud.widget.status.split("\n")[0] if hud.widget.status else "")
            return
        _ed_was = (HOLD["edges"], HOLD["thr"], HOLD["bfill"], HOLD["hide"])
        HOLD["edges"] = bool(c.get("edges", False))
        HOLD["bfill"] = bool(c.get("bfill", False))
        HOLD["hide"] = bool(c.get("hide", False))
        try:
            HOLD["thr"] = min(0.99, max(0.01, float(c.get("thr", HOLD["thr"]))))
        except (TypeError, ValueError):
            pass
        fr = HOLD["frame"]
        if HOLD["paint"] != _was:
            _show()
            if HOLD["paint"] in (None, "raster", "s2raster"):
                z = _vsd(HOLD["vs"])["zoom"]
                kept = " · fold kept" if fr is not None else ""
                if HOLD["edges"] and fr is not None:
                    kept = " · boundaries stay (they keep folding with the camera)"
                what = _RASTER_WHAT.get(HOLD["paint"], "nothing on")
                _paint()
                _say_lines(f"zoom {z:.1f} · {what}{kept}")
                return
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            if _paint():
                _say_lines(f"{_PAINT_NAME.get(HOLD['paint'], HOLD['paint'])} · {len(fr['cellid']):,} cells painted"
                           + (" · a fold is in flight, this view is queued" if HOLD["busy"] else "")
                           + (f" · {HOLD['last_status']}" if HOLD.get("last_status") else ""))
            else:
                _say_lines(f"{_PAINT_NAME.get(HOLD['paint'], HOLD['paint'])} · no fold held for this view · "
                           + ("a fold is in flight, this view is queued" if HOLD["busy"] else "folding…"))
            _request()
            return
        if c.get("act") == "dres":
            HOLD["dres"] = max(-2, min(2, int(c.get("dres", 0))))
            _say_dres()
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            _request(force=True)
            return
        if c.get("act") == "clear":
            hud.widget.panel = ""
            return
        if c.get("act") == "refresh":
            st = lc_raster_stats()
            n_png, n_blk = lc_raster_clear()
            s2st = s2_raster_stats()
            n_s2 = s2_raster_clear()
            HOLD["tilegen"] += 1
            _cfg(tilegen=HOLD["tilegen"])
            HOLD["raster_note"] = (
                f"raster: {st['served']:,} tiles served"
                + (f" ({st['ms'] / max(1, st['served']):.0f} ms each)" if st["served"] else "")
                + f", {st['blank']:,} all nodata"
                + (f", {HOLD['tile_errs']:,} FAILED (last: {HOLD['tile_err']})" if HOLD.get("tile_errs") else "")
                + f" · refresh dropped {n_png:,} tiles, {n_blk:,} blocks and asked deck for them again"
                + f" · S2: {s2st['served']:,} tiles served, {s2st['blank']:,} empty, {n_s2:,} dropped"
            )
            HOLD["tile_errs"], HOLD["tile_err"] = 0, ""
            _say_lines(HOLD.get("last_status") or "refresh")
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            _request(force=True)
            return
        if c.get("act") == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
            return
        if c.get("act") == "analyze":
            hud.widget.panel = _analyze_html(fr) if fr is not None else f"<span style='opacity:.7'>no fold in view (zoom in past {HEX_ZOOM:g} with the hexagons on)</span>"
            return
        if c.get("act") == "search":
            q = str(c.get("q") or "").strip()
            try:
                if q:
                    _say(f"searching: {q}")
                    HOLD["stask"] = _spawn(_search(q))
            except Exception as e:
                _say(f"search error: {type(e).__name__}: {e}")
            return
        if HOLD["acov"] != _cov_was:
            _show()
        if fr is not None:
            hud.widget.panel = _selection_panel(fr)
        painted = _paint()
        if HOLD["edges"] and not _ed_was[0] and not HOLD["show_hexes"]:
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            if not painted:
                _say_lines("boundaries · no fold held for this view · folding…")
            _request()
            return
        if (HOLD["edges"], HOLD["thr"], HOLD["bfill"], HOLD["hide"]) != _ed_was and HOLD.get("last_status"):
            _say_lines(HOLD["last_status"])
        elif not painted and HOLD["show_hexes"]:
            _say_lines("no fold held for this view (zoom in past the hexagon zoom, or move the map)")

    def _on_ctl(change):
        # every branch inside one guard: comm-handler exceptions are silent under
        # marimo, so whatever dies lands in the status line with its line number.
        # Defined BELOW the body (marimo mangles the reference otherwise).
        try:
            _on_ctl_body(change)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say_lines(f"control failed: {type(e).__name__}: {e}{where}")

    if HOLD.get("h_ctl") is not None:
        try:
            hud.widget.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    hud.widget.observe(_on_ctl, names="ctl")
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

    The tables below are DuckDB over the CURRENT view's cells (press the button
    after the map settles): `cls` / `name` (the cell's main change class, or
    Stable when under `CHG_MIN` of its pixels changed), `maj` / `maj_name` (the
    majority over every pixel), `p_chg` (share of pixels with any change code),
    `p_dist` (a disturbance code), `disp` (the AlphaEarth displacement, 1 -
    cos, the largest step in the window) with its steps `disp_pre` (Y-2 to
    Y-1), `disp_in` (Y-1 to Y), `disp_out` (Y to Y+1), `moved` (disp
    above the view's D0), `agree`, `verdict` / `verdict_name` (0 neither, 1
    both, 2 LCMS only, 3 AEF only), `cluster` (the difference vectors' k-means)
    and the Sentinel-2 indices `ndvi`, `ndwi`, `ndbi`, `nbr`, `mndwi`.
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
               round(median(disp_pre), 4) AS disp_pre_p50,
               round(median(disp_in), 4) AS disp_in_p50,
               round(median(disp_out), 4) AS disp_out_p50,
               round(median(agree), 3) AS agree_p50,
               round(avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END) * 100, 1) AS pct_below_half,
               round(median(nbr) FILTER (WHERE moved), 3) AS nbr_moved,
               round(median(nbr) FILTER (WHERE NOT moved), 3) AS nbr_still,
               round(median(ndvi), 3) AS ndvi_p50
        FROM view_cells GROUP BY cls, name ORDER BY cells DESC
        """,
        engine=con,
    )
    return (per_class,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None)
    verdicts = mo.sql(
        """
        PIVOT (SELECT name, verdict_name FROM view_cells)
        ON verdict_name USING count(*) GROUP BY name ORDER BY name
        """,
        engine=con,
    )
    return (verdicts,)


if __name__ == "__main__":
    app.run()
