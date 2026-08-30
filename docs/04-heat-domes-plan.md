# Plan: heat domes on the native HRRR grid, H3 as the transformation

Written 2026-08-30 after the prototype in `proto/` flew. This is the hand-off document
for the next session: what exists, what was measured, and the steps to fold it into a
full heat-domes notebook copied from `x-sql-marimo/xsql-hrrr-heat-domes.py`.

## The concept, in one paragraph

The map draws the HRRR field on its own 3 km Lambert grid: one textured quad per pixel,
mesh corners from the store's GeoTransform through pyproj, nearest-filter texture, no
warp, no resample. Every land pixel carries its H3 res 7 label (1,905,141 labels for
1,905,141 pixels: a relabel, one cell per pixel, measured) and the label's res 6 parent.
Nothing hexagonal is ever drawn. H3 is used for: picking (click to cell to pixel),
county identity (res 6 polyfill join), dome membership (threshold per pixel, decide per
parent cell, hand the decision back to the pixels), and later the DuckDB dome table
(dissolve, area, centroid, track) and cross-grid joins. The dome outline is drawn on
pixel edges.

## What exists (proto/)

- `proto/prep.py`: opens the icechunk store (metadata + 2D lat/lon only), builds
  pixel-corner web-mercator coords via pyproj from `spatial_ref` + GeoTransform, res 7
  labels with `h3ronpy.vector.coordinates_to_cells`, res 6 parents with
  `change_resolution`, land mask + county per pixel from the cached Overture counties
  parquet (DuckDB `h3_polygon_wkb_to_cells_experimental` 'center' at res 6, same as
  heat-hex), and a 48 h slab (Jul 1-2 2026, eastern dome) of T and RH for the 524 land
  blocks read block-wise from the disk mirror (`$TMPDIR/x-sql-marimo/hrrr-mirror`),
  NWS heat index per pixel, uint8. Writes `proto/cache/*.npy`. Runs in ~8 s with the
  caches warm.
- `proto/raster_mesh.py`: marimo notebook, one anywidget `RasterFilm`. deck.gl 9.3.10
  from esm.sh (same pins as x-sql-marimo), `SimpleMeshLayer` with
  `COORDINATE_SYSTEM.CARTESIAN` world coords, `material: false`, nearest texture
  params, texture repainted per frame from the land-pixel frame matrix via a LUT into
  a canvas. Pick = `latLngToCell(res 7)` into a label Map, with the LCC forward
  formula as the snap for empty cells. Accumulator (sustained heat) per pixel.
  Dome rule select: per pixel / res 6 any / majority / all. Pixel-edge outline as a
  `PathLayer` in world coords.
- `proto/fly.py`: `marimo run` + playwright, four screenshots in `proto/shots/`.

Bytes across the bridge: 42 MB frames (48 h x 879,370 land px), 15 MB mesh corners,
7 MB labels, 3.5 MB lidx, 3.5 MB parent idx, 1.7 MB county idx.

## Flight results

First flight 2026-08-30, headless Chrome 151 via playwright against `marimo run`:

- boot to first frame 25 s (the bridge carrying 42 MB frames + 15 MB corners + 12 MB
  indices; kernel prep from cache is ~2 s); label Map (879k) + 1.9M-quad mesh built in
  the browser in 350 ms
- no deck errors, no page errors; esm.sh pins (deck 9.3.10 incl. mesh-layers) resolved
  to one core
- the raster renders on its native grid: coastline pixel-crisp, no warp artefacts
- pick at a map point: "px (561, 1320) · Harlan County, KY · via H3 label (res 7) ·
  cell 872a9b198ffffff · 35.0°C"; a click at sea reports "not land" through the snap
- sustained heat, +3 °C level, 2026-07-02 16Z: 249,519 pixels above; res 6 majority
  rule 245,518 member pixels, 31,924 outline edges; per-pixel rule 249,519 and 47,044
  edges. The cell rule removes single-pixel speckle and smooths the edge to ~4-pixel
  granularity; that is the whole visible effect of H3 on the picture.
- Carto basemap tiles now carry an "API KEY REQUIRED" watermark on the labels layer
  (also affects the x-sql-marimo notebooks); swap the basemap or add a key.

## Steps to fold into the full notebook

1. **Copy the chassis** from `xsql-hrrr-heat-domes.py`: constants cell, DuckDB
   connect, PMTiles county reader (cache-first), `MirrorStore`, store cell, window
   cell, HOLD memo. Keep READ_RAIN / READ_WIND flags and the wx packing.
2. **Replace the fold.** Keep the xarray-sql read and the land block predicate. Drop
   the `GROUP BY`: `SELECT t, y, x, temperature_2m, relative_humidity_2m [, mm, ws]
   FROM cube JOIN pix2h USING (y, x) WHERE ... ` returns pixel-hours. No DataFusion
   pool knobs (there is no aggregate). Pivot to (F, N) with searchsorted on (y, x)
   as the frames cell already does on cells. Decide whether the res 7 label is
   still emitted by the h3ronpy UDF in the SQL (x-sql-marimo's rule) or computed once
   in Python from the store's lat/lon (this repo's premise; the prototype does this).
   Either works; the label never changes, so once in Python is the cheaper one and
   the `pix2h` table can carry `hex7` and `hex6` columns.
3. **Replace the widget** with `RasterFilm`, carrying over from `HexFilm`: the window
   loader (the one trait back), the click chart, the field switch, the rain/wind
   sliders, the HUD collapse, fullscreen, keyboard, the contour levels.
4. **Dome table** (DuckDB): input becomes the res 6 parents of member pixels (per
   frame, per level, under the chosen rule); `h3_cells_to_multi_polygon_wkb`, area,
   centroid, track unchanged. Also compute the pixel-based area (count x 9 km2) next
   to the dissolved area so the table shows what the cell rule adds.
5. **Bytes.** A week at CONUS is 148 MB per field; the notebook needs either a BOX
   default (the East dome region) or a cap on hours, or kernel-side accumulation
   with frames streamed. Start with BOX.
6. **Cross-grid join (stretch).** MRMS hourly precip (source.coop, 1 km lat/lon):
   label it at res 7, parent to res 6, join to the HRRR labels by id, use it as the
   rain flush instead of HRRR precip. First demonstration of two native grids joined
   without regridding.

## Decisions taken in the prototype (revisit if wanted)

- Pick tries H3 first, LCC snap second, and reports which one hit. The snap is the
  raster's own inverse; H3 is the concept.
- Dome membership default is "res 6 majority".
- Texture alpha for sustained heat scales with value (dark = faint), same as heat-hex.
- Basemap Carto dark, labels on top at 0.6, as the other notebooks.
- Ramps: index diverging blue/yellow/orange (no red leg), load inferno. Protan-safe.

## Open questions carried forward

- Whether `SimpleMeshLayer` texture re-upload per frame is fast enough at 12 fps for
  a 1799x1059 RGBA canvas; the flight measures boot only. If not, keep the texture
  static and pass the frame matrix as a second texture with the LUT in a shader
  (a small custom layer).
- Whether to ship the full-grid frames (simpler texture paint, 2.2x the bytes) or
  keep the land-only matrix (the prototype's choice).
- The label layer in the store: write `hex7` as a (y, x) uint64 array next to the
  data (the docs/02 "stored label layer") and read it back instead of recomputing,
  to test the zarr-conventions/dggs vocabulary on a real store.

## Flight results, the notebook (2026-08-30, later the same day)

`hrrr-heat-domes.py` at the repo root; `fly.py` flies it headless (playwright,
Chrome, software GL: no GPU on the runner, so render times are an upper bound).

What was done against the steps above:

1. Chassis copied: constants, DuckDB, PMTiles counties (cache-first), `MirrorStore`,
   store cell, window cell, HOLD memo, READ_RAIN / READ_WIND and the wx packing.
2. The fold is gone, and so is xarray-sql. A pixel-hour `SELECT` with no aggregate is
   only a pivot, and it would materialise ~148M rows x 32 B = 4.7 GB of Arrow for a
   CONUS week before the pivot. The read is block-wise off the Zarr: for each variable
   and each 45x45 store block holding a land pixel of BOX, slice the window's hours and
   write them into the (F, N) land-pixel matrix (8 threads, through the mirror). The
   res 7 label is computed once in Python from the store's lat/lon (this repo's
   premise); `pix2h` does not exist any more.
3. `RasterFilm` carries everything from `HexFilm`: window loader, click chart (with the
   contour levels as dashed lines on the sustained-heat chart), field switch, rain /
   wind sliders, HUD collapse, fullscreen, keyboard, day ticks, fps. Boundaries at
   every CONTOURS level on pixel edges, under the rule select (pixel / any / majority /
   all). A frame is one pass: per pixel a level code (how many contours it clears), the
   rule per res 6 parent, then every pixel edge whose two sides differ in code is a
   segment for the levels in between.
4. Dome table: `DOME_RULE` applied per parent in numpy, the member cells dissolved per
   (frame, level) with `h3_cells_to_multi_polygon_wkb`, area (Albers) and centroid;
   next to the largest blob the table shows every blob's area and the member pixel
   count x 9 km2 at that hour.
5. Bytes: BOX is None by default (the full grid, 879,370 land px, 148 MB per field for
   the week, boot 158 s); (-95, 29, -69, 46), the eastern dome's region, is a 745x834
   grid window, 339,323 land px, 57 MB per field, boot 36 s. The HUD's window note shows
   the MB per field for the dates picked and says "seconds (chunk on the disk
   mirror)" when every store chunk the window touches is mirrored for all variables.
6. Cross-grid join (MRMS): not started.

Measured, default window (Jun 29 to Jul 5, 2026, mirrored chunk 47, five variables),
first with the eastern-dome BOX, then the full grid:

- Kernel, all cells as a script: 11.5 s. Store open 2.6 s, counties 0.0 s (parquet
  cache), geometry 1.4 s (mesh 0.1 s), read 2.9 s (219 blocks x 5 variables, 2,190
  ranges from disk, 0 fetched), dome table 2.9 s (10.1M cell-hours over 1 C, 6,363
  blobs >= 500 km2).
- Browser: boot to first frame 36 s (57 MB frames + 57 MB wx + 5 MB mesh + 6 MB
  indices over the bridge); label index + 621,330-quad mesh 80 ms; texture paint 5 ms
  per frame; dome pass (level codes, rule, edges) 8-13 ms per frame.
- Pick: "px (377, 446) · Craig County, VA · via H3 label (res 7) · cell
  872a8e76dffffff · 25.0 C".
- 2026-07-02 16Z, +3 C: majority 196,730 member px; pixel 200,148; all 182,723.
  Boundary edges over all four levels: 53,866 / 75,972 / 53,424. Largest >= 3 C blob
  peaks at 2.40M km2 (frame 95, Jul 3 23Z, centred 37.0N 85.1W); >= 5 C at 1.71M km2.

Two findings that change the prototype's story:

- deck's `SimpleMeshLayer` re-uploads `texture` only when the prop is a different
  object; `updateTriggers` do not reach it. The prototype painted one canvas in place,
  so its "sustained heat" screenshot still shows the heat-index ramp. Two canvases,
  alternated per paint, fix it.
- A `PathLayer` of ~54k pixel-edge segments costs ~1.2 s per frame in CPU tesselation.
  As four `LineLayer`s (instanced, binary attributes) a step is ~820 ms in-page in
  software GL against ~270 ms with boundaries off; as one `LineLayer` with per-edge
  colour and width, 817 ms: identical, so the ~550 ms is the software rasteriser
  filling 54k-76k wide segments, not deck's layer machinery (the dome pass itself is
  10 ms). The raster alone at ~270 ms per step is the 621k-quad mesh in software GL.
  Neither number says anything about a real GPU; that measurement is Stephen's
  browser's to make (the ruler shows paint / render / dome ms live).
- Full grid (BOX = None): boot 158 s, mesh 305 ms, paint 10 ms, dome pass 16-21 ms,
  77k-107k edges; software GL step ~0.9 s raster alone, ~2.0-2.3 s with boundaries.
  Pick "px (578, 964) · Labette County, KS · via H3 label (res 7)". Jul 2 16Z, +3 C:
  majority 233,138 px, pixel 237,613, all 213,778.
- The raster paints at alpha 255 (the prototype's value-scaled alpha for sustained
  heat is gone); an opacity slider in the HUD sets the mesh layer's opacity prop.
- Carto's tiles (dark_nolabels included) watermark "API KEY REQUIRED"; OpenStreetMap
  refuses headless apps (403). Esri's Dark Gray Canvas (base + reference) is keyless
  for light use and is the basemap now.
