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
