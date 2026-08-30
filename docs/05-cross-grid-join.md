# Cross-grid join: HRRR x CONUS404 through the res 7 label

Written 2026-08-30. The first join of two native grids through the label layer, with
the cost of "nearest in the hierarchy" measured against "nearest in metres". All data
from source.coop.

## Why this pair

HRRR (3 km, NCEP Lambert, sphere 6371.229 km) and CONUS404 (4 km, WRF Lambert, sphere
6370 km, lat_1 30 / lat_2 50 / lat_0 39.1 / lon_0 -97.9) are two projected grids of
similar pixel size. Both are unique at H3 res 7 (cell 5.16 km2 against pixels of 9 and
16 km2; measured: 1,905,141 unique labels for HRRR, 1,387,505 for CONUS404, zero
collisions). So a res 7 cell holds at most one pixel of each, and the join is an
equi-join on the label with no target grid, no weights, no resample.

CarbonPlan's CONUS404 Fosberg store gives the hourly index 1979-2022 (chunked
`(376945, 10, 10)`, a time series per block, unreadable as a map) and its p95 / p99
over time as single-chunk `(1015, 1367)` arrays. The percentiles are what an HRRR
Fosberg window needs: "is this unusual here."

## Sources

- HRRR lat/lon: `dynamical/noaa-hrrr-analysis/v0.2.0.zarr` on source.coop (zarr v3
  mirror of the icechunk store; `latitude` / `longitude` are whole-array chunks, 7.6 MB
  each). The mirror's coordinates reproduce the cached res 7 labels exactly.
- CONUS404: `carbonplan/carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/
  fosberg-fire-weather-index-{p99,p95}.icechunk`. The `crs` array carries no
  attributes; the WRF CONUS404 Lambert was applied to the store's `x` / `y` (4000 m,
  row 0 south) and checked against the store's own 2D `lat` / `lon` at seven pixels:
  worst 1.3 m.

## The join (join/prep_conus404.py, 8 s wall)

1. Label every CONUS404 pixel at res 7 from its projected centre (h3ronpy
   `coordinates_to_cells`), sort, and look HRRR labels up in it (searchsorted).
2. **Direct**: the HRRR pixel's own cell holds a CONUS404 centre. 287,367 of 879,370
   land pixels (32.7%). That share is the cell area over the CONUS404 pixel area
   (5.16 / 16 = 0.32) and barely moves with latitude (0.348 at 25-30 N, 0.313 at
   45-50 N).
3. **Ring**: for the rest, `gridDisk(label, 1)` offers up to six neighbour cells; of
   those holding a CONUS404 centre, the nearest in metres is kept. 592,003 pixels.
   No pixel was left without a partner.
4. **Check**: the true nearest CONUS404 pixel by KD-tree in CONUS404 metres.

| | pixels | equals true nearest | centre distance median / p95 / max |
|---|---|---|---|
| direct | 287,367 | 96.0% | 1,155 / 2,120 / 2,907 m |
| ring | 592,003 | 97.2% | 1,766 / 2,519 / 3,216 m |
| all | 879,370 | 96.8% | |

Extra distance paid by the hierarchy over the metric nearest: median 0, p95 0, max
1,678 m (3.2% of pixels pay anything). Sub-pixel on a 4 km field. A direct hit is not
always the nearest: a res 7 cell is ~1.3 km across, so a CONUS404 centre in the same
cell can sit 2.9 km away while the neighbouring centre is 1.1 km away.

Reverse: 628,075 of 1,387,505 CONUS404 pixels have an HRRR pixel in their cell (the
rest are ocean, Canada, Mexico, or cells where the HRRR centre fell in a neighbour).

`join/cache/p99_on_hrrr.npy` is the CONUS404 p99 drawn on HRRR pixels; the picture
(`join/cache/check.png`) shows terrain structure and no seam or stripe from the
direct / ring alternation.

## What this says about the label layer

- Two projected grids of similar pitch join at the label res with 1:1 semantics per
  cell, but only a third of cells hold both centres; the ring carries the rest. "1:1 at
  the label" means "at most one of each per cell", not "one of each in every cell".
- The hierarchy alone cannot choose among ring candidates; metres did (the pixel
  centre in the other grid's CRS). A pure-H3 alternative is to label at a finer res
  (res 8 or 9) and take the candidate whose fine label is closest in `gridDistance`,
  which is nearest-in-cells rather than nearest-in-metres. Not tried.
- The check against the KD-tree is the honest cost of the idea: 3% of pixels get a
  neighbour that is not the metric nearest, never more than 1.7 km off.
- The join needs no geometry at runtime once both stores carry their label array
  (docs/02 "stored label layer"): two uint64 arrays and an equi-join. The CONUS404 side
  needed geometry here only because its `crs` is empty and the 2D lat/lon is chunked
  `(10, 10)`.

## Next

- HRRR Fosberg for a window (T, RH, wind from the source.coop mirror), compared per
  pixel against the joined p99 / p95: the "unusual here" column for the window table.
- Draw `p99_on_hrrr` on the raster mesh (a CONUS404 field on HRRR pixels) and HRRR
  FFWI beside it; the res 6 rule and dome table over "FFWI above local p99".
- Burn probability (270 m Albers, same host) as the many-to-one case at the parent.
- ASOS stations as points into the same key space.

## Plan: the notebook (agreed 2026-08-30, not started)

Start with the two rasters, each on its own grid, and the join shown separately,
the way the heat-domes notebook shows one thing per view. Extrusion is parked (too
heavy for the machine at 879k pixels). The ratio-to-p99 view (view 3 above, FFWI
against the pixel's own climatological p99) is not ruled out; it was not clear from
the description alone, so it waits until there is something on screen to judge it
against. Stephen's working lean, not a rule: maps tend to make sense when they are
most honest to the data.

New notebook at the repo root, `hrrr-conus404-join.py`, same chassis as
`hrrr-heat-domes.py` (marimo, one anywidget, deck.gl 9.3.10 pins, Esri dark basemap),
reading `join/cache/` from `join/prep_conus404.py`.

1. **Geometry, twice.** Lift the corner builder out of the heat-domes geometry cell
   (line 1450: store x/y + CRS -> pyproj -> web-mercator world coords, `(ny+1)(nx+1)`
   corners) into a function of `(x, y, crs)`. Call it for HRRR (NCEP Lambert, 1059 x
   1799) and for CONUS404 (WRF Lambert from `prep_conus404.py`, 1015 x 1367). Two
   meshes, two `SimpleMeshLayer`s, each with its own texture. CONUS404 corners are
   ~11 MB more over the bridge.
2. **View A: CONUS404 p99 on CONUS404 pixels.** The field as published, on its 4 km
   grid. Viridis (or another protan-safe ramp), fixed scale shared with view B.
3. **View B: CONUS404 p99 on HRRR pixels.** `p99_on_hrrr.npy`, the same numbers
   carried by the join onto the 3 km grid. Same ramp and scale as A so a toggle
   between A and B shows only the footprint changing. Land mask from the HRRR side.
4. **View C: the join itself, on HRRR pixels.** Two fields from `join.parquet`:
   `how` (direct / ring, two-class, luminance not hue) and `dist_m - nn_dist_m`
   (extra metres over the metric nearest, 0 for 96.8% of pixels). A view select in
   the HUD like the heat-domes field buttons.
5. **Pick.** Click -> `latLngToCell(res 7)` -> HRRR pixel (as now), then the HUD
   shows: HRRR (i, j), its cell id, the partner CONUS404 (i', j'), how (direct /
   ring), centre distance, the metric-nearest pixel if different. On the map, drawn
   only for the picked pixel: the HRRR pixel outline, the partner CONUS404 pixel
   outline (a 4 km square on its own grid), and the res 7 cell outline (the one
   hexagon ever drawn, on demand, as the key). Ring cells as a faint outline when the
   match was via ring.
6. **Ruler.** Pixel counts for both grids, direct / ring counts, agreement %, build
   and paint ms, as the heat-domes ruler does.
7. **Flight.** `fly.py` takes a notebook path already; add the selectors for the
   view buttons and the pick readout, screenshots to `shots/`.

Not in this plan: any HRRR weather read (the notebook is static fields only), the
dome table, the accumulator. Those come back when the HRRR Fosberg window is built
on top of this.
