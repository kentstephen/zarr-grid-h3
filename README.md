# zarr-grid-h3

H3 traversal and hierarchy functions over data that stays on its native Zarr grid.
The raster is what gets drawn; H3 is the transformation underneath it.

The seed: give every pixel of a Zarr grid one H3 cell (the cell containing the pixel
centre, at a resolution fine enough that no two pixels share one). That label never
moves the data. It gives each pixel a global, hierarchical identity that H3's
`cellToParent`, `gridDisk`, `gridPathCells` and friends can operate
on, and every result maps back to pixels. No resampling, no hexagons on screen.

## The notebook (hrrr-heat-domes.py)

The full heat-domes notebook on the native grid: the chassis of
`x-sql-marimo/xsql-hrrr-heat-domes.py` (constants, DuckDB, the PMTiles county reader,
the disk mirror of the store's shards, the window loader, the accumulator, the DuckDB
dome table) with the prototype's raster widget in place of the hexagon film. There is
no fold: the read is block-wise off the Zarr into an (hours x land pixels) matrix, the
heat index is per pixel, and H3 enters as the label per pixel (pick, county), the res 6
parent (dome membership by rule: per pixel / any / majority / all), and the cell set the
dome table dissolves (next to the plain pixel count x 9 km2). Boundaries are drawn on
pixel edges at the contour levels, every frame.

```bash
uv sync
uv run marimo edit hrrr-heat-domes.py   # eastern-dome week from the disk mirror (needs the x-sql-marimo caches)
uv run python fly.py                    # headless flight, screenshots in shots/
```

Measured 2026-08-30 (headless Chrome, software GL, the full grid): 879,370 land pixels,
210,724 res 6 parents, 168 hourly frames = 148 MB per field; kernel end to end ~12 s
from the mirror (read for five variables, dome table ~3 s each), boot to first frame
158 s (two 148 MB fields over the anywidget bridge), label index + 1.9M-quad mesh
305 ms, texture paint 10 ms per frame, dome pass ~20 ms. Pick "px (578, 964), Labette
County, KS, via H3 label (res 7)". At 2026-07-02 16Z the majority rule has 233,138
member pixels at +3 C and 77,048 boundary edges over all levels; the pixel rule
237,613 and 107,176; the all rule 213,778 and 76,258. `BOX` in the constants cell
cuts the grid to a region (the eastern dome box is 339k pixels, 57 MB per field, boot
36 s). Two findings on the way: deck's `SimpleMeshLayer` re-uploads its texture only
when the prop is a new object (the prototype's field switch never reached the GPU; the
texture is double-buffered now), and a `PathLayer` of 54k pixel edges costs ~1.2 s a
frame to tesselate (one instanced `LineLayer` now). The raster is drawn at full
opacity, with an opacity slider in the HUD.

## Prototype (proto/)

HRRR 2 m heat index (dynamical.org's Zarr, 3 km Lambert grid) drawn as one textured
quad per pixel on a deck.gl mesh built from the store's own GeoTransform, with an H3
res 7 label per pixel (1,905,141 labels for 1,905,141 pixels: a relabel, one cell per
pixel) and its res 6 parent underneath.

- pick: click to `latLngToCell(res 7)` to pixel, with the grid's own LCC inverse as the
  snap when the click lands in an empty cell
- domes: sustained-heat threshold per pixel, membership decided per pixel or per res 6
  parent (any / majority / all), outline drawn on pixel edges
- county name from a res 6 polyfill join

```bash
uv sync
uv run python proto/prep.py          # labels, mesh corners, land mask, a 48 h slab (needs the x-sql-marimo caches)
uv run marimo edit proto/raster_mesh.py
uv run python proto/fly.py           # headless flight, screenshots in proto/shots/
```

Measured 2026-08-30 (headless Chrome): boot to first frame 25 s (42 MB frames + 15 MB
mesh over the anywidget bridge), label index + 1.9M-quad mesh 350 ms in the browser,
pick "px (561, 1320), Harlan County, KY, via H3 label (res 7)". At +3 °C sustained
heat on 2026-07-02 16Z: 249,519 pixels above; majority rule 245,518 member pixels and
31,924 outline edges; pixel rule 249,519 and 47,044 edges.

## Docs

- `docs/01-landscape.md`: what exists (h3ronpy, raster2dggs, xdggs, zarr-conventions/dggs, GeoZarr) and the gap
- `docs/02-label-layer-approach.md`: the idea, operations it unlocks, where it strains, directions
- `docs/03-resolution-choice.md`: pixel size to H3 resolution tables
- `docs/04-heat-domes-plan.md`: the plan for folding the prototype into the notebook, with the flight results

Data: NOAA HRRR via dynamical.org (CC-BY 4.0); counties from Overture Maps.
