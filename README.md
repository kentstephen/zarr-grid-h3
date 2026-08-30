# zarr-grid-h3

H3 traversal and hierarchy functions over data that stays on its native Zarr grid.
The raster is what gets drawn; H3 is the transformation underneath it.

The seed: give every pixel of a Zarr grid one H3 cell (the cell containing the pixel
centre, at a resolution fine enough that no two pixels share one). That label never
moves the data. It gives each pixel a global, hierarchical identity that H3's
`cellToParent`, `gridDisk`, `gridPathCells`, `compactCells` and friends can operate
on, and every result maps back to pixels. No resampling, no hexagons on screen.

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
- `docs/04-heat-domes-plan.md`: the plan for folding the prototype into a full heat-domes notebook

Data: NOAA HRRR via dynamical.org (CC-BY 4.0); counties from Overture Maps.
