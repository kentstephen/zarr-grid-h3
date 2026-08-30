# The label-layer approach: H3 identities over a native Zarr grid

## The seed idea, restated
Polyfill the Zarr grid at a high H3 resolution and, for each pixel, keep the one H3 cell whose centre falls inside that pixel. The pixel keeps its value, its position in the array, its chunk, and its CRS. What it gains is a global, hierarchical, hexagonal identity.

Two simplifications fall out immediately:

1. "Polyfill then keep the centre cell" collapses to `latLngToCell(pixel_center_lat, pixel_center_lon, res)`. On the gnomonic face plane H3 cells are the Voronoi regions of their centres, so the cell containing the pixel centre is the cell whose centre is nearest to it. Polyfilling the whole extent at res 13 or 14 would generate tens of cells per pixel just to throw them away; converting an array of pixel-centre coordinates is a single vectorized call (`h3ronpy.coordinates_to_cells`, or h3-py v4 `latlng_to_cell` over numpy).
2. The mapping needs to be computed from pixel centres in EPSG:4326, but the data does not need to be reprojected. For projected grids, `pyproj` transforms the centre coordinates; the array is untouched.

Working vocabulary used in the rest of this doc:

- **Label resolution (r_L)**: fine enough that every pixel gets a distinct cell (cell area < pixel area everywhere in the grid, with margin for the ~2x global area spread). Each pixel has exactly one label; some cells at r_L have no pixel.
- **Traversal resolution (r_T)**: coarser, `cellToParent(label, r_T)`. Many pixels share a cell here. This is where `gridDisk`, `gridRing`, `gridPathCells`, `gridDistance` become meaningful, because at r_L a cell's neighbours mostly sit inside the same pixel.
- **Chunk resolution (r_C)**: coarser still. The set of r_C cells touched by each Zarr chunk forms a chunk index.

## What it would look like in a Zarr store

```
store.zarr/
  data            (time, y, x)   native values, native chunking
  y, x            coordinates (or spatial:transform per GeoZarr)
  h3_label        (y, x) uint64, same chunking as data, or virtual
  attrs:
    dggs: { name: "h3", refinement_level: 13, role: "label",
            coordinate: "h3_label", spatial_dimension: ["y","x"] }
    h3_chunk_index: { refinement_level: 6, cells_per_chunk: <array or JSON> }
```

`h3_label` can be materialised (8 bytes/pixel before compression; neighbouring labels share long prefixes so shuffle+zstd should compress well) or virtual (computed per chunk on read from the affine transform; cost is one `latLngToCell` per pixel, cheap). The zarr-conventions/dggs vocabulary (`name`, `refinement_level`, `coordinate`, `spatial_dimension`) fits with the addition of a `role` or similar to distinguish "this array IS on the DGGS" from "this array is labelled BY the DGGS".

## Operations this unlocks, and how each one maps back to pixels

### A. Reverse lookup: cell to pixel
For any r_L cell c, `cellToLatLng(c)` then inverse affine gives a pixel (row, col). If c is not a label of that pixel (because c is one of the empty cells between labels), snap: take that pixel's own label. So `pixel_of(cell) = inverse_affine(cellToLatLng(cell))` is total, and `label_of(pixel_of(cell)) == cell` exactly when cell is a label. No hash table needed, unlike the xoak proposal.

### B. Hex windows on native pixels (focal operations)
`gridDisk(cellToParent(label, r_T), k)` gives a near-isotropic, near-equal-area window. Select pixels whose parent-at-r_T is in that set (vectorized membership test on the parent array) and reduce. On a global lat/lon grid this is a real difference from a 3x3 or 5x5 pixel window: the pixel window shrinks in ground area toward the poles, the hex window does not.

### C. Zonal aggregation without resampling
`groupby(cellToParent(label, r))` over the native array yields per-hex statistics at any resolution r <= r_L in one pass. This is what raster2dggs "point sampling" computes, except here it is a view over the original data rather than a new dataset, and any r can be chosen after the fact from the same label layer. Weighting is by pixel count, so on lat/lon grids a cos(lat) or true pixel-area weight has to be carried alongside for area-correct sums.

### D. Cross-grid joins by identity
Two rasters on different grids or CRSs (say a 10 m UTM Sentinel-2 tile and a 0.25 degree ERA5 field) each get a label layer. `cellToParent` to a common resolution gives a shared key; no reprojection, no warping. Same key joins vector points (`latLngToCell`) and any H3-native dataset from CARTO, Fused, Foursquare, etc.

### E. Paths and distances in hex metric
`gridPathCells(a, b)` at r_T, then pixels-per-cell, gives a raster corridor with a hex-metric length `gridDistance(a, b)`. Useful for line-of-sight, cost-path seeding, sampling transects. Subject to H3's pentagon and long-distance failure modes; for global paths, chain through intermediate cells.

### F. Chunk selection driven by H3
Precompute, per Zarr chunk, the set of r_C cells whose centres fall in the chunk (or overlapping cells, using `polygonToCellsExperimental` with `overlapping` mode on the chunk footprint). Invert to cell -> chunks. Then:
- A query expressed as H3 cells (a polyfilled AOI, a gridDisk around a point, a list of hexes from an analytics platform) resolves to chunk ids without touching the affine at all.
- `gridDisk(chunk_cells, 1)` gives the chunks to prefetch for a window that straddles chunk boundaries.
- `compactCells` on the full label set gives a compact coverage footprint for sparse or irregular datasets (swaths, tiles with nodata), reusable as a STAC-style spatial descriptor.

### G. Multiscale alignment
GeoZarr multiscales levels each get their own label layer at their own r_L; because labels are hierarchical, `cellToParent` links levels. Overviews and H3 hierarchy become the same tree, offset by a constant number of levels (H3 aperture 7 vs raster 2x downsampling, so the offset is not integer; that mismatch is itself worth studying).

## Where it strains

- **Resolution budget.** A 10 m pixel is 100 m2; r_13 average is 43.9 m2 but max at r_13 is roughly 1.99x the min, so r_13 may be marginal at some longitudes/faces and r_14 (6.27 m2) is safe. Every extra level costs nothing in storage per pixel (still uint64) but changes how many empty cells sit between labels.
- **Two-way mapping is not a bijection.** Label -> pixel is exact. Cell -> pixel needs the snap in (A). Parent cell -> pixels is one-to-many with a latitude-dependent count on lat/lon grids.
- **Anisotropic pixels.** Lat/lon grids at high latitude have tall thin pixels on the ground. Labels are still unique if r_L is chosen against the smaller pixel dimension; traversal windows then contain uneven pixel counts. That is the feature (equal-area windows) and the cost (uneven sample sizes).
- **H3 distortion and pentagons.** Cell shapes vary across icosahedron faces; 12 pentagons per resolution; `gridDistance`, `gridPathCells`, `cellToLocalIj` all have documented failure regions. Ocean-centred pentagons keep this rare for land data but not for global climate fields.
- **Storage.** A materialised label layer on a (time, y, x) cube is only (y, x), so the overhead is one 2D uint64 array per grid, not per time step. For a 40k x 40k grid that is 12.8 GB uncompressed; the virtual option avoids it entirely.
- **Semantics.** Users of H3 datasets expect a value per cell. A label layer is a different contract (a value per pixel, an id per pixel). Metadata must say which one it is.

## Directions this could go (not ranked)

1. **Virtual label index in xarray.** A `rasterix`-style `RasterIndex` subclass that materialises labels lazily and supports `.sel(h3=cells)`, `.h3.disk(cell, k)`, `.h3.parent(r)` groupby. Nothing stored.
2. **Stored label layer + Zarr convention proposal.** Materialise `h3_label`, write it up as a `role: label` extension to zarr-conventions/dggs, contribute an H3 section to that convention.
3. **Chunk-level H3 index only.** Skip per-pixel labels; index chunks at r_C. Smallest footprint, biggest win for cloud reads keyed by hexes. Could live in consolidated metadata.
4. **Traversal-driven access API.** A small library where the primitives are `cells -> pixel index sets -> zarr orthogonal indexing`, with gridDisk/gridPath/polyfill as the query language.
5. **Equal-area focal statistics for global grids.** Narrow but concrete: hex-window rolling stats on ERA5/CMIP lat/lon cubes, compared against pixel windows for polar bias.
6. **Comparison study against conversion tools.** Same inputs through h3ronpy, raster2dggs and the label layer; measure information loss, storage, query latency, and how each handles joins across grids.
7. **Multiscale / aperture mismatch study.** How raster overview levels (2x) and H3 levels (7x area) align, and whether `cellToCenterChild` chains give a usable pyramid over native overviews.
8. **Beyond H3.** The same label construction works for HEALPix (which zarr-conventions/dggs already documents) and A5/S2. HEALPix labels would align the project with the DestinE / EOPF direction; H3 keeps the traversal richness.

## Open questions worth holding open
- Is there a closed form for the "safe" r_L from a pixel size that accounts for the max cell area at that resolution and the face the grid sits on?
- Should `h3_label` be indexed by pixel centre (current idea) or by pixel corner, and does it matter for anything downstream?
- What does a cell-to-pixel snap do at pixel boundaries between two chunks, and does the chunk index need overlapping cells to be safe?
- Does compaction of the label set give a useful coverage descriptor for swath-shaped data, or do the empty inter-label cells make compaction useless at r_L (it probably needs to be done at r_T)?
- Which H3 traversal functions have a meaningful pixel-space analogue and which do not (local IJ probably does, as a way to get axial hex coordinates for kernels)?
