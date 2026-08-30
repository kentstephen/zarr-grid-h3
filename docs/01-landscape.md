# Landscape: H3, DGGS, and Zarr/COG grids (as of 2026-08-30)

Survey of what exists today for combining H3 (or any DGGS) with regular raster grids stored in Zarr or COG. The short version: every existing tool either converts the raster into a DGGS dataset (a resampling step) or assumes the data already lives on a DGGS. None of them treat H3 as a side-index over a raster that stays on its native grid.

## 1. Raster-to-H3 converters (resampling approaches)

### h3ronpy / rasterh3 (nmandery)
- Rust core (h3o), Python bindings, Arrow-based.
- Algorithm (from `rasterh3/src/array.rs`): the array is split into rectangles of non-nodata pixels; each rectangle's geographic bbox is polyfilled in `ContainsCentroid` mode; for every resulting H3 cell the cell centroid is inverse-transformed to a pixel and that single pixel value is assigned. Quote from the docs: "this raster conversion process only takes the raster value under the centroid of the cell into account."
- `nearest_h3_resolution(shape, transform, search_mode=...)` picks the resolution; `smaller_than_pixel` is the documented mode, so by default there are several cells per pixel and `compact=True` merges equal-valued children back into parents.
- Also exposes vectorized cell ops on arrays: `grid_disk`, `grid_disk_distances`, `grid_ring_distances`, `change_resolution`, `compact`, `uncompact`, `cells_to_localij`, `cells_parse`, with pandas and polars adapters.
- Direction of mapping: iterate cells, sample pixels. This is the inverse of the label-layer idea.

Sources: https://h3ronpy.readthedocs.io/en/stable/usage/raster.html , https://github.com/nmandery/rasterh3 , https://h3ronpy.readthedocs.io/en/stable/api/index.html

### raster2dggs (Manaaki Whenua)
- CLI, parallel over GDAL windows (one worker per window), Parquet/GeoParquet output hive-partitioned by a parent resolution (default `resolution - 6`).
- Three pixel-to-cell methods: point sampling (pixel centre to containing cell, then aggregate with mean/sum/mode/... when multiple pixels share a cell), overlay (exact pixel-cell intersection areas via exactextract, area-weighted), windowed sampling (sample raster at each cell centre with nearest/bilinear/cubic/lanczos kernels).
- Multi-DGGS: H3, S2, A5, HEALPix, rHEALPix, geohash and more.
- Direction of mapping: iterate pixels, assign to cell, aggregate. Data is moved onto the DGGS.

Source: https://github.com/manaakiwhenua/raster2dggs

### Smaller examples
- `raster-analysis-using-h3` (kshitijrajsharma): reproject to EPSG:4326, COG, h3ronpy with a resolution chosen from pixel size, then PostgreSQL joins on the H3 id. Typical of the "convert then join" pattern.
- `opengeos/geolibre-rust` issue 26 proposes a `raster_to_h3` aggregation function with the same shape.

## 2. Data already on a DGGS

### xdggs (xarray-contrib)
- Xarray accessor `.dggs`; a 1D `cell_ids` coordinate backed by a `DGGSIndex` (subclasses `H3Index`, `HealpixIndex`). One grid and one resolution per Dataset. Selection/alignment by cell id, cell centers and boundaries, lat/lon selection, hierarchy ops.
- Stores naturally in Zarr because the grid is just a coordinate plus attrs. Dask for scale.
- Design doc has no chunking or spatial-index story; spatial queries are deferred to companions like xvec. Mixed resolutions in one coordinate are unsupported.
- Ecosystem momentum is HEALPix-heavy (DestinE climate digital twin, EOPF-DGGS / GRID4EARTH, healpix-geo).

Sources: https://github.com/xarray-contrib/xdggs/blob/main/design_doc.md , https://lps25.esa.int/lps25-presentations/presentations/1170/_1170.pdf , https://eopf-dggs.github.io/

### zarr-conventions/dggs
- A `dggs` object in Zarr group/array attributes: `name`, `refinement_level` (nullable), `spatial_dimension`, `coordinate` (array holding cell ids), `compression` in `none | compacted | ranges`, optional `ellipsoid`.
- Maturity: Pilot. HEALPix is the documented grid; H3 is not yet written up but the schema leaves room for grid-specific parameters.
- Relevant here because it already defines how a set of cell ids (including compacted sets and contiguous ranges) is stored as a Zarr array. A label layer could reuse the vocabulary.

Source: https://github.com/zarr-conventions/dggs

## 3. Zarr raster georeferencing and chunk access (no DGGS involved)

### GeoZarr and the zarr-conventions family
- `spatial:transform` (affine) is authoritative per resolution level; `proj` convention carries CRS; `multiscales` describes overviews. v1 targeted for end of 2026.
- Chunk traversal in every implementation (GDAL Zarr driver, deck.gl-raster ZarrLayer, xarray/dask) is bbox to affine to chunk index arithmetic. There is no DGGS-keyed chunk index anywhere.
- `rasterix` (xarray-contrib) provides a `RasterIndex` built on the affine transform, which is the natural place to hang a pixel-to-H3 mapping in xarray.

Sources: https://github.com/zarr-conventions/proj , https://geozarr.org/conventions.html , https://gdal.org/en/stable/drivers/raster/zarr.html , https://developmentseed.org/deck.gl-raster/blog/initial-geozarr/ , https://github.com/xarray-contrib/rasterix

### xoak issue 16 (closest prior art to the label-layer idea)
- Proposal: compute an H3 id at a fixed resolution for every lat/lon grid point, hash id to array positions, answer nearest-neighbour queries by `kRing` around the query cell and brute force inside the candidates. Concerns raised: resolution sensitivity, hash-table size, lack of vectorized kRing at the time (h3ronpy now has it). Still open, never implemented.
- This is exactly "H3 as an index over a native grid", but only for point lookup, not for traversal or chunk selection.

Source: https://github.com/xarray-contrib/xoak/issues/16

## 4. H3 primitives that matter for this project

- Polyfill modes (v4 `polygonToCellsExperimental`): `center` (cell centre in polygon; fastest), `full` (cell entirely inside), `overlapping` (any intersection). For a rectangle the size of one pixel, `center` mode at a fine resolution returns the cells whose centres sit inside the pixel; the one containing the pixel centre is `latLngToCell(pixel_center, res)` directly.
- Traversal: `gridDisk`, `gridDiskDistances`, `gridRing`, `gridPathCells`, `gridDistance`, `cellToLocalIj` / `localIjToCell`. Caveats: distance and path functions fail across pentagons and for far-apart cells; local IJ is valid only in the origin's base cell or a neighbour and is not stable across H3 versions.
- Hierarchy: `cellToParent`, `cellToChildren`, `cellToCenterChild`, `compactCells`, `uncompactCells`.
- Cell size (average, v4 table):

| res | avg edge (m) | avg area (m2) |
|----:|-------------:|--------------:|
| 8   | 531.4        | 737,328       |
| 9   | 200.8        | 105,333       |
| 10  | 75.9         | 15,048        |
| 11  | 28.7         | 2,150         |
| 12  | 10.8         | 307           |
| 13  | 4.09         | 43.9          |
| 14  | 1.55         | 6.27          |
| 15  | 0.58         | 0.895         |

Max/min cell area ratio at a given resolution is about 1.99 across the globe, so a resolution that is "smaller than pixel" at the equator is still smaller than pixel everywhere only if chosen with margin.

Sources: https://h3geo.org/docs/api/traversal/ , https://github.com/uber/h3/blob/master/dev-docs/RFCs/v4.0.0/polyfill-modes-rfc.md , https://h3geo.org/docs/core-library/restable/

## 5. Why hexagonal neighbourhoods are attractive on rasters at all
- All six neighbours share an edge and are equidistant; square grids mix 4 edge neighbours and 4 corner neighbours (anisotropy, the "Manhattan effect").
- Flow direction, dispersal, and erosion models preserve direction better under hex tiling across scales.
- Hexagons minimise perimeter-to-area ratio, reducing edge effects when aggregating.

Sources: https://www.researchgate.net/publication/252865933 , https://conwaylife.com/wiki/Hexagonal_neighbourhood

## 6. Gap statement
Nothing in the surveyed ecosystem lets you (a) keep a Zarr/COG on its native grid and CRS, (b) attach stable global H3 identities to its pixels and chunks, and (c) run H3 traversal and hierarchy functions to drive reads, windows, joins and aggregations on those native pixels. That is the space this repo explores.
