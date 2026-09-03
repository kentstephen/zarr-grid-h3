# 13: The S2 settlement pair (`s2-wsf-aef-overture-pair.py`)

A fork of the S2 hex pair (doc 12). Same chassis: two maplibre maps in one
anywidget, one camera, the Sentinel-2 yearly mosaic on the left as a picture
that is never covered, one opaque H3 fill on the right. What changes is the
label side. LCMS and MTBS go; the World Settlement Footprint Tracker comes in
as the record of when ground became built-up, AlphaEarth stays as the
embedding that should see the same thing, and Overture Maps supplies the
county line on both panes and the county every hexagon belongs to.

Stephen, 2026-09-02: "pair the aef embeddings with [mindearth/wsf] and follow
xsql-canopy-deforest.py to join the data with overture boundaries using
cboettig/overturemaps pmtiles ... pair aef and the mind earth zarr for
agreement and vectorize for h3 for viz and join with overture." On the S2
mosaic: "might as well use the s2 dataset too so we can look at several years
and see if there's change." On the raster over the hexagons: "I'm not sold on
showing the raster over the h3 either way." Overture BUILDINGS from the fused
partition on source.coop are a stated later step (the bias-bounty tutorial
notebook draws a lot of buildings only when zoomed in): noted, not started.

## The data, as measured (2026-09-02)

### WSF Tracker (mindearth/wsf on source.coop)

- One GeoZarr v3, `World_WSF_20160701-20260101.zarr`, one variable
  `wsf_tracker`, int8, global 10 m in EPSG:4326 (plate carree, 8.98e-5
  degrees per pixel), 1,536,433 x 4,007,502 pixels, 60.01 S to 78.01 N.
  Shards of 8192 x 8192 holding 256 x 256 inner chunks, zstd. Ranged reads of
  small windows through obstore and zarr 3 with no extra tooling.
- The value is a date index: 0 never built-up, 1 already built-up by July
  2016, k = 2..20 the half-year the pixel first read as built-up, ending at
  January 2017 .. January 2026. Index k's period ENDS at its date, so the
  pixels built DURING calendar year Y carry `2 (Y - 2016) + 1` (January to
  July) and `2 (Y - 2016) + 2` (July to January).
- Levels 1..12 are a MIN pyramid over the nonzero pixels: the earliest date
  under the window, a dilation. Verified against native pixels over Chico: the
  built share is 3.4% at level 0, 4.6% at level 2, and level 2 matches neither
  a plain min nor an any-built rule exactly (96% of blocks either way). The
  README says the same: analysis from group 0 only. So the pyramid DRAWS (the
  right pane below the hexagon zoom, the earliest date's year color) and the
  fold READS level 0 on a stride.
- Read cost at the Paradise home: 77 Mpx (zoom 10 pane, padded) in 0.7 s;
  308 Mpx (zoom 9) in 0.8 s whole or 1.0 s on a stride of 6. The bytes are
  cheap; the h3 pass is what the sample budget (`WSF_MAX_PX` = 12 M)
  protects. Stride s = ceil(sqrt(window / budget)), every s-th pixel in each
  axis: an unbiased sample of a cell's share and date mix. At res 8 a cell
  holds ~9,800 native pixels; on a stride of 3 about 1,100 samples.
- AEF and WSF sit on near-identical 10 m lat/lon grids with different origins
  and a 4.6e-10 degree pixel-size difference. Both fold through lat/lon to a
  cell, so alignment never matters.

### Overture divisions (cboettig/overturemaps, release 2026-02-18.0)

- Three levels (countries, regions, counties), each as parquet, PMTiles and
  an H3 res 8 partition. `counties.pmtiles`: z0-14, one layer `counties`
  with `name_en`, `region`, `country`, `admin_level`, `class`, `id`; 6.0 GB,
  ranged GETs with `access-control-allow-origin: *`, so maplibre's pmtiles
  protocol reads it from the browser the way the MTBS perimeters were read.
- `counties/hex/h0=<res 0 cell>/data_0.parquet`: one row per (county, res 8
  cell), `h8` UBIGINT plus the county's attributes, no geometry. A
  materialised polyfill. The Chico/Paradise res 0 cell holds 2,604,758 rows
  for 299 counties; 285,287 rows (11%) share an h8 with another county (the
  partition is coverage, not centre-ruled). The join is an integer equi-join
  on the res 8 parent (finer frames) or a parent group-by of the h8 rows
  (coarser frames): no polyfill, no dissolve, no ST_ anything.
- The file has no row-group statistics on `h8` (22 groups, stats None), so a
  range predicate cannot prune. But reading the id column alone and
  filtering by the id range of a res 2 cell's children returns in 3.1 s
  where the whole file with its strings takes 13.4 s. The notebook reads by
  res 2 tile, one query per res 0 file, and holds the rows.

### Sentinel-2 (Earth Genome), AlphaEarth: unchanged from doc 12.

One addition, 2026-09-03. Nusantara (Penajam Paser Utara, East Kalimantan)
drew nothing in 2022: not a bug. Earth Genome publishes TWO Sentinel-2
composites on the same STAC and the same source.coop host, same MGRS tiles,
ids, band names, EPSG:3857 pyramid, different pixels:

- `sentinel2-yearly-mosaics` (bucket `earthgenome/earthindeximagery`): the
  Earth Index imagery, method undocumented, STAC licence `proprietary`
  (the notebook credit says CC-BY 4.0: confirm on source.coop), 2022-2024 in
  the STAC plus 2025 in the bucket. Cleaner where clouds are rare (Berlin
  good_pxl_pct 0.99 vs 0.91), hollow where they never lift: the 2022 tiles
  50MME/50MMD hold 0 valid pixels over the city, a cloud-shaped hole the
  size of Balikpapan Bay across four tiles (good_pxl_pct 0.22, 0.56).
- `sentinel2-temporal-mosaics` (bucket of the same name): documented, L2A
  masked by SCL then the per-pixel median over the year, CC-BY 4.0, 2022 and
  2023 only on this STAC. 43% (50MME) and 100% (50MMD) valid over the city
  in 2022 (good_pxl_pct 0.72, 0.69).

The pair keeps the yearly as primary (it alone has 2024 and 2025, where
Nusantara's growth is) and BACKFILLS: `_stac` asks for both collections in
one search, `_s2_items` lists the yearly footprints first and the same
year's temporal footprints after them (ids suffixed `#fill`), and the tile
compositor's first-to-paint-wins rule does the rest. Per year the kernel
counts the painted pixels that came from the fill; the status line says
"S2 2022: 41% of pixels backfilled from the temporal median" whenever the
share is nonzero (cumulative over the tiles served, not the view). Checked:
Nusantara z11/z13 2022 now paints fully, 100% from the fill; 2023 takes
0.1%; Sacramento 2022/2024 take none. The two composites differ slightly in
colour, so a backfilled frame is a patchwork and the strip says so. The
indices deck (`xsql-aef-lcms-s2-deck.py`) is left on the yearly alone:
mixing two composites inside one avg() per hexagon is not a picture problem,
it is a measurement problem.

When this notebook moves to its own repo, carry the temporal-mosaics credit
with it (README / attribution block): "Sentinel-2 L2A temporal mosaics by
Earth Genome, CC-BY 4.0, Copernicus Sentinel data", beside the yearly
credit. And settle the yearly collection's licence line at the same time.

## What the notebook does

- `wsf_fold(box, res)`: one strided window of level 0, pixels -> cells
  through the h3 UDF inside DataFusion (repo rule), per cell `npx`, `built`
  and one count per date index `c01`..`c20`. Every cell with a sample is
  present, so the grid is whole. Cached per (res, box). A window change is a
  FRAME, never a refold: the window's shares are derived from the counts.
- `county_rows(box)` / `county_for_cells(cells, res, rows)`: the h8 rows
  under the box (by res 2 tile, cached), then the cell -> county lookup in
  DuckDB h3 (`h3_cell_to_parent`), keeping the first county by name on a
  border cell and flagging `border`.
- `build_frame(wsf, aef_by_year, y0, y1, cty)`: from the counts, `p_built`
  (built-up by the end of y1), `p_new` (became built-up in y0+1..y1, the
  years the y0 and y1 composites straddle), `grew` (p_new >= `NEW_MIN`, 1%),
  `byear` (the year with most of the new pixels; -1 built before the window,
  -3 nothing built), `first_date` (the record's first built-up half-year).
  AlphaEarth as before: `disp` between the ends, the steps inside, D0 as the
  95th percentile of the largest step among the cells WSF says did NOT grow
  (LCMS Stable used to play this part), `when` the first step above D0.
- Five fills, keys 1-5: `WSF built` and `WSF grew` on a blue lightness ramp
  (matplotlib Blues less its white end, stretched to the view's p2-p98 of the
  nonzero cells, zero drawn faint); `WSF build year` and `AEF change year` on
  ONE palette per year (Okabe-Ito less the red, plus teal, brown, near-black;
  2016 mid grey), so a hexagon that is the same color on both is a hexagon
  where the record and the embedding agree on the year; `AEF changed` on
  viridis as before. The WSF pyramid tiles use the same year palette.
- Controls: S2 year and scale on the left header as before; FILL and the
  WINDOW slider on the right (the LCMS year control is gone; the window is
  the one control both sources read). `B` toggles the county lines. Hover a
  line and the strip names the county (`.sp-place`). Click a hexagon: the
  county, what WSF says (share built, share new in the window, the year), what
  AlphaEarth saw.
- Under the map: the view's counties ranked by the share of their sampled
  ground that became built-up inside the window (the join become a number,
  the canopy notebook's box ranking without the box), and a crosstab of WSF
  build year against AlphaEarth change year.
- Home: Paradise, California (-121.60, 39.76, zoom 10.2). The Camp Fire
  (November 2018) took most of the town; WSF dates the rebuilding half-year by
  half-year, and the four mosaics show the roofs coming back.

## On the raster over the hexagons

The S2 pair settled this once: the imagery is the floor of the left pane and
the hex fill owns the right. WSF as a raster does not add a picture the way
S2 does; it adds a second label. A 10 m date raster under a hex fill of a
date is the overlay the pair rejected. So WSF is drawn as a raster only where
the hexagons are off (below zoom 9, from the pyramid, which is what a browse
pyramid is for), and everywhere else it is a fill on the same hexagons as
AlphaEarth. The pixel-count backing per cell is a number in the row, not a
picture.

## Not carried from doc 12

LCMS (the fold, the raster, the year control, `CHG_MIN`), MTBS (the severity
fold, the perimeters, the fire under the pointer). The chassis, the S2 tiles,
the AEF fold, the camera loop, the strip, the full-screen fix, the header
cards and the AEF slider are verbatim.

## Build log

Round 1 (2026-09-02): assembled from the pair notebook by line range plus
exact-string patches on the widget (scratchpad `build1.py`), new cells for
WSF, counties, frame, map, wiring, tables. `fly_wsf.py` (port 2735,
shots/wsf/) is the pair harness with the label-year step replaced by a
window step (`+` key) and the perimeter checks replaced by county-line
checks.

Flight 1 ran end to end; the hover and click missed because the widget sits
below the 900 px viewport (the intro is longer than the pair's): a harness
fix (scroll the widget into view before the pointer work). Flight 2 clean:

- boot 20 s; res 8 at the home, 3,081 cells.
- WSF 3,028 x 2,395 samples (stride 2, 20 m) read 0.5 s, fold 1.3 s.
- counties 221,989 h8 rows, 2 res 2 tiles fetched, 3.1 s (once; 0.0 s after).
- AEF 2020..2023 ov4, ~3.4 s each in parallel, fold 0.1 s each; frame 0.2 s;
  whole serve 3.8 s. Window step to 2024: one more AEF year, no WSF refold,
  2.3 s. Pan: 1.5 s.
- hover: "Butte County, US-CA (Overture county)"; the ring on both panes.
  Click: "Butte County, US-CA. WSF: 9% of this hexagon is built-up by the end
  of 2023; 1.1% of it was built 2021 to 2023, most of that in 2021.
  AlphaEarth saw no change from 2020 to 2023."
- county lines on both maps (5 features rendered each), layer order
  cty-hit / cty-case / cty-line under watername_ocean; zoom 8 and 7 WSF
  pyramid tiles settle in ~5 s, the pyramid almost all "by 2016" grey with the
  towns as blobs (Chico, Paradise, Yuba City) under the county lines.
- the home window: 23 of 3,081 cells grew 2021..2023, 151 of 3,036 scored
  cells moved in AlphaEarth, none in both. The quiet level D0 (0.29) is set
  by the cells WSF says did not grow, and around Paradise those include the
  whole Camp Fire scar regrowing, so it sits high. Worth a look at the
  crosstab with a window that starts before the fire (2018..2021).

Round 2 (2026-09-02, flown): the lines redone. Stephen: "we need states,
maybe localities ... the black lines for the counties, not looking good ...
make it align, silver or gold, I think gold"; "states and the counties and
the localities are zoom dependent just like in the other notebook". So the
lines now come from Overture's own divisions.pmtiles (overturemaps-extras
bucket, release 2026-08-19.0, 19.5 GB, CORS on the bucket), the
`division_area` layer filtered by `subtype`, ONE level per zoom band the way
the canopy notebook does it: states below zoom 8, counties 8 to 10,
localities from 10. The floors are where the build first holds each subtype
(counties z8, localities z10); maplibre does not underzoom, so a band cannot
open before its tiles. One gold line (#ffc828, the MTBS perimeters' gold),
no casing; 2.2 / 1.6 / 1.2 px. Hover names the division, nothing else
("drop (Overture county)"). cboettig's h8 partition stays the join. Flight 3:
Paradise named as the locality at zoom 10.2, 28 county features at zoom 8,
14 state pieces at zoom 7, on both panes. Boot 51 s this time (the pmtiles
header and the first directory of a 19.5 GB file, plus a slow S2 STAC).

Round 3 (2026-09-02, flown, COMMITTED): Stephen: "wsf built should be the
zarr buildings not h3. the grew can be h3, blue is the wrong cmap for that,
and the detail from both these datasets could get us to res 12, not that we
need to go that high but I want to try." So:

- `WSF built` is the raster. Picking it draws the WSF tiles on the right at
  every zoom (the pyramid below 9, level 0 from about zoom 13) and no
  hexagons; the legend is the year palette. The frame still folds underneath,
  so the hover ring, the click story and the tables keep working.
- `WSF grew` moves to a warm lightness ramp (matplotlib YlOrBr less its white
  end). The share fills no longer use blue at all.
- The ladder opens to res 12 (MAX_RES 12, CELL_BUDGET 300k): zoom 14.6 and
  up. A res 12 cell is 307 m2, about three pixels of WSF and three of the
  AEF mosaic, which the mosaic path (MOSAIC_MIN_RES 11) reads natively.
- Flight 4 over Paradise at zoom 14.8: res 12, 232 x 199 WSF samples on a
  stride of 1, fold 11,263 cells in 0.0 s, four AEF mosaic years 1.3-1.8 s
  each, 8.9 s to the screen; 142 cells grew 2021..2023, D0 0.121. The built
  raster at that zoom shows the record at its own grain: most of Paradise
  still "by July 2016" grey, because WSF books a FIRST built-up date and
  never unbuilds, so the Camp Fire's losses do not show and only the new
  ground since carries a year color. Worth stating on the panel at some
  point.

Round 4 (2026-09-02): localities off. Stephen, over Las Vegas at zoom 11:
"the locality thing is a bit much, these look like neighborhoods or
microhoods, getting in the way of observation. comment out locality for
now." The county band now runs from zoom 8 with no upper end; the
three-band constant is kept as a comment in DIVISION_BANDS.

## Opens

- WSF never unbuilds: a burned lot stays "built by 2016" in the record. The
  grew fill and the mosaics are where the rebuild reads; the built raster
  cannot show the loss.

- THE ZOOM INVERSION (Stephen, round 2): "we have this great detail from the
  zarr ... and then we zoom in and we get these coarse h3. It doesn't make
  any sense. We need to step back and think about this fundamentally." Below
  zoom 9 the right pane is the pyramid, which reads as building-level detail
  because settlements are sparse blobs; above it the frame is res 8 hexagons
  at 0.74 km2 while the data is 10 m. Directions, none chosen:
  raster zoomed in and hexagons zoomed out (the coarse fold needs a
  precomputed WSF -> H3 table, since a native strided read at zoom 7 decodes
  gigabytes); a much finer ladder (res 10-12 from zoom 10, the canopy
  notebook ran res 10-11 for the 1 m CHM; WSF supports res 11 at ~215 px);
  the raster as the right pane at every zoom with H3 kept as the analysis
  container (hover, click, tables) and never drawn; a raster / hexagons
  toggle on the right pane, never both.

- Overture buildings from the fused partition, drawn only zoomed in (Stephen's
  next layer).
- The county tie rule on border cells is "first by name"; the partition
  carries no area share. A centre-rule would need the geometry.
- The WSF pyramid's dilation means the right pane below zoom 9 overstates
  built-up extent; it is labelled as the earliest date, not a share.
- Countries without counties (48 of 219 in Overture) fall through to no
  county; the regions partition could stand in.
