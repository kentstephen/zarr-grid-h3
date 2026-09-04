# 14: The CTrees pair (`s2-ctrees-pair.py`)

Status 2026-09-04: decided and built, first flight pending. A fork of the
settlement pair (docs/13) with the label side swapped for one source, CTrees
global aboveground biomass, and the pair made simpler, not bigger. The
earlier version of this doc (2026-09-03) weighed two shapes; Stephen chose
the second and dropped the rest.

## What was decided (Stephen, 2026-09-04)

- "This is just like a worldwide notebook." No home story, no Paradise, no
  fire. The geocoder is the home; the fold reads whatever box is looked at.
- "Res nine hexagons are fine." The ladder stops at res 9, one hexagon per
  CTrees pixel, and never runs finer while the picture keeps zooming.
- "The Sentinel shows how it's changed over time, this smaller window."
  S2 2022 to 2025 on the left is the live picture. The biomass window on the
  right is free across 2000 to 2025; the two never need to agree.
- "This dataset shows where biomass was gained and lost." The primary fill
  is a diverging change, not a green stock ramp. Cool to warm, blue for gain
  and orange for loss, white in the quiet middle.
- Uncertainty is in the store (a residual standard error per pixel per
  year). "We can set that to opacity."
- Landsat comes in, but never on its own: "use both of the datasets and
  start with Sentinel-2 for the fast. And when you find an AOI, you just
  switch to the Landsat if you want. When you move around, you don't move
  around with the Landsat." Landsat is a still, fetched on a key for the box
  in view, pinned there; S2 keeps drawing around it. One picture at a time
  on the left.
- The worry about "burning through" the Copernicus quota by playing is a
  time worry, not a quota worry (numbers below), and the manual fetch is
  the answer: nothing loads unless asked.
- The other things that came up (WSF beside it, AlphaEarth, Overture,
  "cleared before built") are "another project altogether".

## The sources

### CTrees AGB (AWS Open Data, measured 2026-09-04)

- Icechunk (Zarr v3), anonymous, bucket `ctrees-agb-100m-global`, prefix
  `agb_100m_global`, us-west-2, branch `main`, group `aboveground_biomass`.
- Grid EPSG:4326 plate carree, 0.000888 degrees (100 m at the equator),
  dims (26, 202500, 405000), x from -180 east, y from 90 south. Pixel
  centres at the edge plus half a pixel.
- `agb` and `uncertainty`, int16 stored (xarray decodes to float64 unless
  `mask_and_scale=False`), divide by 10 for Mg/ha, valid 0 to 6000 stored,
  nodata -9999. Uncertainty is the residual standard error, same shape.
- Chunks (1, 2000, 2000). A 0.3 by 0.4 degree box (Ji-Parana, Rondonia) is
  one chunk per year: 26 years of `agb` in 10 s sequential from here, two
  years in 1.1 s; the notebook reads the years in parallel threads.
- The uncertainty is large. Over that box the median ratio of uncertainty
  to stock is 0.78 (p10 0.37, p90 1.4) where the stock is above 5 Mg/ha.
  A rule that greys every cell whose uncertainty is a large share of its
  stock would grey most of the forest. So the change fill's opacity is the
  change measured against the uncertainty of its two ends
  (|a1 - a0| / sqrt(u0^2 + u1^2), full at 1 and above), and the stock fill
  fades only where the uncertainty exceeds the stock.
- Preprint Yang, Saatchi et al. 2026 (doi 10.31223/X5KJ4Q), dataset doi
  10.82924/7vmb-zv66, CC-BY 4.0. Method: DenseNet on Landsat, PALSAR,
  GEDI and ICESat-2, topography. Smoothed: a clearing can ramp over years,
  so the loss year is the first year the drop from the window's running
  maximum passes a threshold (25% of that maximum and at least 10 Mg/ha),
  not a single-step test.

### Sentinel-2 (Earth Genome): unchanged from docs/12 and 13

Both composites, the yearly first and the temporal median filling its
holes. 2022 to 2025. The left pane's live picture.

### Landsat, the OpenGeoHub mosaics (probed 2026-09-04)

The one yearly global Landsat picture that matches the format, and where it
lives now:

- OpenLandMap's STAC (`s3.eu-central-1.wasabisys.com/stac/openlandmap`)
  still lists `landsat_glad.swa.ard2_yearly.p50`: 28 yearly items 1997 to
  2024, 30 m, seven bands plus RGB composites, one global COG per band per
  year, CC-BY 4.0. Every asset href points at `s3.opengeohub.org`, which no
  longer resolves. The OpenLandMap `arco` bucket itself is open, listable,
  ranged reads and CORS `*`: a walk of all 11,946 keys found no global
  Landsat reflectance, only Europe-only derived products (EPSG:3035), the
  GLAD land cover maps at five-year steps 2000 to 2020, and two stray 2022
  NDVI bimonthlies. The catalog is stale; the data moved.
- Copernicus Data Space (CDSE) has it: STAC collection
  `opengeohub-landsat-bimonthly-mosaic-v1.0.1`, 1997 to 2024, global, tiled
  by 1 by 1 degree cell and two-month period, seven band COGs per item
  (B01 blue, B02 green, B03 red, B04 NIR, B05 SWIR1, B06 SWIR2, B07
  thermal) plus a clear-sky mask. EPSG:4326, 0.00025 degrees, 4000 by 4000
  per cell, uint8 already, about 7.5 MB per band file. No yearly collection
  on CDSE; a year is one period picked or a median of six. STAC licence
  field "other" (the OpenLandMap copy said CC-BY 4.0); read the CDSE terms
  before publishing anything derived.
- Every asset is `s3://eodata/...`. Anonymous HTTPS on the eodata path is
  403, the OData download is 401, the thumbnail WMS answers without a key
  but renders blank for this collection. The STAC search itself is open
  (that is how the notebook finds the cells under a box).
- Quota for a free general user on eodata S3 (documentation.dataspace.
  copernicus.eu/Quotas.html): 2000 requests per minute, 4 concurrent
  connections, 20 MB/s per connection, 12 TB per month (after which 1 MB/s
  and one connection). A city box at 30 m is about 2.4 MB per band per
  period; one year as RGB from one period is about 7 MB, from six periods
  about 45 MB; all 28 years from one period about 200 MB. Ten thousand
  boxes a month before the cap. The four connections and the Europe round
  trip are the slow part, not the quota.

#### Getting the key

1. Register at https://dataspace.copernicus.eu (free). Stephen has an
   account (2026-09-04: "their website is a hellhole").
2. From the terminal, no website: `uv run python cdse_keys.py` prompts for
   the account's email and password, takes a Keycloak token by password
   grant (`client_id=cdse-public` at identity.dataspace.copernicus.eu),
   POSTs to `https://s3-keys-manager.cloudferro.com/api/user/credentials`
   and appends the returned pair to `.env` as `CDSE_S3_ACCESS_KEY` /
   `CDSE_S3_SECRET_KEY`. That is CDSE's own recipe (the Python example on
   their S3 page). `--list` shows the account's pairs, `--delete ID`
   removes one. Two-factor on the account breaks the password grant; then
   the site's keys manager (eodata-s3keysmanager.dataspace.copernicus.eu)
   is the only way.
3. `.env` is gitignored; the notebook loads it with python-dotenv at
   start. Restart the kernel after writing it.
4. The endpoint is `https://eodata.dataspace.copernicus.eu`, bucket
   `eodata`, path-style requests, any region string. Without the two
   variables the Landsat buttons say so and the left pane is S2 only.

## The notebook

`s2-ctrees-pair.py`, forked from `s2-wsf-aef-overture-pair.py`. Kept:
the two-map anywidget, the camera loop, the S2 tile cell verbatim, the
DataFusion H3 fold, the DuckDB click row, the geocoder, the headers, the
strip. Gone: WSF, AlphaEarth, the WSF pyramid tiles, the quiet-level D0,
the county join. New: the CTrees fold, the frame, the Landsat still.

### Right pane

- **Fold**: `ct_fold(box, res)` reads all 26 years of `agb` and
  `uncertainty` under the padded box on a stride that fits CT_MAX_PX
  samples per year (a zoom 9 pane is about 8 Mpx per year unstrided, so a
  stride of 2 there; zoom 10 and up unstrided), the years in parallel
  threads through one semaphore, then one DataFusion query: the cell at
  `res`, the sample count, and 52 means (`a_2000`..`a_2025`,
  `u_2000`..`u_2025`, in Mg/ha, nodata excluded). Cached per (res, box).
  Reading all 26 years at once is the design: a window change is a frame,
  never a refetch.
- **Ladder**: res 6 at zoom 6.2, one finer per 1.4 zoom, capped at res 9
  (from zoom 10.4 on). Below HEX_ZOOM 9 the right pane is the basemap and
  the status says zoom in: CTrees has no pyramid, and a zoom 6 pane would
  be 300 Mpx per year. An open.
- **Frame** (`build_frame(tab, y0, y1)`): `stock` = a_y1; `change` =
  a_y1 - a_y0; `z` = |change| / sqrt(u_y0^2 + u_y1^2); `lossyear` = the
  first year in y0+1..y1 where the drop from the running maximum since y0
  passes max(LOSS_ABS 10, LOSS_FRAC 0.25 * that maximum); `lost` = the
  deepest such drop; `recovered` = a_y1 minus the window's minimum;
  `ushare` = u_y1 / a_y1.
- **Fills** (keys 1 to 3):
  - **change**: diverging blue (gain) / white / orange (loss), symmetric
    about zero on the larger of |p2| and |p98| of the view's changes;
    alpha from ALPHA_QUIET at z = 0 to ALPHA_FILL at z >= 1.
  - **loss year**: the cividis ramp from y0+1 (dark) to y1 (light), a
    lightness ramp so 26 years read as an order; cells with no loss past
    the threshold light grey and faint.
  - **stock**: greens on a_y1, view p98; faint where the uncertainty
    exceeds the stock.
- **Window**: the two-handle slider, 2000 to 2025, keys `-` `=` and
  `_` `+`. Opens at 2000..2025.
- **Click**: three plain sentences (the stock at both ends and the change;
  the sharpest fall and its year, or none, and what came back; the
  uncertainty at the to-end and the change as a multiple of it) and a
  small inline SVG of the 26 values with the window shaded.

### Left pane

Round 2 (2026-09-04, after Stephen's first run with the key): stills
accumulate across pans instead of being cleared (the first version wiped
the held years whenever the box changed, which read as "toggling between
years and nothing happens"); one true-colour stretch per box, taken from
the first year fetched, so a year with more bare ground stays brighter;
the compositing runs in a thread; the last Landsat message sits in the
left header. And NDVI as the picture (Stephen: "the ndvi should cut
through the tci noise"; the Landsat true colour "doesn't look so good in
tropical places"): a SHOW toggle, true colour or NDVI (key `n`), for
whichever sensor the slider is on. Sentinel-2 NDVI reads B04 and B08 of
the yearly mosaic (uint16, the TCI's pyramid, source.coop; the Sentinel-2
offset of 1000 removed first, dense-forest red reads ~1235); the temporal
mosaic's bands are on `ei-imagery` (us-east-2), which refuses anonymous
reads, so NDVI has no cloud backfill. Landsat NDVI reads B04 (NIR) beside
B03 and each still is sent in both modes. One fixed scale, -0.1 to 0.9,
on Carto's Emrld (Stephen's pick), for every year and both sensors.
Round 3 (same day, from Stephen's screenshot): the ramp reversed, dark low
and bright high; two legends, the picture's under the left pane (sensor,
year, mode, the NDVI ramp) and the fill's under the right; the low-value
clipping fixed (the first NDVI treated an offset-corrected band at zero as
nodata, so water and shadow vanished; nodata is raw zero only, corrected
bands floor at 1); the offset per item (the yearly mosaic carries the
Sentinel-2 offset of 1000, red p1 1177 over 20LPP 2022; the temporal
median does not, red p1 201; the 2022 and 2023 temporal bands ARE on
source.coop, only 2024's sit on the private ei-imagery bucket, so NDVI
backfills 2022 and 2023 and the 2024 items are skipped); a slider move
onto a Landsat year already on disk for the box loads it without a key
press.
Marimo lesson: a cell-private `_name` referenced before its `def` in the
same cell fails at run time with a mangled NameError; define helpers
above their callers.

- **Picture slider**: 1997 to 2025 in one control. 2022 to 2025 are the S2
  mosaics, live tiles. 1997 to 2021 are Landsat years: the readout says
  "not fetched" until a still is held; then the still is drawn over S2
  2022 (which keeps drawing around it) at the box it was fetched for.
  Arrows and `[` `]` step it as before; the scale slider is the S2 gain.
- **Landsat buttons** (left header): `this year` (key `k`) fetches the
  picture year for the box in view (the fold's padded box when the
  hexagons are on, else the padded view); `all years` (key `j`) fetches
  every Landsat year not yet held for that box, in the background, one at
  a time. A fetch for a different box clears the stills. Stills are
  cached on disk under the tmp cache dir as PNG plus bounds, keyed by year,
  period and box, so a second session on the same box reads no bytes.
- **The still**: for each period in LS_PERIODS (default one, `07-08`; more
  than one and the median is taken), the STAC search for the box and the
  period's first day, then B03/B02/B01 windows from each degree cell,
  nearest-sampled onto an output grid of the box (columns even in lon,
  rows even in Mercator y so deck's BitmapLayer is exact) at 30 m capped
  at LS_MAX_PX, a per-band p2-p98 stretch, alpha where any band is
  nonzero. Four connections at once (the quota). The status line carries
  a running total of the bytes asked for this session.

### What stays open

1. The right pane below zoom 9 is empty. A coarse tile render from the
   store on a stride, or a precomputed res 5-7 layer, are the two ways.
2. Which period stands for a Landsat year in the tropics; `07-08` is a
   guess for the dry season in the southern Amazon and wrong elsewhere.
   A per-place choice or the six-period median are the options.
3. The loss threshold (25% and 10 Mg/ha) is a first guess against a
   smoothed product; the click story prints the series so it can be read
   against the fill.
4. Licence of the CDSE copy of the Landsat mosaics.
5. The Landsat cell is written against the STAC's stated layout (EPSG:4326,
   uint8, 0.00025 degrees) and has not been run: no key here yet.

## Watch-outs

- `mask_and_scale=False` on the open, or the int16 comes back float64 and
  the nodata is NaN, not -9999.
- Divide by 10 after masking, never before.
- No red, no red-vs-green: gain blue, loss orange, years cividis, stock
  greens (greens are fine).
- The maplibre keyboard is off on both maps (docs/13); the widget's keys
  need a click on the widget first.
