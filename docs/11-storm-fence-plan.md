# 11: The storm fence

One univariate field on its native grid; on top of it, a boundary line that is
some OTHER dataset's version of the same storm, translated into hex-edge
geometry through the shared H3 labels. Agreement is the fence hugging the
colour. Disagreement is rain spilling out of the pen, or an empty pen around
dark ocean. The comparison happens in the viewer's eye, not in a ramp.

This is the heat-domes grammar exactly (field + H3 boundary layer), with one
change: the fence comes from a different dataset on a different grid. That
change is the whole demonstration of the repo. The join never appears as
colour; it appears as the fact that the fence can be drawn at all.

Mocks that landed: `shots/mock_fence.png` (the shape of the thing),
`shots/mock_contingency.png` and `shots/mock_palette.png` (rejected roads,
kept for the record).

## Axioms (settled across sessions, do not relitigate)

- Univariate colour only. No bivariate in any form, including value-by-alpha
  and brightness x saturation. The second dataset enters as a LINE.
- One physical, feelable quantity on the ramp. Maps of relationships
  (differences, ratios, categories of agreement) do not read for Stephen.
- Protan-safe: no red anywhere, no red-vs-green. Blues ramp (dark blue to
  white on the dark ground) is the field; the fence takes the yellow accent.
- One moving thing per map is the instinct; the fence is a guide layer, not a
  second subject. Whether two fences at stepped weights stay readable is an
  open question to judge on screen, not in prose.

## v1 pairing: radar vs model

Field: MRMS observed precipitation (the closest thing to truth, univariate,
physically feelable). Fence: HRRR analysis precipitation, the model's claim
about the same hour. Live to the hour on both sides.

The fence slot is generic, and that is the instrument's future, not v1 scope:

- forecast fence around the analysis field (a promise being kept or broken)
- GFS fence around the HRRR field (what the global model thinks this storm is)
- CONUS404 p99 fence around today's Fosberg ("unusual for here" as a line,
  which may be why it never worked as a fill)
- GEFS member fences bundling where the ensemble agrees

## Data (source.coop, dynamical/, anonymous)

- `mrms-conus-analysis-hourly` v0.3.0: 0.01 deg lonlat CONUS, chunks
  (648, 100, 100), so one time chunk is 27 days for a 100x100 tile. Verified
  in the S3 listing on 2026-08-30; variable names, units, and the exact grid
  origin still need a probe before anything else is written.
- HRRR analysis: the icechunk store already mirrored on disk
  (join/hrrr_mirror.py, chunk 47 = May-Jul 2026 window for precip among
  others), or the source.coop zarr v3 mirror with the same shape/chunks.
- Both stores are live to roughly the hour; LIVE mode (last N hours through
  the newest) is a first-class preset like the AR plan intended.

### Probe results (2026-08-31, proto/probe_mrms.py)

- Store is plain zarr v3, NOT icechunk: bucket `us-west-2.opendata.source.coop`,
  key `dynamical/noaa-mrms-conus-analysis-hourly/v0.3.0.zarr`, anonymous HTTP.
  No MirrorStore inner-session plumbing needed; the HTTPStore + ObjectStore path
  from join/prep_conus404.py works as-is.
- Six 3D vars, all (103721, 3500, 7000) float32, chunks (648, 100, 100),
  shards (648, 700, 1400): `precipitation_surface` (the QC'd blend; equals
  pass_2 at the sampled wet pixel), `precipitation_pass_1_surface`,
  `precipitation_pass_2_surface`, `precipitation_radar_only_surface`,
  `categorical_precipitation_type_surface` (codes; -3 = no coverage, up to 96),
  `flash_qpe_ffg_max_surface`.
- UNITS TRAP: precip vars are `kg m-2 s-1` (mm/s). Multiply by 3600 for mm/h.
  Typical CONUS hour: 5-6% of pixels wet at >0.1 mm/h, hourly max 60-90 mm/h.
- Grid: lat 54.995 -> 20.005 step -0.01 (row 0 is NORTH, descending), lon
  -129.995 -> -60.005 step +0.01. Wider than CONUS proper (reaches 60W).
- Time: hourly since 2014-11-01, T=103721 at probe time, newest hour
  2026-08-31 16:00Z, lag ~40 min. LIVE confirmed. Youngest 1-2 hours are
  partially filled (T-1 was 34% NaN); treat the newest 2 hours as provisional.
- NaN story: a STATIC radar-coverage mask, 6.18% of the frame, bit-identical
  at T-24 and T-720. Offshore south and east edges. Outside-coverage is NaN in
  the precip vars (the -3 sentinel lives only in the categorical var).
- Read rates over HTTP: one full-CONUS hour ~5 s; a (24 h, 700, 1000) box
  0.6-0.9 s (19-26 Mpx/s). Time chunk 648 h = 27 days per 100x100 tile.
- HRRR forecast-48-hour is `noaa-hrrr-forecast-48-hour/v0.1.0.zarr` (plain
  zarr, not the -virtual ones): (init 11885, lead 49, 1059, 1799), chunks
  (1, 49, 265, 300), has `precipitation_surface` and `composite_reflectivity`.
  That is the future forecast-fence source; not v1.

## What gets drawn

1. **The field**: MRMS precip on its native 0.01 deg mesh, RasterFilm
   carry-over. Ramp dark blue to white, low end faded to transparent so dry
   ground stays dark. Nothing else is filled.
2. **The fence**: per frame, HRRR precip >= threshold rolled to fence cells,
   dissolved to the outline edges, drawn as one line in the accent yellow.
   Hex-edge walk, not a smoothed contour: the hexes ARE the honesty about
   where the join lives.
3. **The transport**: window loader, day ticks, HOUR (UTC) stamp, keyboard,
   in the heat-hex panel language (hf/hw classes, dark blurred cards,
   #e6c14a accent). That accent already matches the fence colour.
4. **The pick**: click anywhere, get both time series for that cell (MRMS
   aggregated to the cell vs HRRR aggregated to the cell) as two lines in a
   chart. The chart is the one place the two datasets share a frame directly.

## The fence pipeline (kernel side)

- HRRR pixels already carry res 7 labels (heat domes cache). Fence cells at
  res 6 via cellToParent, membership by the RasterFilm rule select (pixel /
  any / majority / all: how much of a hex the model must wet to be fenced).
- Dissolve member cells per frame to boundary edges. Two options, measure
  both: h3_cells_to_multi_polygon_wkb per frame in Python (heat domes did
  this fine), or ship the nbrs table from hexagg.py and let the browser find
  member/non-member edges per frame (the hex-waves convolution pattern, no
  per-frame geometry shipping at all). The second is likely the win: the
  fence then moves with zero kernel round-trips, threshold slider included.
- MRMS pixels get labels too (for the pick chart and any future field/fence
  swap): latLngToCell per pixel centre at the label res, parent to res 6.

## Label resolutions

- MRMS pixel ~0.9 x 1.1 km, area ~1 km2 mid-CONUS. docs/03 rule (avg cell
  area < half pixel area): res 9 (0.105 km2) is safe, res 8 (0.737 km2) is
  borderline. Measure uniqueness on the box like the heat-domes relabel
  check before committing; labels are only needed where the box is.
- HRRR: res 7 labels exist; fence at res 6 (~4 HRRR pixels per cell).
- Common join res for the pick chart: res 6 parents on both sides.

## Steps

1. Probe MRMS: open the store, list variables/units, verify grid origin and
   chunking, measure a box read (rate, NaN story over ocean). Numbers into
   this doc before any code.
2. Chassis copy from hrrr-heat-domes.py: constants, DuckDB, MirrorStore keyed
   on the MRMS prefix, window cell with BOX + presets + LIVE. Two linear
   lonlat index formulas for MRMS; the HRRR LCC math carries over unchanged.
3. Labels: MRMS res 9 (or 8 if uniqueness holds) + res 6 parents, cached npy.
   Reuse the HRRR res 7 cache.
4. Field film: MRMS on its mesh through RasterFilm, blues ramp, opacity
   slider (Stephen sets opacity, not us).
5. Fence: HRRR membership per frame at res 6, browser-side edge finding via
   the nbrs table; threshold slider; rule select.
6. Pick chart: both series per res 6 cell, two lines, ladder level as a
   dashed line.
7. Fly it headless (fly.py sibling); the numbers that matter are boot time,
   fence step cost, and whether the fence visibly chases a real storm.
   Then Stephen watches it in a browser before anything else is added.

## Flown (2026-08-31, storm-fence.py + fly_fence.py)

v1 built and flown headless. Chassis is hrrr-heat-domes.py minus DuckDB and
counties (the browser does the dissolve; MRMS's radar-coverage mask replaces
the land mask). hrrr_mirror.MirrorStore is reused for both stores (MRMS young
shard index by 648-hour chunks). The fence pipeline as planned: static edge
table (one row per hex edge, cell row each side, built by ring-midpoint
pairing of the res 6 WKB at 1e-6 deg, zero odd groups) shipped once; per
frame the browser counts wet HRRR pixels per cell, applies the rule, and
draws every edge whose two sides disagree. No smoothing, no "pixel" rule
(hex-edge geometry by design). Numbers, default box (-100, 27, -82, 40.5),
LIVE 2 days (39 common frames at flight time):

- 2,430,000 MRMS px (covered), res 9 labels UNIQUE for the box; 325,017 HRRR
  px; 78,688 res 6 fence cells; 237,209 edges; ~34 MRMS px and ~4 HRRR px
  per cell (the pick readout confirms the join arithmetic).
- Boot 72 s cold (live window, both mirrors empty); 132 MB to the browser
  for 2 days; static geometry ~15 MB.
- Fence step: 2-5 ms for membership + edge finding at any threshold/rule.
  Threshold and rule never touch the kernel. majority 634 cells / any 1,033
  / all 502 at 1 mm/h: the ordering behaves.
- Frame step ~1.3 s in headless software GL (texture upload dominates,
  fence on or off identical); heat domes measured the same effect. Judge
  real speed in a hardware browser.
- The fence visibly hugs the blue blobs at 1 mm/h and retreats to cores at
  4 mm/h. Whether it reads as the mock is Stephen's call on screen.

## Open questions

- Does a real fence read as clean as the mock? Real HRRR rain areas are
  raggier than a Gaussian. SETTLED (2026-08-31): no gridDisk smoothing is
  borrowed from hex waves; that notebook was a viz experiment only, not a
  source for the fence pipeline. The fence is raw membership (threshold +
  rule) with browser-side edge finding; if it shimmers, that is judged on
  screen first, not pre-fixed.
- One fence or two: forecast fence + radar-derived fence around the same
  field at stepped line weights. Lines may not compete the way fills do.
  Judge on screen only.
- Threshold ladder: single slider (one fence) vs the heat-domes ladder
  (nested fences from one dataset). Nested fences of ONE dataset were
  already legible in heat domes; that stays available.
- Timing semantics: HRRR analysis vs MRMS at the same valid hour is model
  vs radar nowcast, near-agreement expected. The forecast fence (HRRR f24
  around today's MRMS) has more story in it; needs the forecast-48-hour
  store instead of analysis. Decide after watching v1.
- Whether MRMS 0.01 deg at CONUS scale needs the box constraint always on
  (12M pixels full CONUS; heat domes carried full CONUS at 3 km, this is
  ~9x that per frame).

## Flown again (2026-08-31, evening): the load-window wall, found and measured

Loading fixed windows from the transport hit a wall that live never hits.
Symptom: press load on a long window and the config lands (title, labels,
stamp, note all flip to the new dates) but the frames stay the old live
buffer. The ruler keeps saying 39 frames under a 168-frame header; the
raster shows yesterday's weather labeled with the requested hours. A
stray "Cannot read properties of undefined (reading 'op')" in the console
is downstream wreckage of the same failure, not a separate bug.

Root cause, confirmed by headless repro: marimo ships anywidget buffer
traits as base64 strings inside one JSON websocket message, and V8 caps
strings at 0x1fffffe8 = 536,870,888 chars. At this BOX (2.43M MRMS px +
325K HRRR px per frame, ~2.76 MB/frame):

- 5 days = 120 frames = 291.6 MB = 388.8M base64 chars: fits. The Ida
  window (2021-08-28..09-01) loaded fully in ~3.3 min cold, fence 4,201
  cells at the 18Z landfall hour, zero errors. Slow is not broken: the
  cold HRRR chunk is minutes from S3 by design.
- 7 days = 168 frames = 408.2 MB = 544.3M chars: over the cap. The trait
  update dies silently in the browser. The repro harness itself crashed
  on the same limit (node ERR_STRING_TOO_LONG, 0x1fffffe8) relaying it.
- So the honest limit at this BOX is 6 days (349.9 MB = 466.6M chars),
  not the advertised 7. HOURLY_MAX_DAYS lies by one day.

Fix directions (not yet applied):

1. Byte-aware cap: checkWindow already computes the MB estimate; block
   load past ~450M base64 chars and drop HOURLY_MAX_DAYS to match. Three
   lines, makes today's notebook honest.
2. Chunk the transfer: frames as per-day traits reassembled in JS. No
   ceiling, more wiring.
3. Hexes instead of pixels: ship per-res-6-cell bytes (78,688 cells,
   ~157 KB/frame both fields, 17x smaller). The string cap moves from 6
   days to ~4.5 months and the binding constraint becomes read time
   (~20M MRMS px-hours/s server-side), not bytes. The pick, chart, and
   fence already live at cell level; only the field fill changes, and
   the cost is the thesis: cell-mean fill smooths convective cores, the
   native-grid raster reads as radar. Cell max keeps cores, exaggerates
   area. Res 7 (~550K cells) is a 4x middle ground, ~3 weeks of window.
   At 30+ days the frames themselves want rethinking (daily max is
   another 24x; a year of daily maxima is 29 MB).

Direction chosen (Stephen, 2026-08-31): hexes for longer windows in
general; raster stays the short-window look. Shape TBD between the
window-size switch and an explicit toggle.

## Hexified (2026-08-31, storm-fence-hex.py)

Direction 3 built as a SEPARATE notebook (the raster stays the short-window look
in storm-fence.py; the copy answers the toggle question by being two files).
Goal it serves (Stephen): surgical windows on the 2024 hurricane season, watched
as a timelapse with the fence as the live analytic; season-scale daily
aggregates are a later layer on the same carrier.

- Field: MRMS res 7 cell means (~5 px per cell, no holes; HRRR at res 7 would
  have holes, ~9 km2 px vs ~5 km2 cells, which is also why the fence stays at
  res 6). Rendered as one SolidPolygonLayer with per-vertex colours, the
  hex-waves binary carrier: closed rings + ring starts + uint64 cells.
- Fence rules kept EXACT without shipping HRRR pixel-hours: per res 6 cell per
  hour the kernel ships order statistics (any = max, majority = the
  floor(n/2)+1-th largest, all = min; no-data counts as never-wet, matching the
  raster's browser rule) plus the cell mean for the pick chart. q is monotone
  in mm/h so the statistics commute with the quantisation.
- Fence is silver now (205,210,216); the yellow accent moved to the pick ring.
- Window cap computed, not advertised: largest single trait (frames, K7
  bytes/frame) against the measured V8 cap. 34 days at the default box.

Flown, default box, LIVE 2 days (39 common frames): 469,609 res 7 cells,
3,287,779 verts, boot 26.5 s (raster: 72 s), 38 MB to the browser (raster:
132 MB), paint 7-22 ms, fence step 1-7 ms, frame step ~386 ms headless
software GL (raster: ~1.3 s). THE PROOF: fence counts at 1 mm/h are 634
majority / 1,033 any / 502 all, bit-identical to the raster notebook's flown
numbers; the planes reproduce the pixel rule, not approximate it.

Flown, Helene (2024-09-24..29 fixed, 144 frames, past the raster's 6-day
wall): boot 34.8 s, 113 MB, majority fence 1,447 cells at 1 mm/h, 438 at
4 mm/h, zero errors. Shots in shots/helene/.

Costs and opens: cell means smooth convective cores (the thesis's price;
judge on screen vs the raster). wet% is now cell-mean-based, px-weighted.
The forecast fence (f24/f48 vs the radar truth, the 2024-season story) still
needs the forecast-48h store; this notebook joins ANALYSIS to radar.

### Round 2: the season stage (2026-08-31)

- BOX widened to (-100, 24, -75, 41): gulf, ALL of Florida (the old east edge
  clipped the peninsula mid-state), the seaboard. Default window is the bad
  stretch, 2024-09-19 to 2024-10-16 (Helene forms to Milton exits, 28 days).
- Frames escape the V8 cap a second way: the per-window frame block ships as
  four part-traits (frames0..frames3, split on the hour axis, concatenated in
  the browser and validated against nf in config). Cap is per trait, so 4x
  headroom; computed limit is now 75 days at the wide box.
- Match panel (right HUD): contingency per res 6 cell per hour, radar
  membership decided by the SAME any/majority/all rule over the cell's res 7
  children (px-weighted via cnt7), scored only where radar reports. CSI, POD,
  FAR for the current hour plus a clickable window series (recomputed 250 ms
  after threshold/rule settles). Series colours are white/silver + yellow
  (protan-safe, no red).
- Both HUDs collapse to small buttons (keys H and M). Ruler moved to a
  bottom-right card, off the basemap text.

### Round 3: vector basemap (2026-08-31)

- Basemap is now OpenFreeMap Dark (https://tiles.openfreemap.org/styles/dark):
  keyless, no registration, no usage limits, donation-funded. Chosen after
  CARTO put raster tiles behind an API key; CARTO's keyless vector style was
  applied first, then swapped out to drop the CARTO dependency entirely.
- Rendering is maplibre-gl 4.7.1 + deck.gl MapboxOverlay (interleaved), hex /
  fence / pick layers slotted beforeId under the style's first symbol layer,
  so place labels draw ABOVE the rain. Picking via map.unproject; TileLayer /
  BitmapLayer and the raster endpoints are gone from the notebook.
- Known cosmetic: the style references a circle-11 sprite icon that 404s
  (console warning only).

Flown, season window (28 days, 672 common frames): 866,693 res 7 cells,
6,067,916 verts, 990 MB to the browser in part-traits, boot 572.4 s cold
(MRMS ~142 s + HRRR minutes from S3), paint ~20 ms, fence step 2-5 ms, frame
step ~444 ms headless software GL. Fence at 1 mm/h: 1,056 majority / 1,608
any / 831 all cells. Match at the flight's hour: CSI 0.22, POD 0.25, FAR
0.37 (506 hit / 1,523 miss / 297 false). Shots in shots/season/.

- Viz option (todo): fill the fence cells, opacity driven by the per-cell
  match (hit/miss/false this hour), so agreement reads as fill strength and
  stays legible even at full overlap.

### Round 4: legibility (2026-08-31)

- Ruler hidden by default (key R toggles; the harness still reads its text).
- Collapsed HUDs shrink-wrap so their buttons pin to the actual corners; the
  match panel's collapsed button is a minimal arrow.
- Match chart: 120 px tall, raw hourly lines faint under 7-hour centred
  means; CSI bold white, POD blue (#6db1f2), FAR yellow. Protan-safe.

### Round 5: the widest SE stage, compressed (2026-08-31)

- BOX = (-100, 20, -60, 41): the data's own edges for the southeast AOI.
  MRMS ends at lat 20 / lon -60; the HRRR cone at 21.1 / -60.9, so the far
  south-east fill shows radar with no fence, which is honest. Probed
  alternatives on record: N 46 adds ~20% cells; W -105 ~35%.
- Every binary trait ships GZIPPED (level 2). Config goes first carrying the
  compressed part lengths; the browser decodes exactly once, when the last
  part lands and every length matches, streaming through
  DecompressionStream("gzip") straight into ONE preallocated array. No
  concatenation copy, no transient duplicates; the widget model retains tens
  of MB, not a GB.
- The window cap is no longer the V8 string cap (compressed bytes never
  bind): it is a 3 GB decoded-memory budget, frames (K7 B/frame) + four
  planes (K B/frame). 52 days at this box.

Flown, season window at the wide box: 1,575,938 res 7 cells, 11,033,859
verts, 197,993 fence cells, 672 frames, 1,591 MB decoded in the browser,
boot 184.9 s (was 572.4 s uncompressed at the SMALLER box: compression cut
the wire time), paint ~34 ms, fence step 3-6 ms, frame step ~488 ms headless
software GL. Fence at 1 mm/h: 1,912 majority / 2,691 any / 1,576 all. Match
at the flight's hour: CSI 0.34, POD 0.37, FAR 0.19 (1,556 / 2,695 / 354).
Known wording debt: the note still says "MB to the browser" for the decoded
size; the wire is now far smaller.
- Fill colour (todo): try orange for the match fill (silver competes with
  the blues ramp's white top end; blue-vs-orange is the protan-safe axis;
  ~rgb(235,140,60) sits between the ramp and the yellow pick accent).

### Round 6: the fill (2026-08-31)

The match made spatial, per Stephen's concept: each fence cell's area filled
silver, opacity = the px-weighted share of the cell's REPORTING radar at or
above the threshold this hour (max ~0.59 so the storm reads through). A full
pen is solid-ish silver, a marginal one a faint wash, a false alarm an empty
pen; misses have no pen by construction. Second SolidPolygonLayer over the
field, under the fence lines, reusing the vertex carrier with silver RGB
baked and alpha painted per frame. `fill` button + key V, default ON; no
speed gating yet (wake/decay and pause-gating discussed, on the table).

Flown: booted 225.8 s, fill costs a second alpha pass over 11M verts, frame
step ~1,046 ms headless software GL (was ~488; real GPUs will care much
less, and the pass only runs while the fill is on). Match numbers unchanged
(CSI 0.34 · POD 0.37 · FAR 0.19): the fill is a view, not a new statistic.

Round 6 amendment: the fill is ORANGE now (235,140,60; the protan-safe
complement of the blues ramp, per Stephen), alpha cap 180/255. The storm's
luminance detail stays readable through a full match.
