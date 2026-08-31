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
