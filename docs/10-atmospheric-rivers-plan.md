# 10: Atmospheric rivers on the native GFS grid, real time

The next weather pattern after heat domes, on a second native grid geometry.
GFS analysis (0.25 deg regular lonlat, global) rendered on its own mesh, no warp,
no resample, H3 as the label and segmentation fabric underneath. The claim this
notebook makes: the docs/02 approach generalizes across grids. Heat domes proved
it on Lambert conformal at 3 km; this proves it on lonlat at 0.25 deg, with the
same chassis.

Hard constraint from Stephen: real time. The notebook streams small windows off
the live store and boots in seconds to low tens of seconds. No bulk downloads,
no two-hour runs. The measurements below say this is comfortable.

## The store (verified 2026-08-31, probed directly)

Dynamical's GFS analysis icechunk store, anonymous S3:

```python
storage = icechunk.s3_storage(bucket="dynamical-noaa-gfs",
    prefix="noaa-gfs-analysis/v0.1.0.icechunk",
    region="us-west-2", anonymous=True)
```

- LIVE: hourly, 2021-05-01T00 through 2026-08-31T14 (probed today; 46,767 steps,
  the youngest hours only a few behind the clock). Same liveness story as the
  HRRR analysis store.
- Grid: regular lonlat 0.25 deg exactly, 721 x 1440, lat 90..-90 (descending),
  lon -180..179.75. No projection machinery at all: the LCC forward/inverse from
  heat domes is replaced by two linear index formulas.
- 25 vars, all (time, latitude, longitude). Ours:
  - `precipitable_water_atmosphere` (kg m-2, i.e. mm; short_name pwat). The AR
    field. Probed range over the Dec 2022 event box: 0.4..53.5; latest global
    frame max 63.2.
  - `wind_u_10m` / `wind_v_10m` (and 80m, 100m) for the IVT-proxy option below.
  - `precipitation_surface`, `pressure_reduced_to_mean_sea_level` as optional
    context fields.
- NOT in the store: pressure-level winds or humidity. True IVT is impossible
  from this store alone; see "IVT question" below.
- Chunking: (1440, 50, 50), shards (1440, 400, 400). Time-major: one chunk is
  60 days of hours for a 50x50 tile. Reading one hour costs the same chunks as
  reading the full 60 days, which inverts the heat-domes economics: the film is
  nearly free once the first frame is paid for, and event windows that sit
  inside one time chunk are the sweet spot.

### Measured rates (this machine, 2026-08-31)

- Open repo + xr dataset: 1.7 s.
- Event slab, Pacific box (20N-55N, 160W-110W = 141 x 201 px), 576 hours
  (Dec 26 2022 - Jan 18 2023): 65 MB in 2.8 s (23 MB/s), zero NaN.
- Latest single frame, same box: 2.0 s.

Budget at those rates: pwat film for the box ~3 s; add u10/v10 for the proxy,
x3 = ~9 s; the whole 60-day time chunk for the box would be ~7 s per variable.
Boot target of "under 30 s to first playable film" holds with margin.

## Events (all inside the 2021-05 archive floor)

- **Dec 2022 - Jan 2023 California AR train** (nine ARs in ~3 weeks). Time chunk
  boundaries fall at 2021-05-01 + k*60 days, so chunk 10 spans 2022-12-22 to
  2023-02-20: the entire train sits in ONE time chunk per spatial tile. Default
  event window Dec 26 - Jan 18.
- **Feb 2024 Los Angeles** (single sharp Cat 4): compact 48-96 h window.
- **Nov 2021 Pacific Northwest / BC** (Cat 5): global store covers BC fine, no
  HRRR-domain excuse needed.
- **LIVE mode**: window = last N days through the newest hour. This is the
  real-time face of the notebook and should be a first-class mode, not a demo:
  most days it shows a quiet Pacific, and that is an honest picture.

Default: the Dec 2022 train (best film), LIVE as the switch position next to it.

## What we segment (the dome-table analog)

Heat domes segmented pixels by heat-index contour levels, rolled membership to
res 6 parents, dissolved to blobs with area/centroid/track. Here:

1. **The plume**: threshold pwat per frame at a fixed ladder, dissolve member
   cells per (frame, level), same table columns. Ladder for pwat: 20/30/40/50 mm
   (to be tuned on the event; 40+ over the midlatitude Pacific is a strong AR
   core). If the IVT proxy ships, use the canonical 250/500/750/1000 instead.
2. **The shape test** (new vs heat domes): a dome is any big blob; an AR must be
   long and narrow (literature: length > 2000 km, length/width > 2). From each
   dissolved cell set compute a major-axis length and elongation (PCA on cell
   centroids in an equal-area frame is enough) and carry an `is_ar` flag. The
   table becomes an AR catalog: in the train event, AR-1..AR-9 as tracked rows.
3. **The coastal strip**: a one-cell-wide line of H3 cells along the west coast,
   accumulating duration-above-threshold and peak value as the film plays. This
   is the Ralph-scale reading (category = peak x duration at a point) and the
   only hexagonal geometry drawn on the map. Kontur is out (Stephen's call);
   this strip is the impact readout instead.

## Label resolutions

0.25 deg pixels are ~28 km x ~28cos(lat) km, area ~440-770 km2 over the box.
docs/03 safe rule (avg cell area < half pixel area): res 5 (253 km2 avg) is
borderline at 55N, res 6 (36 km2) is safe everywhere. Candidate: labels at
res 5, parents at res 4, and measure uniqueness on the box exactly like the
heat-domes relabel measurement; fall back to 6/5 if res 5 collides at the top
of the box. The dissolve resolution for plumes can sit at the parent res.

## The IVT question

No pressure levels in the store, so three honest options:

- **pwat only.** Simple, one variable, ladder in mm. The captions say
  "precipitable water", never "IVT". Weakest scientifically, cleanest to ship.
- **IVT proxy: pwat x wind.** IVT ~ pwat * V_low-level is a known first-order
  proxy; the store has 10/80/100m winds. 100m is probably the best of the three
  (above the surface layer). Label it "IVT proxy" everywhere. 3x bytes, ~9 s.
- **True IVT from AWS.** Stephen's fallback rule: if not on source/dynamical,
  use AWS. NODD GFS pressure-level GRIB2 on `noaa-gfs-bdp-pds` has the u/q
  stacks. But per-timestep GRIB pulls for 576 hours breaks the real-time
  constraint. Out of scope for this notebook; note it as the follow-up if the
  proxy proves misleading.

Plan of record: ship the proxy (pwat x wind_100m) with pwat as the fallback
field switch. Validate the proxy shape against a published event analysis
(CW3E's Dec 2022 summaries) before trusting the ladder.

## The map

- GFS pwat/proxy painted on the native lonlat mesh (BitmapLayer or the
  SimpleMeshLayer path from heat domes; lonlat quads are axis-aligned in web
  mercator's x, so this is the easy case). Mostly-ocean canvas: low end fades
  to transparent so the river reads as a bright ribbon on the dark basemap.
- Ramp: protan-safe. Blue-to-yellow ladder (moisture as water) or cividis;
  load ramp inferno-style is fine, no red-vs-green anywhere.
- Contour boundaries at the ladder levels on pixel edges, rule select
  (pixel / any / majority / all) carried from RasterFilm unchanged.
- Film: window loader, day ticks, keyboard, fps, click chart with ladder
  levels as dashed lines. Click on a coastal cell shows its event: hours above
  threshold, peak, category.
- AR table under the map: blobs per (frame, level) with area, centroid, axis
  length, elongation, is_ar, track id.
- Coastal strip layer accumulating as above.
- Pick: latLngToCell first, lonlat index snap second (the store's own inverse,
  two divisions), report which hit. County names only apply onshore; offshore
  picks report coordinates and cell id.

## Steps

1. **Chassis copy** from `hrrr-heat-domes.py`: constants, DuckDB connect,
   MirrorStore (new mirror dir keyed on this store's prefix; CHUNK_H becomes
   1440), store cell, window cell, HOLD memo. Drop the LCC math; add the two
   linear lonlat index formulas. Drop counties or keep them for onshore picks.
2. **Window cell**: BOX default (20N-55N, 160W-110W), event presets (train, LA,
   PNW) plus LIVE (last N days to newest hour). Block-wise read like heat domes:
   per variable, per 50x50 store tile intersecting BOX, slice the window hours
   into the (F, N) matrix. At 12 tiles x 3 vars this is small; 8 threads as
   before.
3. **Labels once in Python** from the store's lat/lon (res 5 + res 4 parent),
   uniqueness measured and printed, cached to `proto/cache`-style npy.
4. **RasterFilm carry-over** with the lonlat mesh; field switch
   pwat / proxy / precip; ladder slider; rule select.
5. **AR table**: per (frame, level) dissolve via
   `h3_cells_to_multi_polygon_wkb`, area (equal-area), centroid, PCA axis
   length + elongation, is_ar flag, greedy centroid track across frames.
6. **Coastal strip**: coastline cells from Natural Earth (or the counties'
   coastal boundary) labeled at res 5, exceedance-hours and peak accumulated
   kernel-side, drawn as the one hex layer.
7. **Fly it** headless with `fly.py` siblings; boot time is the number that
   matters, target < 30 s for the train preset, < 10 s for LIVE.

## Design direction (plan only, nothing built)

The widget commits to one dark visual world (the map is the page; a dark ocean
canvas does not theme-flip). Subject vernacular to draw from: hydrographic
charts, river flood gauges, NWS teletype products, CW3E AR-scale graphics.

**Palette** (all protan-safe; no red anywhere, no red-vs-green pairs):

- Ground: abyss blue-black `#0B1420` (hue-biased toward the ramp, not neutral
  near-black).
- Ink: foam `#E8EEF2`; muted ink `#8FA3B0`.
- Field ramp: cividis for pwat/proxy, low end faded to transparent so the ocean
  stays dark and the river reads as a lit ribbon. Sequential rule applies
  (lightness monotonic), which cividis satisfies by construction.
- Contour ladder lines: foam at stepped weights/alpha, not stepped hues; the
  level a line marks is carried by weight and by its gauge position, never by
  hue alone.
- Coastal-strip categories (Ralph Cat 1-5): ordered, so sequential steps drawn
  from the same blue-to-yellow world as the field ramp. First validator pass
  (dataviz skill, dark surface `#0B1420`) confirms CVD separation is fine but
  flags that raw cividis steps read gray at the low end and the two brightest
  steps sit too close for normal vision; the ladder needs re-stepping (fewer,
  farther-apart lightness steps, chroma lifted at the dark end) and the two
  darkest steps need direct labels since they sit under 3:1 against the
  ground. Snap-to-passing at build time; do not eyeball it.
- Accent (interactive chrome, focus, playhead): the ramp's crest yellow
  (`#F5E27A` family). One accent; no second.

**Type** (three roles):

- Display: a chart-italic serif (Newsreader or Spectral italic) for the event
  title and AR names only. Hydrographic charts set water-feature names in
  italic; the rivers get the same treatment. Used with restraint: two or three
  places on the whole page.
- UI/body: IBM Plex Sans.
- Data: IBM Plex Mono for timestamps (UTC), readouts, the AR table, the pick
  HUD. Tabular numerals everywhere digits align.

**Layout**: full-bleed map canvas; a thin instrument rail for the HUD (field
switch, rule select, presets, LIVE); film transport along the bottom with day
ticks; AR catalog table below the fold in mono; click chart as an overlay
panel with crosshair + tooltip and the ladder as dashed lines.

**Signature element** (the one memorable thing; everything else stays quiet):
the legend is a river staff gauge. A vertical striped gauge at the map's edge,
drawn like a flood gauge on a bridge pier, doubling as (a) the color-ramp
legend, (b) the ladder control (drag a stripe boundary to move a threshold),
and (c) a live readout: the current frame's basin max floats on the gauge as a
water line. Atmospheric river, read on a river gauge.

**Motion**: the film is the motion. UI animation is limited to the gauge water
line easing between frames; `prefers-reduced-motion` snaps it.

**Dataviz compliance notes**: one axis everywhere; a table view of the AR
catalog exists by construction; identity never rides on color alone (gauge
position, labels, weight); hover layer on chart and map picks.

## Open questions carried forward

- Proxy fidelity: does pwat x wind_100m rank the nine train ARs the way CW3E's
  IVT analyses did? If not, demote to pwat-only and say so in the notebook.
- Whether the 250/500/750/1000 ladder transfers to the proxy's units at all,
  or the ladder should be event-relative quantiles.
- Track identity through splits/merges (two plumes joining mid-Pacific): greedy
  centroid matching will tangle; acceptable for v1, note where it breaks.
- The stored-label-layer test from docs/02 (write hex5 as a (lat, lon) uint64
  array next to the data) applies here too and is cheaper at this grid size.
- Two-grid landfall inset (HRRR Lambert panel sharing the time cursor) is the
  natural sequel if this notebook lands; out of scope for v1.
