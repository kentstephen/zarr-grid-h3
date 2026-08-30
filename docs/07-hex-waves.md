# Hex waves: the join drawn as height and colour

Written 2026-08-30. The first version of the cross-grid view that made sense to build
after docs/06 left the question open.

## The idea (Stephen's, this session)

The res 7 label layer carries two datasets that never leave their own grids, one per
visual channel:

- HEIGHT: USFS burn probability (Dillon et al. 2023, FSim, 270 m NAD83 Conus Albers,
  CarbonPlan's icechunk mirror on source.coop). Static.
- COLOUR: Fosberg Fire Weather Index per hour, computed from Dynamical's HRRR analysis
  (2 m T and RH, 10 m wind, 3 km Lambert). Moving.

This sidesteps the docs/06 constraint (one thing per moving map): height never moves,
so time-stepping animates only the colour, and the tween between hours makes fronts
roll across the standing BP terrain as waves.

Decisions made on the way in:
- BP over WHP / CFL / RPS: continuous, and likelihood x weather is one coherent story.
  WHP is an ordinal class (terraces, not waves); CFL is fire behaviour. The 270 m
  stores are tractable (196M px); the 30 m ones (scott-et-al-2024) are not, over the wire.
- CONUS404 is historical (WRF retrospective, 1979-2022); its role stays climatology
  (p95/p99). The live leg is Dynamical's HRRR analysis, hourly to the current hour.
- Viridis for FFWI, fixed 0..60 ramp top (adjustable). Mean/max per cell toggleable,
  one toggle for both channels. Height scale log by default (BP p50 0.0007, max 0.137:
  linear is a flat plain), linear and rank offered.

## What was built

- `join/prep_bp.py` (65 s): BP 270 m -> res 7 cells by projected centre,
  h3ronpy labels, sort+reduceat groupby. 1,522,472 cells, median 73 px each.
  Zeros (non-burnable) stay in the mean; fill (-3.4e38) dropped.
  Writes join/cache/bp_res7.parquet (cell7, bp_mean, bp_max, n).
- `join/prep_ffwi.py` (7 s mirrored, minutes from the wire): T/RH/u/v for a window,
  block-wise through the disk mirror (join/hrrr_mirror.py, MirrorStore copied from
  hrrr-heat-domes.py, same mirror dir), Fosberg 1978 per pixel-hour, uint8 0.5 steps.
  Writes join/cache/ffwi_{d0}_{d1}.npz. Jun 29 - Jul 5 2026: FFWI p50 9, p99 51.
- `join/hexagg.py`: both onto one cell set. Res 6 (default): 210,724 cells, BP
  n-weighted mean / max of res 7 stats, FFWI mean / max over the ~4 HRRR px per cell,
  2.6 s. Res 7: 1,483,904 cells, FFWI by own pixel else gridDisk(1) ring mean, 11 s,
  249 MB frames. Exact H3 boundaries via WKB walk (icosahedron-edge cells carry 7-10
  vertices, so no fixed stride).
- `hex-waves.py`: marimo + anywidget + deck.gl SolidPolygonLayer, binary attributes
  (closed rings + startIndices, per-vertex colour/elevation expanded through a
  vertex->cell map), _normalize false so nothing retessellates on a frame change;
  double-buffered colour arrays so only 5.9 MB re-uploads per paint. Play tweens
  fractional hours. HUD: mean/max, height scale + exaggeration, ramp top, opacity,
  base tiles, pick. Paint 5 ms at res 6 on the real GPU.
- `fly_hex.py`: headless flight, shots/hex_*.png. Chromium must run with its GPU
  defaults; the swiftshader flags jam the page.

## Where it stands

Flown at res 6: the surface stands, the west carries the structure (Sierra foothills,
Great Basin ranges, central Texas), colour reads. Not yet judged by eye at res 7, not
yet watched in motion by a person. Open:
- Is the wave legible in motion, or does the pitch hide the east behind the west?
- Res 7 in the browser: 8.9M vertices; binary path is ready, untested for feel.
- The live edge: prep_ffwi can take today's dates (store runs to the current hour);
  the youngest chunk is never mirrored, so it reads from the wire.
- The raster underneath (docs/06's HRRR mesh) is omitted; hexes are the view.
