# 12: The S2 hex pair (`s2-lcms-aef-mtbs-pair.py`)

Two maps, one camera. Left is the ground as a picture: Earth Genome's
Sentinel-2 yearly mosaic (true color) as tiles the kernel renders from the
COGs. Right is one H3 fill: what a label product or an embedding says about
the same hexagons. Nothing is stacked on anything. The comparison happens in
the viewer's eye, across the seam, the way the storm fence does it in the
same picture.

This replaces the direction of `xsql-aef-lcms-s2-deck.py` (kept in the repo
as the reference, untracked): seven mutually exclusive paints, an agreement
score per cell, coverage alpha, verdicts, clusters, boundaries, and a
`raster under` checkbox that was the only way to see the imagery next to a
fill. Stephen, 2026-09-01: "any kind of overlay is super confusing most of
the time, and there's no exception here"; "I always want Sentinel-2 zoomed in
by default"; "we don't have any method of changing the year"; "I think we
should just show the raster, not hexified, as S2".

## Axioms

- The imagery is the floor and it is never covered. The left pane draws S2
  and the hover outline of one hexagon, nothing else.
- The right pane draws ONE fill at a time, opaque, picked from a short list.
  No alpha that means something, no coverage scaling, no second layer.
- H3 is the container that makes the two panes the same grid: the fold is
  the H3 UDF inside DataFusion (repo rule), and the outline mirrored on the
  left is the exact cell the fill was folded from.
- Years are a control, not a constant. Only the overlap of the three sources
  is offered: label years 2022 and 2023 (LCMS runs 1999-2023 less 2016, S2
  2022-2024, AEF 2017-2025). The S2 year is its own control (2022, 2023,
  2024) so a fall burn can be looked at in the next year's mosaic while the
  label stays put.
- Protan-safe: the LCMS palette from the previous build (fire orange, no
  reds), viridis for the shift ramp, blue / orange / purple for the "when"
  categorical, light grey for Stable and never.
- No em dashes.

## What is kept from the previous build

- `lc_fold(box, res, year)`: one Albers window of the year's CONUS COG,
  pixel centres to lon/lat, majority code and change share per cell.
- `aef_fold(box, res, year)`: COG overviews per UTM tile (mosaic past
  res 10), mean 64-vector per cell, the dequantised mean.
- The S2 TCI tile renderer (STAC once per z9 ancestor tile, every footprint
  composited, PNG back on the custom-message channel), now keyed by year.
- The zoom -> res ladder, the settle timer, the one-serve-at-a-time loop,
  the memo of folds per (box, res).
- The DeckMap chassis (maplibre + MapboxOverlay interleaved under the
  labels, H3HexagonLayer highPrecision, pick via h3-js at the frame's res).

## What is new

- `PairMap`: one anywidget with two maplibre maps in a row. Camera sync by
  `move` -> `jumpTo` with a guard. One deck overlay per pane: S2 TileLayer +
  hover PathLayer on the left, H3HexagonLayer + hover PathLayer on the right.
- Hover on the right computes the cell with h3-js at the frame's res and
  draws its ring on BOTH panes (gold on the left over the imagery, white on
  the right). Click pins it and asks the kernel for the cell's row.
- Year controls in the pane headers: S2 year on the left, label year on the
  right, plus the fill picker. Keys: `[` `]` step the S2 year, `,` `.` step
  the label year, `1` `2` `3` pick the fill.
- The AEF window is folded per label year and cached per (year, res, box),
  so a label-year step refolds LCMS (one window, ~1 s) and only the AEF
  years not yet held.
- Three fills: `LCMS` (the class), `AEF shift` (the largest step's
  displacement on viridis, stretched to the view), `when` (the first step
  whose displacement is above the view's stable baseline D0: the year the
  embedding moved, or never).

## Not carried

Agreement score and its alpha, coverage scaling, verdict 2x2, difference
clusters, boundaries / fill / hide, the res offset, the S2 index fill, the
LCMS raster tiles, the raster-under machinery, the S2 gain slider, the
analysis panel, the geocoder. Some may come back as a second pass once the
pair reads well on screen.

## Build log

(filled in as the rounds land)

### Round 1 (2026-09-01): the pair stands up

`s2-hex-pair.py` written from the previous build's cells: constants trimmed
to the overlap years, `lc_fold` / `aef_fold` with the year as a real
parameter, the S2 TCI renderer keyed by year (`_png[(year, z, x, y)]`,
STAC per `(year, z9 tile)`), a new `build_frame(lc, aef_by_year, year)`
that returns the three fills and their legends, the `PairMap` widget, and a
wiring cell with one `ctl` channel (`s2`, `label`, `fill`, `labels`). AEF
folds are cached per `(year, res, box)` in `HOLD["aef"]` so a label-year
step folds LCMS again and only the one AEF year the new window adds.

`fly_pair.py`: marimo run headless on port 2734 + playwright, shots to
`shots/pair/`. Steps: boot, hover the right pane (ring on both), click
(panel), the three fills, S2 2023 (tile swap), label 2023 (refold), a drag
on the LEFT map (both cameras must move, then a refold).

**Flight 1 (2026-09-01), home over Dixie, zoom 9.2, two 700 px panes:**
boot 23 s (five AEF index slices already cached under tmp). First fold at
res 8: LCMS 2022 948x971 px at 120 m read 4.1 s, four AEF years (ov4, 320 m,
4 files, 0.09 Mpx each) 1.7-3.7 s concurrent, frame 0.1 s, 11,917 cells,
4.7 s wall. D0 0.153, 3,321 of 11,849 scored cells moved. Hover: the ring
lands on both panes. Pick at the centre: "LCMS 2022: Wildfire, 49% of pixels
changed, shift 0.093 (0.078 / 0.076 / 0.093), did not move". The `when`
legend under label 2022: moved by 2021 21.8%, by 2022 1%, by 2023 5%, did
not move 71.6%. THAT is the composite lag on screen: Dixie burned Jul-Oct
2021, LCMS books it in 2022, the embedding moved in 2021. S2 2023: 32 tiles
served, no fold. Label 2023: 7.8 s, only AEF 2024 folded (the other three
came from the per-year cache). Drag on the left: both cameras moved to the
same centre; the refold took 3.9 s.

Seen in the shots and fixed for flight 2: the right header wrapped at the
pane's width (labels shortened to `S2` / `LCMS` / `class` `shift` `when`,
nowrap, the key hint moved under the legend); numpy's All-NaN warning from
`nanmax` over cells with no embedding (silenced, it is the designed case).

**Flight 2:** clean. Boot 19.4 s, same numbers, headers fit the pane, no
RuntimeWarning, no page errors. Shots in `shots/pair/` (01 home, 02 hover,
03 pick, 04 shift / when, 05 S2 2023, 06 label 2023, 07 dragged).

## Opens

- The right pane at res 8 is coarse beside 10 m imagery at zoom 9.2; the
  ladder is the previous build's with a smaller CELL_BUDGET for the half
  pane. Whether res 9 at home is worth the fold time is a screen judgement.
- Stable / never cells draw faintly (alpha 70) so the grid reads; the axiom
  says opaque. Judge on screen whether the faint sheet helps or muddies.
- The panel line is the only text per cell. A per-cell trace (the three
  steps as a tiny chart) is the natural next piece.
- The `when` fill's blue / orange / purple: check under a protan simulation
  that orange (by Y) and purple (by Y+1) separate at hexagon size.
- A locked mode where stepping the label year also steps the S2 year.
- Wide zoom (below 9) shows two empty basemaps; a coarse S2 overview or a
  note in the panes would help orientation.

### Round 2 (2026-09-01): names, the raster below zoom 9, MTBS

Stephen on the fills: "I don't even understand what shift means." The
header now says what each one is: `LCMS says` / `AEF changed` / `AEF
changed in` / `MTBS burned` (the long form is the button's title and the
status line). The picked cell keeps its color; selection is a gold stroke
on both panes (hover is a thin white one).

Below zoom 9 the right pane draws the label year's LCMS raster (the tile
renderer from the previous build, keyed by year) and the left keeps the
mosaic down to zoom 7: z7-8 tiles are rendered from the pyramid's L5
(306 m) by decimation, STAC per tile instead of per z9 ancestor. A z7 tile
reads up to nine 1024 px windows, so 7 is the floor.

MTBS from Carl Boettiger's `cboettig/fire` on source.coop, two pieces:

- **Perimeters** (`mtbs-perimeters-1984-2024.pmtiles`, layer of the same
  name, z0-14, `Ig_Date` a string): a maplibre vector source on BOTH maps
  through the `pmtiles` protocol (range + CORS verified on
  data.source.coop), two line layers filtered on `slice(Ig_Date, 0, 4)`:
  the label year solid gold, the year before dashed. Key `P`. A click
  inside a perimeter names the fire (queryRenderedFeatures on the line
  layers, carried in the pick message).
- **Severity** (`mtbs-severity-1984-2024-conus/mtbs-severity-conus-<year>-cog.tif`,
  EPSG:4326, 30 m, 83k x 171k px, nine overviews, nodata 0, codes 1-6):
  `mtbs_fold(box, res, year)` is the LCMS fold without the Albers leg
  (the lon/lat box is the window). Folded for the label year AND the year
  before, cached per (year, res, box); a cell burned in both keeps the
  later year. The fill is an orange lightness ramp (unburned-to-low pale,
  high dark), increased greenness teal, unburned faint grey; the legend
  lists class x year.

Why both years: MTBS books the burn year (Dixie: `Ig_Date 2021-07-14`,
979,795 ac), LCMS the year the change shows (2022). Under label 2022 the
`MTBS burned` fill shows Dixie as 2021, and the dashed perimeter is Dixie.

**Flight 3:** boot 17 s. MTBS 2021 836x657 px at 120 m read 2.9 s, fold
5,118 burned cells; the `MTBS burned` legend under label 2022 leads with
"High severity 2021 10.7%". Zoom 8: 88 tiles settled in 4.9 s (LCMS raster
right, S2 from L5 left); zoom 7: 120 tiles in 7.1 s. Both maps reported
the perimeter source and layers mounted, but NO line was visible: deck's
interleaved layers are inserted before `watername_ocean` on every update,
i.e. above line layers that were added before the same slot earlier, so the
S2 tiles and the hexagons painted over the perimeters. Standalone maplibre
+ pmtiles (no deck) drew 104 features at home with `Ig_Date "2021-07-14"`
on Dixie. Fix: deck's `beforeId` is `mtbs-prev` once it exists (per map),
so the stack is basemap, deck, perimeters, labels. Also: the click
hit-test used the line layers (a point inside a polygon does not hit its
outline), now an invisible `mtbs-hit` fill layer. Fullscreen: `⛶` in the
right header or `F`; the panes stretch to the window with the strip kept.

**Flight 4:** clean. Dixie's dashed 2021 perimeter draws on both panes
under label 2022; the click at the centre reads "MTBS: Unburned to low
(2021) · MTBS perimeter: DIXIE (2021-07-14, Wildfire, 979,795 ac)". Style
order on the right: `mtbs-hit, mtbs-prev, mtbs-cur, watername_ocean` with
deck's layers under `mtbs-prev`. Zoom 8 / 7 tile times unchanged (4.9 s /
7.2 s). The fullscreen button collided with the zoom control in the
top-right; the navigation control moved to the bottom-right (not re-flown).

## Opens (round 2)

- A third pane (Stephen asked what it would buy): two fills side by side,
  two S2 years, two label years, or the LCMS raster beside its fold. The
  widget is `mkPane` x N with one camera; three panes are 500 px each.
- The MTBS severity fold uses the label year and the year before; a
  burn-year control of its own is the alternative.
- Hover ring is white on both panes; selection gold on both. Judge on the
  imagery whether white hover reads.

### Round 3 (2026-09-01): the strip in Stephen's dark marimo

Stephen's screenshot in full screen: the strip was grey text on black
("console unreadable, just make it white"). The widget inherited marimo's
dark theme colors. The root, the strip and the pane headers now carry
explicit white backgrounds and dark text whatever the theme. The custom
`⛶` button ("this isn't tidy, maplibre has a built in full screen button
like on all my lonboard notebooks") is gone; maplibre's
`FullscreenControl({container: root})` sits top-right on the right map and
takes the whole widget, both panes and the strip. Key `F` does the same.
Zoom control bottom-right.

Perimeters: the year-before line is no longer dashed ("just make it a
line"); both years are solid gold, the year before slightly thinner (1.6 vs
2.2 px). The click still says which fire and its ignition date.

**Flights 5-6:** the notebook passed both; flight 5's harness failed on
its own check (`document.querySelector` cannot see into anywidget's shadow
DOM; Playwright locators can). Flight 6: strip background rgb(255,255,255),
status text rgb(68,68,68), one `.maplibregl-ctrl-fullscreen` on the right
map, solid perimeters on both panes, zoom 8 / 7 tiles 4.9 s / 8.2 s.

### Round 4 (2026-09-01): fewer buttons, blunter words

- The `MTBS burned` fill is withdrawn ("we don't need a button for that,
  we have other data to look at for that"). `mtbs_fold` and the frame's
  MTBS columns stay in the file; `MTBS_FOLD_YEARS = ()` turns the fold off
  and `FILLS` no longer lists it. Adding it back is those two constants.
- The perimeters carry their own attributes, so hovering one prints the
  fire in the strip with no click: name, ignition date, type, acreage
  (`.sp-fire`, browser-side from `queryRenderedFeatures` on `mtbs-hit`).
- `AEF changed in` -> `AEF change year`; its legend and the cell's row say
  "AlphaEarth changed in 2021 (its 2020 vs 2021 fingerprints)", "AlphaEarth
  saw no change (every step under the quiet level)", "no AlphaEarth
  embedding here".
- The click panel is three plain sentences ("LCMS 2022 says Wildfire (49%
  of the hexagon's pixels). AlphaEarth saw the ground change in 2021. MTBS:
  the DIXIE fire, ignited 2021-07-14, 979,795 acres.") with the shift
  numbers, the threshold, the majority class and the area in a small grey
  line under them.
- A floating tooltip was asked about and deferred ("that might be too much").

**Flight 8:** clean. Hover over Dixie: "DIXIE · ignited 2021-07-14 ·
Wildfire · 979,795 acres (MTBS perimeter)" in the strip, no click. Panel:
"LCMS 2022 says Wildfire (49% of the hexagon's pixels)." then the
AlphaEarth and MTBS sentences, then the grey detail line.

### Round 5 (2026-09-01): the header, designed

Stephen: "the ui for the right panel is still not tidy, too long and
covers the full screen button" (with /frontend-design and /dataviz
invoked). The chrome's job is to stay out of the map's way and never
compete with the data colors, so: paper rgba(255,255,255,.94), ink
#1d1d1b, muted #6b6b68, ONE accent #2a5db0 for the selected state (blue is
protan-safe and no data layer uses it; gold stays the map's). System UI at
12 px with tabular numerals; eyebrow labels (S2, LCMS, FILL) 11 px small
caps muted. Loose buttons became segmented controls (joined, one border).
The right header is two rows (years, then fills) and capped at
calc(100% - 72px) so maplibre's control column is never touched. The two
headers read as one line across the seam: `S2 · 2022 | LCMS · 2022 · FILL ·
LCMS says`. Nothing else added.

**Flights 9-10:** flight 9 showed the right card stretched to its full
allowed width (a flex row-break makes the wrapping container grow to
max-width); the card is now a column of two shrink-wrapped rows. Flight 10
clean, boot 16.8 s. (Flight 10's first launch died with exit 127: the
shell's cwd had drifted into shots/pair from a crop command; the harness
is now called by absolute path.)

Correction: flight 10 flew the OLD header (the column patch had been run
from the drifted cwd and never touched the file; its error went to a
background log). Measured in the DOM: card 482 px, the row-break span 466
px wide. Patch re-applied from the root; flight 11 is the real check.

**Flight 11:** clean, boot 17.7 s. The right card is two shrink-wrapped
rows (`LCMS · 2022 2023` over `FILL · LCMS says · AEF changed · AEF change
year`), clear of maplibre's fullscreen control. Committed at this state.

## State at first commit (2026-09-01)

`s2-lcms-aef-mtbs-pair.py`, `fly_pair.py`, this doc, and the
`async-geotiff` / `pillow` additions to `pyproject.toml` and `uv.lock`.
Left out on purpose: `xsql-aef-lcms-s2-deck.py` (the reference copy from
x-sql-marimo, untracked) and the earlier uncommitted storm-fence edits
(`storm-fence-hex.py`, `docs/11-storm-fence-plan.md`), which are a
different piece of work.
