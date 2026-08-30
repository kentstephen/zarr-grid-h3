# Showing the cross-grid join: open, not solved

Written 2026-08-30, end of a session that did not resolve it.

## Where it stands

The join in `join/prep_conus404.py` works and is measured (docs/05): every HRRR land
pixel paired with one real CONUS404 pixel through the res 7 label, 96.8% identical to
the metric nearest, worst case 1.7 km. That result stands.

What has not been found is a way to show it that makes sense to look at.

## What was tried and did not land

- The static join notebook (docs/05, views A / B / C). A lookup table has no picture:
  view B looks the same as view A by design, view C is a national speckle of direct
  versus ring that only means something once you already understand the mechanism.
  Dropped.
- Pixel-scale drawings (`join/draw_join.py`, `shots/join_*.png`): two grids, hexagons,
  arrows from each HRRR centre to its partner, and A / B side by side for a 36 km
  window. Not helpful.
- A univariate "above this place's p99" map on HRRR pixels, where the threshold arrived
  by the join and the viewer never sees CONUS404's grid. Described only.
- A bivariate map (absolute FFWI x local percentile class). Rejected: not a bivariate
  map, ever.

## What the join is for, in one line

Arithmetic between two datasets that stay on their own grids: one HRRR number and one
CONUS404 number in the same row, per pixel, with nothing resampled. Without a
comparison there is no reason to join. Any visualization has to be of the comparison,
not of the pairing.

## Open

Something else is needed. Not the plumbing, not bivariate. Directions not yet
explored: two synchronized single-variable maps with a shared pick; a time series at
a picked pixel (HRRR FFWI hours against the CONUS404 p95 / p99 lines); a table or
ranking rather than a map (counties or res 6 cells by hours above local p99); the join
used for something other than Fosberg entirely (burn probability, ASOS points).
Nothing chosen.

Option 3 from docs/05 (ratio FFWI / local p99) carries a cross-model caveat: HRRR and
WRF CONUS404 wind biases differ and do not cancel in the ratio, and CarbonPlan's
Fosberg formula and units must be matched before any comparison is made.

## Prior attempts

This is not the first try. Stephen worked on the same problem in a separate Fosberg
project on this machine and never got it to work. The diagnosis from that experience:
it is hard to convey multiple things in a moving map. Time-stepping already carries
one dimension; asking a second variable to ride along with it is where every version
has broken. Whatever comes next should treat that as the constraint: one thing per
moving map, or a still map when there are two.
