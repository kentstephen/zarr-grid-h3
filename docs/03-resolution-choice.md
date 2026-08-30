# Choosing H3 resolutions against pixel size

Reference numbers for picking the label resolution r_L (unique cell per pixel) and a traversal resolution r_T. Average values from the H3 v4 statistics table; the max/min area ratio across the globe at any resolution is about 1.99, so "safe" here means average cell area below roughly half the pixel area.

| pixel size | pixel area | r_L candidate (avg area < pixel) | r_L safe (avg area < 0.5 x pixel) | notes |
|-----------|-----------:|:-------------------------------:|:--------------------------------:|-------|
| 0.25 deg (~27.8 km eq.) | ~772 km2 | 5 (252.9 km2) | 5 | ERA5-class; pixels shrink toward poles, so check the minimum pixel width in the grid, not the equatorial one |
| 0.1 deg (~11.1 km eq.) | ~123 km2 | 6 (36.1 km2) | 6 | |
| 1 km | 1 km2 | 8 (0.737 km2) | 9 (0.105 km2) | MODIS-class |
| 500 m | 0.25 km2 | 9 | 9 | |
| 250 m | 62,500 m2 | 10 (15,048 m2) | 10 | |
| 100 m | 10,000 m2 | 11 (2,150 m2) | 11 | |
| 30 m | 900 m2 | 12 (307 m2) | 12 | Landsat |
| 10 m | 100 m2 | 13 (43.9 m2) | 13 | Sentinel-2; r_13 max area is ~2x avg so borderline on some faces, r_14 is unambiguous |
| 3 m | 9 m2 | 14 (6.27 m2) | 15 (0.895 m2) | Planet |
| 1 m | 1 m2 | 15 | none safe | r_15 is the floor; below ~1.3 m pixels labels are not guaranteed unique |

Approximate H3 stats used (v4, average):

| res | avg edge (m) | avg area | approx cells per 1 km2 |
|----:|-------------:|---------:|----------------------:|
| 5 | 8,544 | 252.9 km2 | 0.004 |
| 6 | 3,229 | 36.13 km2 | 0.028 |
| 7 | 1,220 | 5.161 km2 | 0.19 |
| 8 | 531 | 0.737 km2 | 1.36 |
| 9 | 201 | 105,333 m2 | 9.5 |
| 10 | 75.9 | 15,048 m2 | 66 |
| 11 | 28.7 | 2,150 m2 | 465 |
| 12 | 10.8 | 307 m2 | 3,256 |
| 13 | 4.09 | 43.9 m2 | 22,800 |
| 14 | 1.55 | 6.27 m2 | 159,600 |
| 15 | 0.58 | 0.895 m2 | 1,117,000 |

Rules of thumb that fall out:

- Each H3 level is a factor of 7 in area. Raster overview levels are a factor of 4. Two raster overview levels (16x) are close to one and a half H3 levels; there is no exact alignment, which matters for direction 7 in docs/02.
- For traversal, r_T = r_L - 2 gives roughly 49 label cells per r_T cell, so on the order of 20 to 50 pixels per traversal cell depending on how many r_L cells were empty. r_T = r_L - 3 gives ~343.
- For a chunk index, pick r_C so that a chunk contains tens to a few hundred cells. A 1024 x 1024 chunk of 10 m pixels is ~105 km2, so r_C = 7 (5.16 km2) gives ~20 cells, r_C = 8 gives ~140.
- h3ronpy's `nearest_h3_resolution(..., search_mode="smaller_than_pixel")` implements the "r_L candidate" column, not the "safe" column. raster2dggs offers both `smaller-than-pixel` and `larger-than-pixel`.

Sources: https://h3geo.org/docs/core-library/restable/ , https://h3ronpy.readthedocs.io/en/stable/usage/raster.html , https://github.com/manaakiwhenua/raster2dggs
