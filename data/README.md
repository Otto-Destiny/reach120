# Reach120 data

No data is committed. One script downloads every file the pipeline reads, about 9.5 GB in all,
and skips anything already there:

```bash
python scripts/get_data.py --dry-run   # what is missing, and how big it is
python scripts/get_data.py             # download it; safe to re-run, resumes broken downloads
```

## Where the files go

The national layers go into a shared folder outside the repo, so several projects can use one
copy: `../00-shared-data` by default (`paths.shared_data` in `config/settings.yaml`), or the
folder named by `$REACH120_SHARED_DATA`.

| Dataset | Path in the shared folder | Download | Licence | Used for |
|---|---|---|---|---|
| [GRID3 health facilities v4.0](https://www.arcgis.com/home/item.html?id=9df52e12d0b5422f8f73c44d0dc91f8b) | `grid3/health-facilities/` | 24 MB | CC BY 4.0 | Capable hospitals and upgrade candidates |
| [GRID3 operational boundaries](https://www.arcgis.com/home/item.html?id=74cf76766bec4481aa03523e0e1fdddc): states, wards v4.0 | `grid3/boundaries/` | 244 MB | CC BY 4.0 (states), CC BY-SA 4.0 (wards) | Reporting by ward, LGA and state |
| [GRID3 travel-time friction surface v1.0](https://www.arcgis.com/home/item.html?id=fdddb08abd19493a8d4dc8158fd5fd75), vehicles and walking | `grid3/travel-time-friction/` | 6.6 GB, kept zipped | CC BY-SA 4.0 | Travel time |
| [WorldPop/GRID3 population v3.0](https://data.worldpop.org/repo/wopr/NGA/population/v3.0/), women aged 15-49 | `worldpop/agesex/` | 1.8 GB archive; the 50 MB layer is kept | CC BY 4.0 | Women aged 15-49 |
| WorldPop/GRID3 population v3.0, total and state totals | `worldpop/total/` | 46 MB | CC BY 4.0 | Context and a cross-check |

The two OnTIME datasets go into this folder:

| Dataset | Path | Download | Licence | Used for |
|---|---|---|---|---|
| [OnTIME verified C-section facilities](https://doi.org/10.6084/m9.figshare.22689667.v2) | `raw/ontime-cs-facilities/` | 1 MB | CC BY 4.0 | Scoring the hospital-type stand-in against ground truth |
| [OnTIME travel times](https://doi.org/10.6084/m9.figshare.22699759.v1) | `raw/ontime-travel-times/` | 713 MB | CC BY 4.0 | Validating modelled travel times against Google's |

The friction surfaces stay zipped: each zip holds one 6.39 GB uncompressed GeoTIFF (44,466 ×
35,733 cells of 30 m, UTM zone 32N, minutes per metre), and Reach120 reads just the tiles it
needs straight from the zip.

## What the verified list holds

2,021 facilities in 15 city regions, all geocoded, all verified for comprehensive emergency
obstetric care. 1,900 are private and 121 public. Kano has 145.

Codes in the list (full dictionary in `Data_dictionary.csv`):

| Field | Codes |
|---|---|
| `urban_conurb` | 1 Lagos, 2 Aba, 3 Abuja, 4 Benin City, 5 Ibadan, 6 Ilorin, 7 Jos, 8 Kaduna, 9 Kano, 10 Maiduguri, 11 Onitsha, 12 Owerri, 13 Port Harcourt, 14 Uyo, 15 Warri |
| `owner` | 1 public, 2 private |
| `facility_level` | 0 primary, 1 secondary, 2 tertiary |

## Facility facts worth knowing (GRID3 v4.0)

- 70,235 facilities in 37 states.
- Hospital-type facilities: Hospital 6,031 · Medical Center 1,482 · General Hospital 969 ·
  Specialist Hospital 911 · Cottage Hospital 241 · Federal Medical Center 54 ·
  Teaching/Tertiary Hospital 33.
- `functional`: Functional 35,434 · Unknown 29,945 · blank 4,551 · Not-Functional 265 · Closed 38.
  Among hospital-type facilities, 9,137 of 9,736 are marked Functional.
- Kano: 2,122 facilities, including 41 General, 6 Teaching/Tertiary, 15 Specialist and 71 plain
  "Hospital". Jigawa: 1,073 facilities, including 8 General and 9 plain "Hospital".

## What the build found in these data

- **Facility records without coordinates:** 264 of Kano's 2,122 GRID3 v4.0 records have none and
  cannot be placed; 2 to 41 of them are hospitals, depending on the capability definition.
- **Coordinates disagree between sources:** the same hospital is often 0.5-4 km apart in OnTIME
  and GRID3, so the scorecard accepts strong name matches up to 1.5 km apart.
- **GRID3 has duplicate records** of some hospitals; the scorecard counts matches in both
  directions so duplicates do not count against a definition.
- **Verified C-section facilities are mostly typed as something other than "General Hospital"**
  in GRID3 ("Hospital" 675, "Medical Center" 159, "Specialist Hospital" 122, "Clinic" 98,
  "Maternity Home" 76, "General Hospital" 65).
- **Rivers are barriers in the friction surface,** which isolates small pockets with travel times
  in the thousands of minutes; Reach120 takes medians onto the population grid so they do not
  distort the counts.

## Not scripted

| Dataset | Why | How to get it |
|---|---|---|
| Nigeria DHS 2023-24, with the GPS file (optional) | Released only to registered users | Register at dhsprogram.com, describe the project, request the Nigeria datasets, and save them under `raw/dhs-2023-24/` |
