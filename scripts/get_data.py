"""Download the open data Reach120 reads, into the places config/settings.yaml points to.

    python scripts/get_data.py --dry-run                   # what is missing, and how big it is
    python scripts/get_data.py                             # fetch what is missing (about 9.5 GB)
    python scripts/get_data.py --skip ontime_travel_times  # leave out the 713 MB validation file

National layers go into the shared data folder: `paths.shared_data` in config/settings.yaml
(a folder named 00-shared-data next to this repo), or $REACH120_SHARED_DATA if it is set. The
OnTIME files go into data/raw/. Re-running skips what is already there and resumes broken
downloads where the server allows it. A download is refused if it would leave less than
--min-free-gb free on the drive.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reach120.config import project_path, settings, shared_path  # noqa: E402

WORLDPOP = "https://data.worldpop.org/repo/wopr/NGA/population/v3.0"
WORLDPOP_FTP = "ftp://ftp.worldpop.org/repo/wopr/NGA/population/v3.0"
FIGSHARE = "https://ndownloader.figshare.com/files"
COLUMBIA = "https://academiccommons.columbia.edu/doi/10.7916"

session = requests.Session()
session.headers["User-Agent"] = "reach120-get-data/1.0 (python-requests)"


def datasets() -> list[dict]:
    """Every file the pipeline reads: where it goes, where it comes from, and its size in MB."""
    p = settings()["paths"]
    ontime = project_path(p["ontime_facilities"]).parent
    return [
        dict(key="facilities", path=shared_path(p["facilities"]), mb=24, licence="CC BY 4.0",
             hub=("9df52e12d0b5422f8f73c44d0dc91f8b", "csv"), what="GRID3 health facilities v4.0"),
        dict(key="states", path=shared_path(p["states"]), mb=1, licence="CC BY 4.0",
             hub=("c41532b720504f4799fe20438b7e3b7f", "geoPackage"), what="GRID3 state boundaries"),
        dict(key="wards", path=shared_path(p["wards"]), mb=243, licence="CC BY-SA 4.0",
             hub=("74cf76766bec4481aa03523e0e1fdddc", "geoPackage"), what="GRID3 ward boundaries v4.0"),
        dict(key="friction_mixed", path=shared_path(p["friction"]["mixed"]["zip"]), mb=3225, licence="CC BY-SA 4.0",
             url=f"{COLUMBIA}/zvnx-2v35/download", what="GRID3 friction surface, vehicles and walking (kept zipped)"),
        dict(key="friction_walk", path=shared_path(p["friction"]["walk"]["zip"]), mb=3367, licence="CC BY-SA 4.0",
             url=f"{COLUMBIA}/97nc-wt39/download", what="GRID3 friction surface, walking only (kept zipped)"),
        dict(key="women", path=shared_path(p["women"]), mb=1841, licence="CC BY 4.0",
             archive=f"{WORLDPOP}/NGA_population_v3_0_agesex.zip", ftp=f"{WORLDPOP_FTP}/NGA_population_v3_0_agesex.zip",
             what="WorldPop women aged 15-49 (one layer of the age-sex archive)"),
        dict(key="population_total", path=shared_path(p["population_total"]), mb=46, licence="CC BY 4.0",
             archive=f"{WORLDPOP}/NGA_population_v3_0_gridded.zip", what="WorldPop total population"),
        dict(key="state_totals", path=shared_path("worldpop/total/NGA_population_v3_0_admin.zip"), mb=0.05,
             licence="CC BY 4.0", url=f"{WORLDPOP}/NGA_population_v3_0_admin.zip",
             what="WorldPop state totals (a cross-check only)"),
        dict(key="ontime_facilities", path=project_path(p["ontime_facilities"]), mb=0.5, licence="CC BY 4.0",
             url=f"{FIGSHARE}/41570451", what="OnTIME verified C-section facilities"),
        dict(key="ontime_dictionary", path=ontime / "Data_dictionary.csv", mb=0.01, licence="CC BY 4.0",
             url=f"{FIGSHARE}/41570448", what="OnTIME data dictionary"),
        dict(key="ontime_travel_times", path=project_path(p["ontime_travel_times"]), mb=713, licence="CC BY 4.0",
             url=f"{FIGSHARE}/40721504", what="OnTIME Google travel times (validation only)"),
    ]


def size(mb: float) -> str:
    return f"{mb / 1e3:.1f} GB" if mb >= 1000 else f"{max(mb, 1):.0f} MB"


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def check_room(d: dict, floor_gb: float) -> None:
    need_mb = d["mb"] + (50 if d["path"].suffix == ".tif" else 0)  # a WorldPop layer is extracted next to its zip
    free = free_gb(d["path"])
    if free - need_mb / 1e3 < floor_gb:
        sys.exit(f"stopped before {d['key']}: it needs about {size(need_mb)} and {free:.1f} GB is free, "
                 f"which would leave less than {floor_gb:g} GB. Free some space or lower --min-free-gb.")


def fetch(url: str, dest: Path, tries: int = 8) -> None:
    """Stream url into dest through a .part file, resuming after a dropped connection if the server allows."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with session.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True,
                             timeout=(30, 300)) as r:
                if r.status_code == 416 and have:  # nothing left to fetch
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # the server ignored the range: start again
                    have = 0
                total, rng = None, r.headers.get("Content-Range", "")
                if rng.rpartition("/")[2].isdigit():
                    total = int(rng.rpartition("/")[2])
                elif r.headers.get("Content-Length", "").isdigit() and not r.headers.get("Content-Encoding"):
                    total = have + int(r.headers["Content-Length"])
                done, report = have, time.time() + 60
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() > report:
                            print(f"    {done / 1e6:,.0f} MB" + (f" of {total / 1e6:,.0f}" if total else ""), flush=True)
                            report = time.time() + 60
                if total is not None and done < total:
                    raise OSError(f"connection closed after {done:,} of {total:,} bytes")
            break
        except (requests.RequestException, OSError) as e:
            if attempt == tries:
                raise
            print(f"    {type(e).__name__}: retrying ({attempt}/{tries - 1})", flush=True)
            time.sleep(min(15 * attempt, 90))
    part.replace(dest)


def fetch_ftp(url: str, dest: Path) -> None:
    """WorldPop's web server cannot resume a broken download; its FTP mirror can, through curl."""
    part = dest.with_name(dest.name + ".part")
    subprocess.run(["curl", "-sS", "-f", "-C", "-", "--retry", "200", "--retry-delay", "20", "--retry-all-errors",
                    "-o", str(part), url], check=True)
    part.replace(dest)


def hub_export(item: str, fmt: str, minutes: int = 40) -> str:
    """ArcGIS Hub builds an export on request; poll until its file URL is ready."""
    api = f"https://hub.arcgis.com/api/download/v1/items/{item}/{fmt}"
    for _ in range(minutes * 4):
        j = session.get(api, params={"redirect": "false", "layers": 0}, timeout=120).json()
        if j.get("resultUrl"):
            return j["resultUrl"]
        time.sleep(15)
    raise OSError(f"ArcGIS Hub had not finished exporting {item} after {minutes} minutes")


def extract(archive: Path, dest: Path) -> None:
    """Copy the one layer the pipeline reads out of a WorldPop archive."""
    with zipfile.ZipFile(archive) as z:
        info = next((i for i in z.infolist() if Path(i.filename).name == dest.name), None)
        if info is None:
            raise OSError(f"{dest.name} is not in {archive.name}")
        part = dest.with_name(dest.name + ".part")
        with z.open(info) as src, open(part, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 22)
    part.replace(dest)


def get(d: dict, keep_archives: bool) -> None:
    dest = d["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if "hub" in d:
        print("    waiting for ArcGIS Hub to prepare the export", flush=True)
        fetch(hub_export(*d["hub"]), dest)
    elif "archive" in d:
        archive = dest.parent / d["archive"].rsplit("/", 1)[1]
        if archive.exists():
            pass
        elif d.get("ftp") and shutil.which("curl"):
            fetch_ftp(d["ftp"], archive)
        else:
            fetch(d["archive"], archive)
        extract(archive, dest)
        if not keep_archives:
            archive.unlink()
    else:
        fetch(d["url"], dest)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what is missing and download nothing")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="names from the list to leave out")
    ap.add_argument("--keep-archives", action="store_true", help="keep the WorldPop zips after extracting the layer used")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="never leave less than this free (default 2)")
    args = ap.parse_args(argv)

    items = [d for d in datasets() if d["key"] not in args.skip]
    missing = [d for d in items if not d["path"].exists()]
    for d in items:
        status = "have" if d["path"].exists() else f"get {size(d['mb']):>7}"
        print(f"  {status:12} {d['key']:20} {d['what']} ({d['licence']})")
    total = sum(d["mb"] for d in missing)
    print(f"\n{len(missing)} of {len(items)} missing" + (f", about {size(total)} to download." if missing else "."))
    print(f"National layers: {shared_path()}\nOnTIME files:    {project_path('data', 'raw')}")
    if args.dry_run or not missing:
        return
    for d in missing:
        check_room(d, args.min_free_gb)
        print(f"\n{d['key']}: {d['what']}", flush=True)
        get(d, args.keep_archives)
        print(f"    saved {d['path']}", flush=True)
    print("\nDone. Next: reach120 all --state Kano")


if __name__ == "__main__":
    main()
