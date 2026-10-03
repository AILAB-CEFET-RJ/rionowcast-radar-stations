"""Download resumable GOES-16 ABI CMIPF files from the public NOAA S3 bucket."""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from nowcasting.goes16 import load_goes_config, parse_goes_filename_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa cenas GOES-16 ABI-L2-CMIPF com manifesto auditável.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    args = parse_args()
    if args.end_date < args.start_date:
        raise ValueError("--end-date deve ser igual ou posterior a --start-date.")
    try:
        import s3fs
    except ImportError as error:
        raise RuntimeError("Download GOES requer s3fs; instale requirements.txt.") from error
    config = load_goes_config(args.config)
    source = config["source"]
    filesystem = s3fs.S3FileSystem(anon=True)
    manifest_path = args.output_root / "download_manifest.jsonl"
    args.output_root.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failures = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for current in iter_days(args.start_date, args.end_date):
            year, day = current.year, current.timetuple().tm_yday
            print(f"[{current.isoformat()}] listando cenas ABI...", flush=True)
            for hour in range(24):
                prefix = f"{source['bucket']}/{source['product']}/{year}/{day:03d}/{hour:02d}"
                try:
                    objects = filesystem.ls(prefix, detail=False)
                except Exception as error:
                    manifest.write(json.dumps({"prefix": prefix, "status": "list_error", "error": str(error)}, ensure_ascii=False) + "\n")
                    failures += 1
                    continue
                for remote in objects:
                    try:
                        channel, start, end = parse_goes_filename_bounds(remote)
                    except ValueError:
                        continue
                    if channel not in config["channels"]:
                        continue
                    destination = (
                        args.output_root / f"year={start.year}" / f"channel={channel}" /
                        f"month={start.month:02d}" / f"day={start.day:02d}" / Path(remote).name
                    )
                    record = {"remote": remote, "local": str(destination), "channel": channel, "scene_start_utc": start.isoformat(), "scene_end_utc": end.isoformat()}
                    if destination.exists() and not args.overwrite:
                        manifest.write(json.dumps({**record, "status": "skipped_existing"}, ensure_ascii=False) + "\n")
                        skipped += 1
                        continue
                    try:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        filesystem.get(remote, str(destination))
                        manifest.write(json.dumps({**record, "status": "downloaded", "bytes": destination.stat().st_size}, ensure_ascii=False) + "\n")
                        downloaded += 1
                    except Exception as error:
                        manifest.write(json.dumps({**record, "status": "download_error", "error": str(error)}, ensure_ascii=False) + "\n")
                        failures += 1
    print(f"Download concluído | baixadas={downloaded} | existentes={skipped} | falhas={failures} | manifesto={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
