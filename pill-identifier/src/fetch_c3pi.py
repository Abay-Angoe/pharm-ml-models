"""Download the 20-class subset from the NIH C3PI Training image set.

The C3PI archive is unmaintained (project sunset 2018, Pillbox retired Jan 2021)
and the RxImage API is dead, but the raw file tree is still served:

    https://data.lhncbc.nlm.nih.gov/public/Pills/

Directory listing returns 403; individual files return 200. The bar-delimited
index (`directory_consumer_grade_images.txt`, fields NDC | part | path | type |
name) is public, so we resolve exactly the images we want and fetch only those -
the full consumer-grade set is 2.6 TB.

    python src/fetch_c3pi.py --dry-run
    python src/fetch_c3pi.py
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_TABLE, EXTENSION, IMAGE_TYPE, NDC9_TO_SLUG  # noqa: E402

BASE = "https://data.lhncbc.nlm.nih.gov/public/Pills"
INDEX_NAME = "directory_consumer_grade_images.txt"
UA = "Mozilla/5.0"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "c3pi"
INDEX = ROOT / "data" / "raw" / INDEX_NAME


def get(url: str, retries: int = 4, timeout: int = 180) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - unmaintained host, expect flakiness
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url} failed after {retries} tries: {last}")


def ensure_index() -> list[list[str]]:
    if not INDEX.exists():
        print(f"fetching index ({INDEX_NAME}, ~16 MB) ...")
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_bytes(get(f"{BASE}/{INDEX_NAME}", timeout=600))
    rows = [
        line.rstrip("\n").split("|")
        for line in INDEX.read_text(encoding="utf8", errors="replace").splitlines()
    ]
    return [r for r in rows if len(r) >= 5]


def select(rows: list[list[str]]) -> list[tuple[str, Path]]:
    """Return [(remote_path, local_dest)] for our classes, de-duplicated.

    The index lists some photographs more than once - occasionally under two
    NDC-11 package variants, sometimes twice under the same NDC-11 with differing
    name strings. Counting rows therefore over-counts images: for nine products in
    an earlier selection *every* image was listed twice, inflating their apparent
    counts by exactly 2x. De-duplicate on the remote path so per-class counts are
    real image counts.
    """
    seen: set[str] = set()
    wanted: list[tuple[str, Path]] = []
    for r in rows:
        if r[3] != IMAGE_TYPE:
            continue
        if r[2].rsplit(".", 1)[-1].upper() != EXTENSION:
            continue
        slug = NDC9_TO_SLUG.get(r[0][:9])
        if slug is None or r[2] in seen:
            continue
        seen.add(r[2])
        wanted.append((r[2], RAW / slug / Path(r[2]).name))
    return wanted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = ensure_index()
    print(f"index: {len(rows)} rows")
    wanted = select(rows)
    per_class = collections.Counter(d.parent.name for _, d in wanted)

    missing = [s for _, s, _, _ in CLASS_TABLE if s not in per_class]
    print(f"\nselected {len(wanted)} images across {len(per_class)} classes")
    counts = sorted(per_class.values())
    print(f"images per class: min {counts[0]}, max {counts[-1]}, "
          f"mean {sum(counts) / len(counts):.1f}")
    print(f"  >=100 real images: {sum(1 for v in counts if v >= 100)} classes "
          f"(spec section 2 target; see README for why no C3PI subset clears it)")
    if missing:
        print(f"!! classes with no images found: {missing}")
        return 1
    for slug in sorted(per_class):
        print(f"  {per_class[slug]:4d}  {slug}")
    if args.dry_run:
        return 0

    done = skipped = failed = 0
    t0 = time.time()
    for i, (remote, dest) in enumerate(sorted(wanted), 1):
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{urllib.parse.quote(remote)}"
        try:
            data = get(url)
        except RuntimeError as exc:
            print(f"  FAILED {remote}: {exc}")
            failed += 1
            continue
        if not data.startswith(b"\xff\xd8"):  # JPEG SOI
            print(f"  NOT-JPEG {remote} ({len(data)} bytes) - skipping")
            failed += 1
            continue
        dest.write_bytes(data)
        done += 1
        if done % 100 == 0:
            rate = done / max(1e-9, time.time() - t0)
            remaining = (len(wanted) - skipped - done) / max(rate, 1e-9) / 60
            print(f"  {done} fetched ({rate:.1f}/s, ~{remaining:.0f} min left)",
                  flush=True)
    print(f"\ndone: {done} fetched, {skipped} already present, {failed} failed "
          f"-> {RAW}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
