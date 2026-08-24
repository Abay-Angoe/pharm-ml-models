"""Fetch a 30-class subset of OGYEIv2 from Kaggle.

No Kaggle credentials are required - the archive is served over plain HTTP with
Range support (verified 2026-08-04).

Two strategies, because measured throughput decides it:
  * bulk (default) - stream the whole 3.5 GB archive once at ~13.5 MB/s (~4 min)
    and extract only the members we want. Optionally delete the archive after.
  * --ranges - pull each member with its own Range request (~0.9 GB but two
    requests per file at ~0.9 s latency each, so ~37 min). Kept for the case
    where disk is too tight to stage the full archive.

Each image has a sibling .txt holding a YOLO *segmentation polygon* of the
pill outline, which preprocess.py uses to crop exactly.

    python src/fetch_subset.py             # bulk download + extract
    python src/fetch_subset.py --dry-run   # report what would be fetched
    python src/fetch_subset.py --ranges    # low-disk path
    python src/fetch_subset.py --keep-archive
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/richardradli/ogyeiv2"
UA = "Mozilla/5.0"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ogyeiv2"

# 30 classes chosen as 12 groups of near-identical pills plus one distinct
# control. Within a group, colour and shape are close to identical and often
# only the imprint separates them - which is what makes top-5 (section 7) and
# the Grad-CAM imprint check (section 8) actually informative.
CLASS_GROUPS: dict[str, list[str]] = {
    "ibuprofen_400": [
        "algoflex_forte_dolo_400_mg",
        "algoflex_rapid_400_mg",
        "ibumax_400_mg",
        "nurofen_forte_400_mg",
    ],
    "proton_pump_inhibitors": [
        "controloc_20_mg",
        "nolpaza_20_mg",
        "pantoprazol_sandoz_40_mg",
        "xeter_20_mg",
    ],
    "diclofenac": ["cataflam_50_mg", "cataflam_dolo_25_mg", "cataflam_v_50_mg"],
    "atorvastatin": ["atoris_20_mg", "atorvastatin_teva_20_mg"],
    "bisoprolol_dose_pair": ["concor_5_mg", "concor_10_mg"],
    "alprazolam_dose_pair": ["frontin_0_25_mg", "frontin_0_5_mg"],
    "doxazosin": ["doxazosin_hexal_4_mg", "doxazosin_sandoz_uro_4_mg"],
    "duloxetine": ["dulodet_60_mg", "dulsevia_60_mg"],
    "ramipril": ["tritace_5_mg", "tritace_hct_5_mg_25_mg"],
    "b_vitamin": ["milgamma", "milgamma_n"],
    "levothyroxine": ["l_thyroxin_henning_50_mikrogramm", "letrox_50_mikrogramm"],
    "cetirizine": ["cetirizin_10_mg", "revicet_akut_10_mg"],
    "distinct_control": ["aspirin_ultra_500_mg"],
}
CLASSES: list[str] = sorted(c for g in CLASS_GROUPS.values() for c in g)


def _fetch(byte_range: str, retries: int = 4) -> tuple[bytes, str | None]:
    """GET a byte range, retrying on transient failures."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={byte_range}", "User-Agent": UA}
            )
            with urllib.request.urlopen(req, timeout=240) as resp:
                return resp.read(), resp.headers.get("Content-Range")
        except Exception as exc:  # noqa: BLE001 - network is genuinely flaky here
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {byte_range} failed after {retries} tries: {last}")


def _zip64_fix(csize: int, usize: int, lho: int, extra: bytes) -> tuple[int, int, int]:
    """Resolve 0xFFFFFFFF sentinels from the zip64 extra field (id 0x0001)."""
    if 0xFFFFFFFF not in (csize, usize, lho):
        return csize, usize, lho
    pos = 0
    while pos + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[pos : pos + 4])
        body, off = extra[pos + 4 : pos + 4 + hsz], 0
        if hid == 1:
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", body[off : off + 8])[0]
                off += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[off : off + 8])[0]
                off += 8
            if lho == 0xFFFFFFFF:
                lho = struct.unpack("<Q", body[off : off + 8])[0]
        pos += 4 + hsz
    return csize, usize, lho


def read_central_directory() -> dict[str, tuple[int, int, int]]:
    """Return {member_name: (compress_method, compressed_size, local_offset)}."""
    _, content_range = _fetch("0-0")
    if not content_range:
        raise RuntimeError("server did not honour Range requests")
    total = int(content_range.split("/")[-1])
    print(f"archive: {total / 1e9:.3f} GB")

    tail, _ = _fetch(f"{total - 100}-{total - 1}")
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("could not locate end-of-central-directory record")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    cd, _ = _fetch(f"{cd_off}-{cd_off + cd_size - 1}")

    entries: dict[str, tuple[int, int, int]] = {}
    pos = 0
    while pos < len(cd) - 4 and cd[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        csize, usize = struct.unpack("<II", cd[pos + 20 : pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        lho = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf8", "replace")
        extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
        csize, usize, lho = _zip64_fix(csize, usize, lho, extra)
        entries[name] = (method, csize, lho)
        pos += 46 + nlen + elen + clen
    print(f"central directory: {len(entries)} members")
    return entries


def extract(name: str, method: int, csize: int, lho: int) -> bytes:
    header, _ = _fetch(f"{lho}-{lho + 29}")
    nlen, elen = struct.unpack("<HH", header[26:30])
    start = lho + 30 + nlen + elen
    blob, _ = _fetch(f"{start}-{start + csize - 1}")
    if method == 0:
        return blob
    if method == 8:
        return zlib.decompress(blob, -15)
    raise RuntimeError(f"{name}: unsupported compression method {method}")


def class_of(stem: str) -> str:
    """'concor_5_mg_s_012' -> 'concor_5_mg' (strip lighting + index suffix)."""
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 and parts[1] in ("s", "u") else stem


def download_archive(dest: Path) -> Path:
    """Stream the whole archive to disk, resuming if a partial file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _, content_range = _fetch("0-0")
    total = int(content_range.split("/")[-1])
    have = dest.stat().st_size if dest.exists() else 0
    if have >= total:
        print(f"archive already complete: {dest} ({have / 1e9:.3f} GB)")
        return dest
    if have:
        print(f"resuming at {have / 1e6:.0f} MB of {total / 1e6:.0f} MB")

    start = time.time()
    req = urllib.request.Request(
        URL, headers={"Range": f"bytes={have}-{total - 1}", "User-Agent": UA}
    )
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "ab") as fh:
        got = have
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            if got % (200 << 20) < (1 << 20):
                rate = (got - have) / max(1e-9, time.time() - start) / 1e6
                print(f"  {got / 1e6:.0f}/{total / 1e6:.0f} MB "
                      f"({rate:.1f} MB/s)", flush=True)
    final = dest.stat().st_size
    if final != total:
        raise RuntimeError(f"size mismatch: got {final}, expected {total}")
    print(f"archive downloaded: {final / 1e9:.3f} GB in "
          f"{time.time() - start:.0f}s")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ranges", action="store_true",
                    help="per-member Range requests instead of bulk download")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the staged 3.5 GB zip after extracting")
    args = ap.parse_args()

    entries = read_central_directory()

    wanted: list[tuple[str, Path]] = []
    for name in entries:
        parts = name.split("/")
        if len(parts) < 5 or parts[3] not in ("images", "labels"):
            continue
        if not name.lower().endswith((".jpg", ".txt")):
            continue
        stem = Path(name).stem
        cls = class_of(stem)
        if cls not in CLASSES:
            continue
        # data/raw/ogyeiv2/<split>/<images|labels>/<class>/<file>
        wanted.append((name, RAW / parts[2] / parts[3] / cls / Path(name).name))

    imgs = [w for w in wanted if w[0].endswith(".jpg")]
    per_split = Counter(w[0].split("/")[2] for w in imgs)
    per_class = Counter(class_of(Path(w[0]).stem) for w in imgs)

    print(f"\nselected {len(CLASSES)} classes -> {len(imgs)} images "
          f"+ {len(wanted) - len(imgs)} label files")
    print(f"per split: {dict(per_split)}")
    counts = set(per_class.values())
    print(f"images per class: {sorted(counts)} "
          f"({'balanced' if len(counts) == 1 else 'UNEVEN'})")
    missing = sorted(set(CLASSES) - set(per_class))
    if missing:
        print(f"!! classes not found in archive: {missing}")
        return 1
    print(f"estimated download: "
          f"{sum(entries[n][1] for n, _ in wanted) / 1e6:.0f} MB")

    if args.dry_run:
        return 0

    todo = [(n, d) for n, d in sorted(wanted)
            if not (d.exists() and d.stat().st_size > 0)]
    skipped = len(wanted) - len(todo)
    if not todo:
        print(f"\nall {len(wanted)} members already present -> {RAW}")
        return 0

    if args.ranges:
        for i, (name, dest) in enumerate(todo, 1):
            method, csize, lho = entries[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(extract(name, method, csize, lho))
            if i % 100 == 0:
                print(f"  fetched {i}/{len(todo)} ...", flush=True)
    else:
        import zipfile

        archive = ROOT / "data" / "raw" / "ogyeiv2_full.zip"
        download_archive(archive)
        print(f"extracting {len(todo)} members ...")
        with zipfile.ZipFile(archive) as zf:
            for i, (name, dest) in enumerate(todo, 1):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src:
                    dest.write_bytes(src.read())
                if i % 400 == 0:
                    print(f"  extracted {i}/{len(todo)} ...", flush=True)
        if not args.keep_archive:
            archive.unlink()
            print("removed staged archive (use --keep-archive to retain)")

    print(f"\ndone: {len(todo)} fetched, {skipped} already present -> {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
