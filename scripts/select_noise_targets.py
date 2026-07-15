#!/usr/bin/env python3
"""Select a reproducible TESS-noise candidate pool from Gaia RV standards."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


CATALOG = "J/A+A/616/A7/rvstdcat"
CATALOG_DOI = "10.26093/cds/vizier.36160007"
VIZIER_TSV = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


def archive_target_id(identifier: str) -> str:
    value = identifier.strip()
    match = re.fullmatch(r"HIP0*(\d+)", value)
    if match:
        return f"HIP {int(match.group(1))}"
    match = re.fullmatch(r"TYC0*(\d+)-(\d+)-(\d+)", value)
    if match:
        return f"TYC {int(match.group(1))}-{int(match.group(2))}-{match.group(3)}"
    if value.startswith("2MASSJ"):
        return "2MASS J" + value[len("2MASSJ"):]
    return value


def parse_catalog_tsv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 4:
        raise ValueError("VizieR response contains no catalog rows")
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    rows = []
    for row in reader:
        # VizieR emits units and separator rows after the header.
        if not (row.get("ID") or "").strip() or (row.get("ID") or "").startswith("-"):
            continue
        try:
            float((row.get("s_RV") or "").strip())
        except ValueError:
            continue
        rows.append({key: (value or "").strip() for key, value in row.items()})
    if not rows:
        raise ValueError("VizieR response contains no parseable catalog rows")
    return rows


def is_selected_standard(row: dict[str, str], *, vmag_min: float = 7.0,
                         vmag_max: float = 11.5, max_rv_scatter: float = 0.1,
                         min_baseline_days: int = 300,
                         min_observations: int = 5) -> bool:
    try:
        vmag = float(row["Vmag"])
        scatter = float(row["s_RV"])
        baseline = int(row["Tbase"])
        observations = int(row["N"])
    except (KeyError, TypeError, ValueError):
        return False
    spectral_type = row.get("SpType", "").replace(" ", "")
    object_type = row.get("otype", "")
    is_fgk_dwarf = bool(re.match(r"^[FGK][0-9.]*V(?:$|[+/:])", spectral_type))
    excluded_type = any(token in object_type for token in (
        "Flare", "BYDra", "Ecl", "SB", "**", "Variable"))
    return bool(
        row.get("Flag") == "CAL1"
        and vmag_min <= vmag <= vmag_max
        and scatter <= max_rv_scatter
        and baseline >= min_baseline_days
        and observations >= min_observations
        and is_fgk_dwarf
        and not excluded_type
    )


def select_targets(rows: list[dict[str, str]], n_targets: int,
                   seed: int,
                   exclude_archive_ids: set[str] | None = None) \
        -> list[dict[str, str]]:
    excluded = set() if exclude_archive_ids is None else {
        value.strip() for value in exclude_archive_ids if value.strip()}
    selected = [
        row for row in rows
        if is_selected_standard(row)
        and archive_target_id(row["ID"]) not in excluded
    ]
    selected.sort(key=lambda row: (float(row.get("RA_ICRS", "0")), row["ID"]))
    if len(selected) < n_targets:
        raise ValueError(
            f"catalog filters produced {len(selected)} targets; requested {n_targets}")
    order = np.random.default_rng(seed).permutation(len(selected))[:n_targets]
    return [selected[int(index)] for index in order]


def fetch_catalog(timeout: int = 60) -> tuple[str, str]:
    query = urllib.parse.urlencode({
        "-source": CATALOG,
        "-out.all": "",
        "-out.max": "unlimited",
    })
    url = f"{VIZIER_TSV}?{query}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "TransitFlow-noise-target-selector/1.0"})
    context = ssl.create_default_context()
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(
            request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8"), url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noise_targets.txt")
    ap.add_argument("--metadata", default="data/noise_targets.json")
    ap.add_argument("--n-targets", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument(
        "--exclude-target-file", default=None,
        help="archive target IDs already used in development; excluded before sampling")
    args = ap.parse_args()
    if args.n_targets < 30:
        raise SystemExit("--n-targets must be at least 30 for a publication pool")

    text, url = fetch_catalog(args.timeout)
    rows = parse_catalog_tsv(text)
    excluded: set[str] = set()
    if args.exclude_target_file:
        exclude_path = Path(args.exclude_target_file)
        excluded = {
            line.strip() for line in exclude_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    targets = select_targets(rows, args.n_targets, args.seed, excluded)
    out = Path(args.out)
    metadata = Path(args.metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    archive_ids = [archive_target_id(row["ID"]) for row in targets]
    overlap = sorted(set(archive_ids) & excluded)
    if overlap:
        raise SystemExit(
            "selected targets overlap the exclusion set: " + ", ".join(overlap))
    out.write_text("\n".join(archive_ids) + "\n")
    metadata.write_text(json.dumps({
        "catalog": CATALOG,
        "catalog_doi": CATALOG_DOI,
        "query_url": url,
        "seed": int(args.seed),
        "n_catalog_rows": len(rows),
        "n_selected": len(targets),
        "exclude_target_file": args.exclude_target_file,
        "excluded_archive_target_ids": sorted(excluded),
        "n_excluded_archive_targets": len(excluded),
        "selected_exclusion_overlap": overlap,
        "filters": {
            "flag": "CAL1",
            "spectral_type": "FGK dwarf",
            "vmag": [7.0, 11.5],
            "max_rv_scatter_km_s": 0.1,
            "min_baseline_days": 300,
            "min_observations": 5,
            "excluded_object_types": ["Flare", "BYDra", "Ecl", "SB", "**", "Variable"],
        },
        "targets": targets,
        "archive_target_ids": archive_ids,
    }, indent=2, sort_keys=True))
    print(f"selected {len(targets)} targets -> {out}")
    print(f"wrote provenance -> {metadata}")


if __name__ == "__main__":
    main()
