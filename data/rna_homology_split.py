#!/usr/bin/env python3
"""
RNA homology-aware dataset split pipeline.

Workflow:
1) Discover input samples (PDB/CIF/H5 files) and infer `pdb_id` + `chain_id`.
2) Query RCSB Data API for canonical polymer sequences.
3) Build a unified FASTA (one sequence per sample).
4) Cluster sequences with cd-hit-est at a target identity threshold (default: 0.60).
5) Split clusters into train/val/test by ratio (default: 70/15/15).
6) Map split assignments back to original files and materialize split directories.
7) Emit manifests and summary statistics.

This script is intentionally decoupled from model code and can be run on a data-prep server.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

try:
    from urllib3.util.retry import Retry
except ImportError as exc:  # pragma: no cover
    raise ImportError("urllib3 is required for robust API retries.") from exc


RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
CDHIT_EST_MIN_IDENTITY = 0.80

PDB_ID_RE = re.compile(r"(?i)([0-9][a-z0-9]{3})")
CHAIN_EXPLICIT_RE = re.compile(r"(?i)chain[_-]?id[_-]?([a-z0-9]+)")
CHAIN_SUFFIX_RE = re.compile(r"(?i)_([a-z0-9])$")
FASTA_HEADER_ID_RE = re.compile(r">([^\s>]+)")
# BGSU-style filename, e.g.:
#   PDB_00001A9N_1_Q.pdb
#   PDB_00001D4R_1_A-B.pdb
BGSU_STEM_RE = re.compile(r"(?i)^PDB_([A-Za-z0-9]+)_([0-9]+)_([A-Za-z0-9\-]+)$")
CHAIN_GROUP_SPLIT_RE = re.compile(r"[-,;+]")


@dataclass
class SampleRecord:
    seq_id: str
    sample_id: str
    source_path: str
    pdb_id: str
    chain_id: str
    sequence: str = ""
    status: str = "pending"
    reason: str = ""
    cluster_id: str = ""
    split: str = ""


def build_requests_session(timeout_retries: int = 4, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=timeout_retries,
        connect=timeout_retries,
        read=timeout_retries,
        status=timeout_retries,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def sanitize_sequence(seq: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", seq or "").upper()
    return cleaned


def is_rna_polymer(polymer_type: str) -> bool:
    t = (polymer_type or "").lower()
    return "ribonucleotide" in t


def infer_ids_from_filename(path: Path) -> Tuple[Optional[str], str]:
    stem = path.stem

    # Prefer explicit parsing of BGSU-style stems:
    #   PDB_<numeric-prefix+4char-pdbid>_<model>_<chain-or-chain-group>
    bgsu_match = BGSU_STEM_RE.match(stem)
    if bgsu_match:
        entry_token = re.sub(r"[^A-Za-z0-9]", "", bgsu_match.group(1))
        pdb_id = None
        if len(entry_token) >= 4:
            candidate = entry_token[-4:]
            if re.fullmatch(r"[A-Za-z0-9]{4}", candidate):
                pdb_id = candidate.upper()
        chain_id = bgsu_match.group(3)
        return pdb_id, chain_id

    pdb_match = PDB_ID_RE.search(stem)
    pdb_id = pdb_match.group(1).upper() if pdb_match else None

    chain_id = ""
    m = CHAIN_EXPLICIT_RE.search(stem)
    if m:
        chain_id = m.group(1)
    else:
        m2 = CHAIN_SUFFIX_RE.search(stem)
        if m2:
            chain_id = m2.group(1)
    return pdb_id, chain_id


def _candidate_chain_ids(raw_chain_id: str) -> List[str]:
    chain_id = (raw_chain_id or "").strip()
    if not chain_id:
        return []

    # Keep full token first (for backward compatibility), then split grouped tokens (e.g., A-B).
    tokens = [chain_id]
    tokens.extend(
        [part.strip() for part in CHAIN_GROUP_SPLIT_RE.split(chain_id) if part.strip()]
    )

    deduped: List[str] = []
    seen = set()
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(token)
    return deduped


def collect_samples(input_root: Path, extensions: Iterable[str]) -> List[SampleRecord]:
    exts = {e.lower().strip() for e in extensions}
    files: List[Path] = []
    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts:
            files.append(path)
    files.sort()

    samples: List[SampleRecord] = []
    skipped_no_pdb = 0
    for i, fpath in enumerate(files):
        pdb_id, chain_id = infer_ids_from_filename(fpath)
        if not pdb_id:
            skipped_no_pdb += 1
            continue
        seq_id = f"S{i:08d}"
        samples.append(
            SampleRecord(
                seq_id=seq_id,
                sample_id=fpath.stem,
                source_path=str(fpath.resolve()),
                pdb_id=pdb_id,
                chain_id=chain_id,
            )
        )

    if skipped_no_pdb > 0:
        print(f"[warn] skipped {skipped_no_pdb} files because PDB ID could not be inferred from filename.")
    return samples


def load_samples_from_metadata(csv_path: Path, input_root: Optional[Path] = None) -> List[SampleRecord]:
    required = {"sample_id", "pdb_id", "path"}
    samples: List[SampleRecord] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = required.difference(set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(f"metadata CSV missing required columns: {sorted(missing_cols)}")
        for i, row in enumerate(reader):
            source = (row.get("path") or "").strip()
            if not source:
                continue
            source_path = Path(source)
            if not source_path.is_absolute():
                if input_root is None:
                    raise ValueError("relative path in metadata requires --input_root.")
                source_path = (input_root / source_path).resolve()
            seq_id = f"S{i:08d}"
            samples.append(
                SampleRecord(
                    seq_id=seq_id,
                    sample_id=(row.get("sample_id") or source_path.stem).strip(),
                    source_path=str(source_path),
                    pdb_id=(row.get("pdb_id") or "").strip().upper(),
                    chain_id=(row.get("chain_id") or "").strip(),
                )
            )
    return samples


def fetch_json(session: requests.Session, url: str, timeout_s: float) -> Optional[dict]:
    resp = session.get(url, timeout=timeout_s)
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except json.JSONDecodeError:
        return None


def fetch_rna_chain_sequences(
    session: requests.Session,
    pdb_id: str,
    timeout_s: float,
) -> Dict[str, str]:
    entry_url = RCSB_ENTRY_URL.format(pdb_id=pdb_id.lower())
    entry = fetch_json(session, entry_url, timeout_s=timeout_s)
    if not entry:
        return {}

    ids_container = entry.get("rcsb_entry_container_identifiers", {})
    entity_ids = ids_container.get("polymer_entity_ids", [])
    if not entity_ids:
        return {}

    chain_to_seq: Dict[str, str] = {}
    for entity_id in entity_ids:
        eurl = RCSB_ENTITY_URL.format(pdb_id=pdb_id.lower(), entity_id=entity_id)
        entity = fetch_json(session, eurl, timeout_s=timeout_s)
        if not entity:
            continue

        entity_poly = entity.get("entity_poly", {})
        poly_type = str(entity_poly.get("type", ""))
        if not is_rna_polymer(poly_type):
            continue

        seq = sanitize_sequence(
            str(
                entity_poly.get("pdbx_seq_one_letter_code_can")
                or entity_poly.get("pdbx_seq_one_letter_code")
                or ""
            )
        )
        if not seq:
            continue

        id_block = entity.get("rcsb_polymer_entity_container_identifiers", {})
        chain_ids = id_block.get("auth_asym_ids") or id_block.get("asym_ids") or []
        for cid in chain_ids:
            chain_to_seq[str(cid)] = seq
    return chain_to_seq


def resolve_sequence_for_sample(sample: SampleRecord, chain_to_seq: Dict[str, str]) -> Tuple[str, str]:
    if not chain_to_seq:
        return "", "rcsb_no_rna_sequence"

    if sample.chain_id and sample.chain_id in chain_to_seq:
        return chain_to_seq[sample.chain_id], ""

    if sample.chain_id:
        matched_sequences: List[str] = []
        for cid in _candidate_chain_ids(sample.chain_id):
            if cid in chain_to_seq:
                matched_sequences.append(chain_to_seq[cid])
                continue
            for k, v in chain_to_seq.items():
                if k.lower() == cid.lower():
                    matched_sequences.append(v)
                    break

        if matched_sequences:
            unique_matched = list(dict.fromkeys(matched_sequences))
            if len(unique_matched) == 1:
                return unique_matched[0], ""
            return "", "chain_group_multi_rna_sequences"

    unique_seqs = list(dict.fromkeys(chain_to_seq.values()))
    if len(unique_seqs) == 1:
        return unique_seqs[0], ""

    return "", "chain_not_found_multi_rna_entities"


def write_fasta(records: List[SampleRecord], fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as f:
        for r in records:
            header = f">{r.seq_id}|sample={r.sample_id}|pdb={r.pdb_id}|chain={r.chain_id or 'NA'}|len={len(r.sequence)}"
            f.write(header + "\n")
            f.write(r.sequence + "\n")


def count_fasta_records(fasta_path: Path) -> int:
    n = 0
    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def infer_cdhit_word_size(identity: float) -> int:
    if identity >= 0.90:
        return 8
    if identity >= 0.88:
        return 7
    if identity >= 0.85:
        return 6
    if identity >= 0.80:
        return 5
    if identity >= 0.75:
        return 4
    return 3


def run_cdhit_est(
    fasta_in: Path,
    output_prefix: Path,
    identity: float,
    threads: int,
    memory_mb: int,
    word_size: Optional[int],
) -> Tuple[Path, Path]:
    n = word_size if word_size is not None else infer_cdhit_word_size(identity)
    cmd = [
        "cd-hit-est",
        "-i",
        str(fasta_in),
        "-o",
        str(output_prefix),
        "-c",
        f"{identity:.3f}",
        "-n",
        str(n),
        "-d",
        "0",
        "-T",
        str(threads),
        "-M",
        str(memory_mb),
    ]
    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    clstr_path = Path(str(output_prefix) + ".clstr")
    clustered_fasta = output_prefix
    if not clstr_path.exists():
        raise FileNotFoundError(f"cd-hit cluster file not found: {clstr_path}")
    return clustered_fasta, clstr_path


def run_cdhit_est_2d(
    query_fasta: Path,
    db_fasta: Path,
    output_prefix: Path,
    identity: float,
    threads: int,
    memory_mb: int,
    word_size: Optional[int],
) -> Path:
    n = word_size if word_size is not None else infer_cdhit_word_size(identity)
    cmd = [
        "cd-hit-est-2d",
        "-i",
        str(query_fasta),
        "-i2",
        str(db_fasta),
        "-o",
        str(output_prefix),
        "-c",
        f"{identity:.3f}",
        "-n",
        str(n),
        "-d",
        "0",
        "-T",
        str(threads),
        "-M",
        str(memory_mb),
    ]
    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_prefix


def parse_cdhit_clstr(clstr_path: Path) -> Dict[str, str]:
    seq_to_cluster: Dict[str, str] = {}
    current_cluster = ""
    with clstr_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                cid = line.split(maxsplit=1)[1]
                current_cluster = f"cluster_{int(cid):08d}"
                continue
            m = FASTA_HEADER_ID_RE.search(line)
            if not m:
                continue
            # CD-HIT clstr lines contain tokens like `>S00000001|sample=...`.
            # Keep only the original FASTA id before metadata after `|`.
            token = m.group(1).replace("...", "").rstrip(".,")
            seq_id = token.split("|", 1)[0]
            seq_to_cluster[seq_id] = current_cluster
    return seq_to_cluster


def split_clusters(
    cluster_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if not cluster_ids:
        raise ValueError("no clusters found.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1.0.")

    ids = list(cluster_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_train <= 0 and n > 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_test = max(0, n_test - 1)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]

    mapping: Dict[str, str] = {}
    for cid in train_ids:
        mapping[cid] = "train"
    for cid in val_ids:
        mapping[cid] = "val"
    for cid in test_ids:
        mapping[cid] = "test"
    return mapping


def write_manifest(rows: List[SampleRecord], output_csv: Path) -> None:
    fields = [
        "seq_id",
        "sample_id",
        "source_path",
        "pdb_id",
        "chain_id",
        "sequence",
        "status",
        "reason",
        "cluster_id",
        "split",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def materialize_split_dirs(
    rows: List[SampleRecord],
    input_root: Path,
    output_dir: Path,
    mode: str,
) -> None:
    if mode == "none":
        return

    split_root = output_dir / "splits"
    split_root.mkdir(parents=True, exist_ok=True)

    for r in tqdm(rows, desc=f"materialize[{mode}]"):
        if not r.split:
            continue
        src = Path(r.source_path)
        if not src.exists():
            continue
        try:
            rel = src.relative_to(input_root)
        except ValueError:
            rel = Path(src.name)
        dst = split_root / r.split / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"unsupported materialize mode: {mode}")


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA homology-aware split with RCSB API + cd-hit-est.")
    parser.add_argument("--input_root", type=str, help="Root directory of original RNA sample files.")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".pdb,.cif,.h5",
        help="Comma-separated file suffixes to scan under input_root.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="",
        help="Optional CSV with columns: sample_id,pdb_id,path[,chain_id]. Overrides file scanning.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--identity",
        type=float,
        default=0.80,
        help="cd-hit-est identity cutoff (cd-hit-est v4.8.x requires >= 0.80).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster split.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--threads", type=int, default=16, help="cd-hit-est threads (-T).")
    parser.add_argument("--memory_mb", type=int, default=0, help="cd-hit-est memory in MB (-M). 0 means unlimited.")
    parser.add_argument("--word_size", type=int, default=0, help="cd-hit-est word size (-n). 0 means auto.")
    parser.add_argument(
        "--materialize",
        type=str,
        default="symlink",
        choices=["none", "symlink", "copy"],
        help="How to materialize split datasets from original files.",
    )
    parser.add_argument("--api_timeout_s", type=float, default=20.0, help="HTTP timeout per API call.")
    parser.add_argument(
        "--skip_train_test_check",
        action="store_true",
        help="Skip leakage validation with cd-hit-est-2d (not recommended).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.identity < CDHIT_EST_MIN_IDENTITY:
        raise ValueError(
            f"--identity {args.identity:.3f} is unsupported by cd-hit-est on this setup. "
            f"Please use --identity >= {CDHIT_EST_MIN_IDENTITY:.2f} "
            f"(or switch to a different clustering backend for lower identity)."
        )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"output_dir is not empty: {out_dir}. Use --overwrite to continue."
        )

    if args.metadata_csv:
        if not args.input_root:
            raise ValueError("--input_root is required when --metadata_csv uses relative paths.")
        input_root = Path(args.input_root).resolve()
        records = load_samples_from_metadata(Path(args.metadata_csv), input_root=input_root)
    else:
        if not args.input_root:
            raise ValueError("--input_root is required when metadata_csv is not provided.")
        input_root = Path(args.input_root).resolve()
        exts = [x.strip() for x in args.extensions.split(",") if x.strip()]
        records = collect_samples(input_root, exts)

    if not records:
        raise RuntimeError("no input samples discovered.")

    print(f"[info] discovered samples: {len(records)}")
    unique_pdb_ids = sorted({r.pdb_id for r in records})
    print(f"[info] unique pdb ids: {len(unique_pdb_ids)}")

    session = build_requests_session()
    cache_path = out_dir / "rcsb_cache.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            pdb_cache = json.load(f)
    else:
        pdb_cache = {}

    for pdb_id in tqdm(unique_pdb_ids, desc="RCSB API"):
        if pdb_id in pdb_cache:
            continue
        chain_to_seq = fetch_rna_chain_sequences(session, pdb_id, timeout_s=args.api_timeout_s)
        pdb_cache[pdb_id] = chain_to_seq

    save_json(pdb_cache, cache_path)

    unresolved = Counter()
    resolved_records: List[SampleRecord] = []
    for r in records:
        chain_to_seq = pdb_cache.get(r.pdb_id, {})
        seq, reason = resolve_sequence_for_sample(r, chain_to_seq)
        if seq:
            r.sequence = seq
            r.status = "ok"
            resolved_records.append(r)
        else:
            r.status = "unresolved"
            r.reason = reason
            unresolved[reason] += 1

    print(f"[info] resolved sequences: {len(resolved_records)}")
    print(f"[info] unresolved samples: {len(records) - len(resolved_records)}")
    if unresolved:
        for k, v in unresolved.items():
            print(f"[warn] unresolved[{k}] = {v}")

    if not resolved_records:
        raise RuntimeError("no resolved sequences; cannot continue clustering.")

    unified_fasta = out_dir / "all_samples.fasta"
    write_fasta(resolved_records, unified_fasta)
    print(f"[info] wrote unified FASTA: {unified_fasta}")

    cdhit_prefix = out_dir / f"cdhit_c{args.identity:.2f}".replace(".", "p")
    _, clstr_path = run_cdhit_est(
        fasta_in=unified_fasta,
        output_prefix=cdhit_prefix,
        identity=args.identity,
        threads=args.threads,
        memory_mb=args.memory_mb,
        word_size=(None if args.word_size <= 0 else args.word_size),
    )
    print(f"[info] cd-hit cluster file: {clstr_path}")

    seq_to_cluster = parse_cdhit_clstr(clstr_path)
    if not seq_to_cluster:
        raise RuntimeError("parsed empty cluster mapping from .clstr file.")

    missing_cluster = 0
    cluster_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
    for r in resolved_records:
        cid = seq_to_cluster.get(r.seq_id, "")
        if not cid:
            r.status = "unresolved"
            r.reason = "missing_cluster_assignment"
            missing_cluster += 1
            continue
        r.cluster_id = cid
        cluster_to_records[cid].append(r)
    if missing_cluster > 0:
        print(f"[warn] missing cluster assignment for {missing_cluster} samples")

    cluster_ids = sorted(cluster_to_records.keys())
    cluster_split = split_clusters(
        cluster_ids=cluster_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for cid, rows in cluster_to_records.items():
        split_name = cluster_split[cid]
        for r in rows:
            r.split = split_name

    write_manifest(records, out_dir / "all_samples_manifest.csv")
    write_manifest([r for r in records if r.split == "train"], out_dir / "train_manifest.csv")
    write_manifest([r for r in records if r.split == "val"], out_dir / "val_manifest.csv")
    write_manifest([r for r in records if r.split == "test"], out_dir / "test_manifest.csv")

    materialize_split_dirs(records, input_root=input_root, output_dir=out_dir, mode=args.materialize)

    split_sample_counts = Counter(r.split for r in records if r.split)
    split_cluster_counts = Counter(cluster_split.values())

    train_rows = [r for r in records if r.split == "train"]
    test_rows = [r for r in records if r.split == "test"]

    train_fasta = out_dir / "train_sequences.fasta"
    test_fasta = out_dir / "test_sequences.fasta"
    write_fasta(train_rows, train_fasta)
    write_fasta(test_rows, test_fasta)

    train_test_check = {
        "performed": False,
        "passed": False,
        "identity_cutoff": args.identity,
        "test_total": len(test_rows),
        "test_nonredundant_vs_train": 0,
        "leakage_count": 0,
    }

    if not args.skip_train_test_check and len(train_rows) > 0 and len(test_rows) > 0:
        check_prefix = out_dir / f"cdhit2d_test_vs_train_c{args.identity:.2f}".replace(".", "p")
        check_out = run_cdhit_est_2d(
            query_fasta=test_fasta,
            db_fasta=train_fasta,
            output_prefix=check_prefix,
            identity=args.identity,
            threads=args.threads,
            memory_mb=args.memory_mb,
            word_size=(None if args.word_size <= 0 else args.word_size),
        )
        nonredundant_test = count_fasta_records(check_out)
        leakage_count = max(0, len(test_rows) - nonredundant_test)
        train_test_check.update(
            {
                "performed": True,
                "passed": leakage_count == 0,
                "test_nonredundant_vs_train": nonredundant_test,
                "leakage_count": leakage_count,
            }
        )
        if leakage_count > 0:
            raise RuntimeError(
                f"train-test leakage detected at identity >= {args.identity:.2f}: "
                f"{leakage_count} test sequences are homologous to train."
            )

    summary = {
        "total_samples": len(records),
        "resolved_sequences": len(resolved_records),
        "unresolved_samples": len(records) - len(resolved_records),
        "total_clusters": len(cluster_ids),
        "identity_cutoff": args.identity,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_sample_counts": dict(split_sample_counts),
        "split_cluster_counts": dict(split_cluster_counts),
        "train_test_check": train_test_check,
        "unresolved_breakdown": dict(unresolved),
        "artifacts": {
            "unified_fasta": str(unified_fasta),
            "cluster_file": str(clstr_path),
            "train_fasta": str(train_fasta),
            "test_fasta": str(test_fasta),
            "all_manifest": str(out_dir / "all_samples_manifest.csv"),
            "train_manifest": str(out_dir / "train_manifest.csv"),
            "val_manifest": str(out_dir / "val_manifest.csv"),
            "test_manifest": str(out_dir / "test_manifest.csv"),
        },
    }
    save_json(summary, out_dir / "summary_stats.json")

    print("\n=== Summary ===")
    print(f"Total sequences: {summary['resolved_sequences']} (resolved) / {summary['total_samples']} (all)")
    print(f"Clusters: {summary['total_clusters']}")
    print(
        "Samples per split: "
        f"train={split_sample_counts.get('train', 0)}, "
        f"val={split_sample_counts.get('val', 0)}, "
        f"test={split_sample_counts.get('test', 0)}"
    )
    print(f"Summary JSON: {out_dir / 'summary_stats.json'}")


if __name__ == "__main__":
    main()
