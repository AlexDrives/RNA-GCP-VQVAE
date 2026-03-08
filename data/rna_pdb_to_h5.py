import argparse
import glob
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm

# Map residue names to RNA bases.
# Only A/U/C/G are treated as valid RNA bases; anything else becomes "X".
RESNAME_TO_BASE = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "URA": "U",
    "CYT": "C",
    "GUA": "G",
    "ADE": "A",
}

PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}


def find_structure_files(directory_path: str, use_cif: bool) -> list[str]:
    patterns = [os.path.join(directory_path, "**", "*.pdb")]
    if use_cif:
        patterns = [os.path.join(directory_path, "**", "*.cif")]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def write_h5_file(file_path: str, seq: str, coords, plddt_scores, backbone_coords=None):
    with h5py.File(file_path, "w") as f:
        f.create_dataset("seq", data=seq)
        f.create_dataset("C4p_C1p_N_coord", data=coords)
        f.create_dataset("plddt_scores", data=plddt_scores)
        if backbone_coords is not None:
            f.create_dataset("P_O5p_C5p_C4p_C3p_O3p_coord", data=backbone_coords)


def _atom_coord(residue, names: list[str]):
    for name in names:
        if name in residue:
            return list(residue[name].coord), residue[name].get_bfactor()
    return None, math.nan


def _coord_or_nan(residue, names: list[str]):
    coord, _ = _atom_coord(residue, names)
    if coord is None:
        return [math.nan, math.nan, math.nan]
    return coord


def evaluate_missing_content(pos, max_missing_ratio=0.3, max_consecutive_missing=15):
    """Return (is_valid, reason_key) based on missing residue statistics."""
    total = len(pos)
    if total == 0:
        return False, "missing_ratio_exceeded"

    missing_flags = []
    for residue in pos:
        c1p_coords = residue[1] if len(residue) > 1 else []
        if len(c1p_coords) != 3:
            missing_flags.append(True)
            continue
        missing_flags.append(any(math.isnan(v) for v in c1p_coords))

    missing_count = sum(missing_flags)
    if missing_count / total > max_missing_ratio:
        return False, "missing_ratio_exceeded"

    longest_run = 0
    current_run = 0
    for is_missing in missing_flags:
        if is_missing:
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0
    if longest_run > max_consecutive_missing:
        return False, "missing_block_exceeded"

    return True, ""


def preprocess_file(
    file_index: int,
    file_path: str,
    max_len: int,
    min_len: int,
    max_missing_ratio: float,
    save_path: str,
    use_cif: bool,
    no_file_index: bool,
    gap_threshold: int,
):
    stats = Counter()
    parser = MMCIFParser(QUIET=True, auth_chains=False) if use_cif else PDBParser(QUIET=True)
    structure = parser.get_structure("rna", file_path)

    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == " "]
            if not residues:
                continue

            rna_seq = ""
            pos = []
            backbone_pos = []
            plddt_scores = []

            for residue in residues:
                resname = residue.resname.strip().upper()
                base = RESNAME_TO_BASE.get(resname, None)

                # Default to unknown if base is unsupported.
                if base not in PURINES and base not in PYRIMIDINES:
                    rna_seq += "X"
                    pos.append([[math.nan, math.nan, math.nan] for _ in range(3)])
                    backbone_pos.append([[math.nan, math.nan, math.nan] for _ in range(6)])
                    plddt_scores.append(math.nan)
                    continue

                # Atom names: support both prime (') and star (*) conventions.
                c4p_coord, _ = _atom_coord(residue, ["C4'", "C4*"])
                c1p_coord, c1p_b = _atom_coord(residue, ["C1'", "C1*"])
                if base in PURINES:
                    n_coord, _ = _atom_coord(residue, ["N9"])
                else:
                    n_coord, _ = _atom_coord(residue, ["N1"])

                p_coord = _coord_or_nan(residue, ["P"])
                o5p_coord = _coord_or_nan(residue, ["O5'", "O5*"])
                c5p_coord = _coord_or_nan(residue, ["C5'", "C5*"])
                c4p_full_coord = _coord_or_nan(residue, ["C4'", "C4*"])
                c3p_coord = _coord_or_nan(residue, ["C3'", "C3*"])
                o3p_coord = _coord_or_nan(residue, ["O3'", "O3*"])

                if c4p_coord is None or c1p_coord is None or n_coord is None:
                    coords = [[math.nan, math.nan, math.nan] for _ in range(3)]
                    plddt_scores.append(math.nan)
                else:
                    coords = [c4p_coord, c1p_coord, n_coord]
                    plddt_scores.append(c1p_b)

                rna_seq += base
                pos.append(coords)
                backbone_pos.append([p_coord, o5p_coord, c5p_coord, c4p_full_coord, c3p_coord, o3p_coord])

            # --- Numeric gap handling (no geometric estimation) ---
            for i in range(len(residues) - 1, 0, -1):
                current_res_id = residues[i].id
                prev_res_id = residues[i - 1].id
                if current_res_id[1] > prev_res_id[1] + 1:
                    numeric_gap_size = current_res_id[1] - prev_res_id[1] - 1
                    insert_count = min(numeric_gap_size, gap_threshold)
                    if insert_count <= 0:
                        continue
                    x_padding = "X" * insert_count
                    nan_coord_padding = [[math.nan, math.nan, math.nan] for _ in range(3)]
                    nan_pos_padding = [nan_coord_padding] * insert_count
                    nan_plddt_padding = [math.nan] * insert_count
                    nan_backbone_padding = [[math.nan, math.nan, math.nan] for _ in range(6)]
                    nan_backbone_pos_padding = [nan_backbone_padding] * insert_count
                    rna_seq = rna_seq[:i] + x_padding + rna_seq[i:]
                    pos[i:i] = nan_pos_padding
                    backbone_pos[i:i] = nan_backbone_pos_padding
                    plddt_scores[i:i] = nan_plddt_padding
                    stats["missing_residues"] += insert_count

            final_len = len(rna_seq)
            if final_len < min_len:
                stats["chains_too_short"] += 1
                continue
            if final_len > max_len:
                stats["chains_too_long"] += 1
                continue

            is_valid, reason = evaluate_missing_content(pos, max_missing_ratio=max_missing_ratio)
            if not is_valid:
                stats[reason] += 1
                continue

            basename = os.path.splitext(os.path.basename(file_path))[0]
            if no_file_index:
                outputfile = os.path.join(save_path, f"{basename}_chain_id_{chain.id}.h5")
            else:
                outputfile = os.path.join(save_path, f"{file_index}_{basename}_chain_id_{chain.id}.h5")

            write_h5_file(outputfile, rna_seq, pos, plddt_scores, backbone_pos)
            stats["written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert RNA PDB/CIF files to HDF5.")
    parser.add_argument("--data", type=str, required=True, help="Root directory of PDB/CIF files.")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory for .h5 files.")
    parser.add_argument("--use_cif", action="store_true", help="Parse CIF files instead of PDB.")
    parser.add_argument("--max_len", type=int, default=511, help="Maximum allowed sequence length.")
    parser.add_argument("--min_len", type=int, default=11, help="Minimum allowed sequence length.")
    parser.add_argument(
        "--max_missing_ratio",
        type=float,
        default=0.30,
        help="Maximum allowed residue-level missing ratio.",
    )
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="Number of workers.")
    parser.add_argument("--gap_threshold", type=int, default=5, help="Max inserted residues for numeric gaps.")
    parser.add_argument("--no_file_index", action="store_true", help="Do not prefix filenames with index.")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    structure_files = find_structure_files(args.data, args.use_cif)

    if not structure_files:
        print(f"No structure files found under {args.data}")
        return

    stats = Counter()
    if args.max_workers and args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    preprocess_file,
                    i,
                    path,
                    args.max_len,
                    args.min_len,
                    args.max_missing_ratio,
                    args.save_path,
                    args.use_cif,
                    args.no_file_index,
                    args.gap_threshold,
                ): path
                for i, path in enumerate(structure_files)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing RNA files"):
                stats.update(future.result())
    else:
        for i, path in enumerate(tqdm(structure_files, desc="Processing RNA files")):
            stats.update(
                preprocess_file(
                    i,
                    path,
                    args.max_len,
                    args.min_len,
                    args.max_missing_ratio,
                    args.save_path,
                    args.use_cif,
                    args.no_file_index,
                    args.gap_threshold,
                )
            )

    print("Finished. Summary stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
