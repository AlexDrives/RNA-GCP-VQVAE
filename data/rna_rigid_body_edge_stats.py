import argparse
import glob
import os
from typing import List

import h5py
import numpy as np


def find_h5_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".h5") else []
    pattern = os.path.join(input_path, "**", "*.h5")
    return glob.glob(pattern, recursive=True)


def compute_edge_lengths(coords: np.ndarray):
    """
    coords shape: (L, 3, 3), atom order [C4', C1', N1/N9]
    returns:
      l_c4_c1, l_c1_n, l_c4_n for valid residues only
    """
    if coords.ndim != 3 or coords.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected coords shape: {coords.shape}, expected (L, 3, 3).")

    valid_mask = np.isfinite(coords).all(axis=(1, 2))
    valid = coords[valid_mask]
    if valid.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            valid_mask,
        )

    c4 = valid[:, 0, :]
    c1 = valid[:, 1, :]
    n_atom = valid[:, 2, :]

    l_c4_c1 = np.linalg.norm(c4 - c1, axis=1)
    l_c1_n = np.linalg.norm(c1 - n_atom, axis=1)
    l_c4_n = np.linalg.norm(c4 - n_atom, axis=1)
    return l_c4_c1, l_c1_n, l_c4_n, valid_mask


def summarize(values: np.ndarray):
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(
        description="Compute RNA rigid-body edge length statistics from H5 files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="A single .h5 file path or a directory containing .h5 files.",
    )
    parser.add_argument(
        "--coord_key",
        type=str,
        default="C4p_C1p_N_coord",
        help="Coordinate dataset key in H5 (default: C4p_C1p_N_coord).",
    )
    args = parser.parse_args()

    h5_files = find_h5_files(args.input)
    if not h5_files:
        print(f"No .h5 files found from input: {args.input}")
        return

    all_c4_c1 = []
    all_c1_n = []
    all_c4_n = []
    total_residues = 0
    valid_residues = 0
    processed_files = 0
    skipped_files = 0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as f:
                if args.coord_key not in f:
                    skipped_files += 1
                    continue
                coords = f[args.coord_key][:]

            l_c4_c1, l_c1_n, l_c4_n, valid_mask = compute_edge_lengths(coords)
            total_residues += int(coords.shape[0])
            valid_residues += int(valid_mask.sum())
            processed_files += 1

            if l_c4_c1.size > 0:
                all_c4_c1.append(l_c4_c1)
                all_c1_n.append(l_c1_n)
                all_c4_n.append(l_c4_n)
        except Exception:
            skipped_files += 1

    if not all_c4_c1:
        print("No valid residues found for statistics.")
        print(f"Processed files: {processed_files}, skipped files: {skipped_files}")
        return

    all_c4_c1 = np.concatenate(all_c4_c1, axis=0)
    all_c1_n = np.concatenate(all_c1_n, axis=0)
    all_c4_n = np.concatenate(all_c4_n, axis=0)

    mean_c4_c1, std_c4_c1 = summarize(all_c4_c1)
    mean_c1_n, std_c1_n = summarize(all_c1_n)
    mean_c4_n, std_c4_n = summarize(all_c4_n)

    pooled = np.concatenate([all_c4_c1, all_c1_n, all_c4_n], axis=0)
    pooled_mean, pooled_std = summarize(pooled)

    print("RNA rigid-body edge length statistics")
    print("------------------------------------")
    print(f"Input: {args.input}")
    print(f"Coordinate key: {args.coord_key}")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Total residues: {total_residues}")
    print(f"Valid residues: {valid_residues}")
    print("")
    print(f"C4'-C1'  mean={mean_c4_c1:.6f}  std={std_c4_c1:.6f}")
    print(f"C1'-N    mean={mean_c1_n:.6f}   std={std_c1_n:.6f}")
    print(f"C4'-N    mean={mean_c4_n:.6f}   std={std_c4_n:.6f}")
    print("")
    print(f"All edges pooled  mean={pooled_mean:.6f}  std={pooled_std:.6f}")


if __name__ == "__main__":
    main()
