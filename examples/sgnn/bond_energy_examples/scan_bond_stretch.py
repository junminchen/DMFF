#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import pickle
import re
import sys

import jax.numpy as jnp
import numpy as np

ATYPE_INDEX = {'H': 0, 'C': 1, 'O': 2}


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_pair(i: int, j: int):
    return (i, j) if i <= j else (j, i)


def _find_default_ch_bond(elems, bonds):
    for i, j in bonds:
        if {elems[i], elems[j]} == {"C", "H"}:
            return int(i), int(j)
    return None


def _bond_contrib_sum(center_bonds, subgraph_scaled, i_target, j_target):
    ti, tj = _normalize_pair(i_target, j_target)
    s = 0.0
    for idx, bond in enumerate(center_bonds.tolist()):
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0:
            continue
        pi, pj = _normalize_pair(i, j)
        if (pi, pj) == (ti, tj):
            s += float(subgraph_scaled[idx])
    return s


def _stretch_positions(positions, i, j, delta_r):
    rij = positions[j] - positions[i]
    r0 = jnp.linalg.norm(rij)
    if float(r0) < 1e-10:
        raise RuntimeError(f"Bond length near zero for atoms ({i}, {j}).")
    u = rij / r0
    return positions.at[j].set(positions[j] + u * delta_r)


def _format_pair(i, j, elems):
    return f"({i}, {j}) [{elems[i]}-{elems[j]}]"


def _load_params_dict(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _infer_arch_from_params(params):
    pat = re.compile(r"^fc([01])\.(\d+)\.weight$")
    layers = {0: [], 1: []}
    for k, v in params.items():
        m = pat.match(k)
        if not m:
            continue
        g = int(m.group(1))
        i = int(m.group(2))
        arr = np.array(v)
        layers[g].append((i, int(arr.shape[0])))
    if layers[0] and layers[1]:
        l0 = sorted(layers[0], key=lambda x: x[0])
        l1 = sorted(layers[1], key=lambda x: x[0])
        return (len(l0), len(l1)), (tuple(x[1] for x in l0), tuple(x[1] for x in l1))

    if "fc0.weight" in params and "fc1.weight" in params:
        l0 = [np.array(w).shape[0] for w in params["fc0.weight"]]
        l1 = [np.array(w).shape[0] for w in params["fc1.weight"]]
        return (len(l0), len(l1)), (tuple(l0), tuple(l1))

    return (3, 2), ((40, 20, 20), (20, 10))


def main():
    repo_root = _repo_root_from_here()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dmff.sgnn.gnn import MolGNNForce
    from dmff.sgnn.graph import from_pdb

    parser = argparse.ArgumentParser(description="DMFF JAX sGNN C-H bond stretch scan (PEG4/model1).")
    parser.add_argument("--pdb", type=Path, default=repo_root / "examples" / "sgnn" / "peg4.pdb")
    parser.add_argument("--params", type=Path, default=repo_root / "examples" / "sgnn" / "model1.pickle")
    parser.add_argument("--nn", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-valence", type=int, default=4)
    parser.add_argument("--bond", type=int, nargs=2, metavar=("I", "J"), default=None)
    parser.add_argument("--dr-min", type=float, default=0.0)
    parser.add_argument("--dr-max", type=float, default=1.5)
    parser.add_argument("--n-points", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    if not args.pdb.exists():
        raise FileNotFoundError(f"PDB not found: {args.pdb}")
    if not args.params.exists():
        raise FileNotFoundError(f"params not found: {args.params}")
    if args.n_points < 2:
        raise ValueError("--n-points must be >= 2")

    params_raw = _load_params_dict(args.params)
    n_layers, sizes = _infer_arch_from_params(params_raw)

    graph = from_pdb(str(args.pdb))
    box = graph.box if graph.box is not None else jnp.eye(3) * 50.0

    model = MolGNNForce(
        graph,
        nn=args.nn,
        n_layers=n_layers,
        sizes=[tuple(sizes[0]), tuple(sizes[1])],
        max_valence=args.max_valence,
        atype_index=ATYPE_INDEX,
    )
    model.load_params(str(args.params))

    elems = list(graph.list_atom_elems)
    bonds = graph.bonds.tolist()
    if args.bond is None:
        picked = _find_default_ch_bond(elems, bonds)
        if picked is None:
            raise RuntimeError("No C-H bond found. Please provide --bond I J.")
        bi, bj = picked
    else:
        bi, bj = int(args.bond[0]), int(args.bond[1])

    if bi < 0 or bj < 0 or bi >= len(elems) or bj >= len(elems):
        raise IndexError(f"bond indices out of range: ({bi}, {bj}), n_atoms={len(elems)}")

    out0 = model.forward_with_components(graph.positions, box, model.params)
    e_total0 = float(out0["total_energy"])
    e_bond0 = _bond_contrib_sum(out0["center_bonds_parent"], out0["subgraph_contrib_scaled"], bi, bj)
    r0 = float(jnp.linalg.norm(graph.positions[bj] - graph.positions[bi]))

    rows = []
    for dr in jnp.linspace(args.dr_min, args.dr_max, args.n_points):
        drf = float(dr)
        pos_stretched = _stretch_positions(graph.positions, bi, bj, drf)
        out = model.forward_with_components(pos_stretched, box, model.params)
        e_total = float(out["total_energy"])
        e_bond = _bond_contrib_sum(out["center_bonds_parent"], out["subgraph_contrib_scaled"], bi, bj)
        rows.append(
            {
                "dr_angstrom": drf,
                "bond_length_angstrom": r0 + drf,
                "total_energy_kcal_mol": e_total,
                "delta_total_energy_kcal_mol": e_total - e_total0,
                "bond_contrib_kcal_mol": e_bond,
                "delta_bond_contrib_kcal_mol": e_bond - e_bond0,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "peg4_ch_bond_scan.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dr_angstrom",
                "bond_length_angstrom",
                "total_energy_kcal_mol",
                "delta_total_energy_kcal_mol",
                "bond_contrib_kcal_mol",
                "delta_bond_contrib_kcal_mol",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    png_path = args.out_dir / "peg4_ch_bond_scan_deltaE.png"
    plotted = False
    try:
        import matplotlib.pyplot as plt

        x = [r["bond_length_angstrom"] for r in rows]
        y_tot = [r["delta_total_energy_kcal_mol"] for r in rows]
        y_bond = [r["delta_bond_contrib_kcal_mol"] for r in rows]
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(x, y_tot, marker="o", label="Delta E_total")
        plt.plot(x, y_bond, marker="s", label="Delta E_bond_contrib")
        plt.xlabel("Bond length (Angstrom)")
        plt.ylabel("Delta E (kcal/mol)")
        plt.title(f"PEG4 C-H bond stretch {_format_pair(bi, bj, elems)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()
        plotted = True
    except Exception:
        plotted = False

    print("=== DMFF JAX sGNN C-H bond stretch scan finished ===")
    print(f"pdb: {args.pdb}")
    print(f"params: {args.params}")
    print(f"nn: {args.nn}, max_valence: {args.max_valence}")
    print(f"atype_index: {ATYPE_INDEX}")
    print(f"inferred n_layers: {n_layers}, sizes: {sizes}")
    print(f"bond: {_format_pair(bi, bj, elems)}")
    print(f"r0 (Angstrom): {r0:.8f}")
    print(f"dr range (Angstrom): [{args.dr_min}, {args.dr_max}], n_points={args.n_points}")
    print(f"csv: {csv_path}")
    if plotted:
        print(f"plot: {png_path}")
    else:
        print("plot: skipped (matplotlib unavailable or plotting failed)")


if __name__ == "__main__":
    main()
