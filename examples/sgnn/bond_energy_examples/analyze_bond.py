#!/usr/bin/env python3
import argparse
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


def _format_pair(i: int, j: int, elems):
    return f"({i}, {j}) [{elems[i]}-{elems[j]}]"


def _find_default_ch_bond(elems, bonds):
    for i, j in bonds:
        if {elems[i], elems[j]} == {"C", "H"}:
            return int(i), int(j)
    return None


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

    parser = argparse.ArgumentParser(description="Analyze bond-centered local contributions using DMFF JAX sGNN (PEG4 C-H).")
    parser.add_argument("--pdb", type=Path, default=repo_root / "examples" / "sgnn" / "peg4.pdb")
    parser.add_argument("--params", type=Path, default=repo_root / "examples" / "sgnn" / "model1.pickle")
    parser.add_argument("--nn", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-valence", type=int, default=4)
    parser.add_argument("--bond", type=int, nargs=2, metavar=("I", "J"), default=None)
    args = parser.parse_args()

    if not args.pdb.exists():
        raise FileNotFoundError(f"PDB not found: {args.pdb}")
    if not args.params.exists():
        raise FileNotFoundError(f"params not found: {args.params}")

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

    out = model.forward_with_components(graph.positions, box, model.params)
    total_energy = float(out["total_energy"])
    center_bonds = out["center_bonds_parent"]
    subgraph_scaled = out["subgraph_contrib_scaled"]
    elems = list(graph.list_atom_elems)

    center_rows = []
    for sg_idx, bond in enumerate(center_bonds.tolist()):
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0:
            continue
        e = float(subgraph_scaled[sg_idx])
        center_rows.append((sg_idx, i, j, e))

    print("=== DMFF JAX sGNN bond-centered contribution analysis (PEG4) ===")
    print(f"pdb: {args.pdb}")
    print(f"params: {args.params}")
    print(f"nn: {args.nn}, max_valence: {args.max_valence}")
    print(f"atype_index: {ATYPE_INDEX}")
    print(f"inferred n_layers: {n_layers}, sizes: {sizes}")
    print(f"total_energy (kcal/mol): {total_energy:.8f}")
    print(f"sum(subgraph_contrib_scaled) (kcal/mol): {float(jnp.sum(subgraph_scaled)):.8f}")
    print("note: total_energy = sum(subgraph_contrib_scaled) + mu")
    print("")

    if args.bond is None:
        picked = _find_default_ch_bond(elems, graph.bonds.tolist())
        if picked is None:
            print("No C-H bond found in topology.")
            return
        bi, bj = _normalize_pair(*picked)
        print(f"Auto-selected C-H bond: ({bi}, {bj})")
    else:
        bi, bj = _normalize_pair(int(args.bond[0]), int(args.bond[1]))
        print(f"Requested bond: ({bi}, {bj})")

    hits = []
    for sg_idx, i, j, e in center_rows:
        pi, pj = _normalize_pair(i, j)
        if (pi, pj) == (bi, bj):
            hits.append((sg_idx, i, j, e))

    if not hits:
        print("No matching center bond found in subgraphs.")
        return

    print("Bond contributions (kcal/mol):")
    for sg_idx, i, j, e in hits:
        print(f"  subgraph={sg_idx:3d}  bond={_format_pair(i, j, elems):>16s}  contrib={e: .8f}")
    print("")
    print(f"Bond contribution sum (kcal/mol): {sum(x[3] for x in hits):.8f}")
    print("warning: this is a model-local contribution, not a strict bond dissociation energy (BDE).")


if __name__ == "__main__":
    main()
