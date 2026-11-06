#!/usr/bin/env python3
"""
Build a single-layer (monolayer) of 100 x 18-MEA-Cys molecules for GROMACS.

Geometry: S-acyl-cysteine head groups at the bottom (z ~ z_offset, the 'protein'
side, where Cys is anchored), anteiso methyl tails pointing up (+z).
Molecules sit on a 10 x 10 periodic lattice (square or hexagonal) whose
nearest-neighbour spacing is --d (nm).

Usage:
  python build_mea_monolayer.py cys_18mea_chimera_gmx.pdb --d 0.94
  python build_mea_monolayer.py cys_18mea_chimera_gmx.pdb --d 0.65 --lattice hex
Outputs: monolayer.gro, posre_S.itp (optional S-atom restraints), build_info.txt
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pdb")
ap.add_argument("--d", type=float, default=0.94, help="grafting (nearest-neighbour) distance, nm")
ap.add_argument("--nx", type=int, default=10)
ap.add_argument("--ny", type=int, default=10)
ap.add_argument("--lattice", choices=["square", "hex"], default="square")
ap.add_argument("--lz", type=float, default=10.0, help="box height, nm (vacuum/water above)")
ap.add_argument("--z0", type=float, default=1.5, help="z of lowest atom, nm")
ap.add_argument("--tilt", type=float, default=0.0, help="max random tilt, degrees")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--anchor", default="S64", help="atom name used for restraint file")
ap.add_argument("--out", default="monolayer.gro")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

# ---- read PDB (Angstrom -> nm)
names, res, xyz = [], None, []
for l in open(a.pdb):
    if l.startswith(("ATOM", "HETATM")):
        names.append(l[12:16].strip()); res = l[17:20].strip()
        xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
xyz = np.array(xyz) / 10.0
names = np.array(names)
heavy = np.array([not n.startswith("H") for n in names])

def rot_to_z(v):
    v = v / np.linalg.norm(v); z = np.array([0, 0, 1.0])
    c = v @ z
    if c > 0.999999: return np.eye(3)
    if c < -0.999999: return np.diag([1, -1, -1.0])
    k = np.cross(v, z); s = np.linalg.norm(k); k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K

def Rz(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rtilt(tx, ty):
    cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

# ---- orient long axis (principal axis of heavy atoms) -> +z, Cys head down, methyl tail up
com = xyz[heavy].mean(0)
u, s, vt = np.linalg.svd(xyz[heavy] - com)
axis = vt[0]
head = xyz[[i for i, n in enumerate(names) if n in ("N73", "C69")]].mean(0)
tail = xyz[[i for i, n in enumerate(names) if n in ("C1", "C28")]].mean(0)
if (tail - head) @ axis < 0: axis = -axis
ref = xyz @ rot_to_z(axis).T
ref -= ref[:, :2].mean(0) * np.array([1, 1, 0]) if False else 0
cxy = ref[heavy, :2].mean(0); ref[:, :2] -= cxy
ref[:, 2] -= ref[:, 2].min()
length = ref[:, 2].max()
width = np.ptp(ref[heavy, :2], axis=0)

# ---- lattice
d, nx, ny = a.d, a.nx, a.ny
if a.lattice == "square":
    bx, by = nx * d, ny * d
    pts = [(i * d, j * d) for j in range(ny) for i in range(nx)]
else:  # hexagonal, rows shifted by d/2; ny must be even for periodicity
    assert ny % 2 == 0, "hex lattice needs even ny"
    dy = d * np.sqrt(3) / 2
    bx, by = nx * d, ny * dy
    pts = [(i * d + (0.5 * d if j % 2 else 0), j * dy) for j in range(ny) for i in range(nx)]
pts = np.array(pts) + d / 2
N = len(pts)
area_per = bx * by / N

# ---- place
atoms = []
for m, (px, py) in enumerate(pts):
    R = Rz(rng.uniform(0, 2 * np.pi))
    if a.tilt > 0:
        R = Rtilt(*np.radians(rng.uniform(-a.tilt, a.tilt, 2))) @ R
    p = ref @ R.T
    p[:, :2] += (px, py); p[:, 2] += a.z0
    for n, q in zip(names, p): atoms.append((m + 1, n, q))

with open(a.out, "w") as f:
    f.write(f"18-MEA-Cys monolayer, {N} molecules, d={d} nm\n{len(atoms)}\n")
    for i, (m, n, q) in enumerate(atoms, 1):
        f.write(f"{m % 100000:5d}{res:<5s}{n:>5s}{i % 100000:5d}{q[0]:8.3f}{q[1]:8.3f}{q[2]:8.3f}\n")
    f.write(f"{bx:10.5f}{by:10.5f}{a.lz:10.5f}\n")

# ---- position restraints on anchor atom (mimics grafting on the protein surface)
with open("posre_S.itp", "w") as f:
    f.write("[ position_restraints ]\n; atom  type  fcx  fcy  fcz   (apply on S/anchor atoms; atom index within molecule)\n")
    idx = list(names).index(a.anchor) + 1
    f.write(f"{idx:5d} 1 1000 1000 1000\n")

info = f"""molecules: {N} ({a.nx} x {a.ny}, {a.lattice})
grafting distance d: {d} nm   area/molecule: {area_per:.3f} nm^2
box: {bx:.3f} x {by:.3f} x {a.lz:.3f} nm
single molecule: length along z {length:.2f} nm, xy footprint {width[0]:.2f} x {width[1]:.2f} nm
net charge: +{N} (NH3+ / COOH) -> add {N} Cl- (gmx genion) or neutralise as required
"""
open("build_info.txt", "w").write(info); print(info)
