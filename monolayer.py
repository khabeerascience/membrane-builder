#!/usr/bin/env python3
"""
Build a single-layer (monolayer) of N x 18-MEA-Cys molecules for GROMACS.
Works with ANY atom naming: head/tail/anchor are found from elements & geometry.

  python monolayer.py 18_CYS_MEA.pdb --d 0.65 --lattice hex
  python monolayer.py 18_CYS_MEA.pdb --d 0.94

Outputs: monolayer.gro, posre_S.itp, build_info.txt
Orientation: Cys head (N) at bottom, methyl tail up (+z).
You can also optimize this script based on the molecular

If the script does not work as expected, use llm's to optimize it for the selected molecule. 
Furthermore, additional membrane components, such as lipids, sterols, or other amphiphilic 
molecules, can be incorporated based on literature-reported biological membrane compositions 
to better mimic the native membrane environment. 
"""
import argparse, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pdb")
ap.add_argument("--d", type=float, default=0.94, help="grafting distance (nm)")
ap.add_argument("--nx", type=int, default=10)
ap.add_argument("--ny", type=int, default=10)
ap.add_argument("--lattice", choices=["square", "hex"], default="square")
ap.add_argument("--lz", type=float, default=10.0, help="box height (nm)")
ap.add_argument("--z0", type=float, default=1.5, help="z of lowest atom (nm)")
ap.add_argument("--tilt", type=float, default=0.0, help="max random tilt (deg)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--anchor", default=None, help="atom NAME for restraint (default: first S atom)")
ap.add_argument("--out", default="monolayer.gro")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

# ---------- read PDB
names, resn, resi, elem, xyz = [], [], [], [], []
for l in open(a.pdb):
    if not l.startswith(("ATOM", "HETATM")):
        continue
    nm = l[12:16].strip()
    e = l[76:78].strip().upper() if len(l) >= 78 else ""
    if not e:
        e = re.sub(r"^[0-9]+", "", nm)[:1].upper()      # 1st letter of name
    names.append(nm); resn.append(l[17:21].strip() or "MOL")
    resi.append(int(l[22:26])); elem.append(e)
    xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
if not names:
    sys.exit("No ATOM/HETATM records found")
names = np.array(names); elem = np.array(elem); xyz = np.array(xyz) / 10.0  # A -> nm
heavy = elem != "H"
nat = len(names)
print(f"read {nat} atoms: " + ", ".join(f"{e}{(elem==e).sum()}" for e in sorted(set(elem))))

# ---------- long axis (PCA of heavy atoms)
com = xyz[heavy].mean(0)
axis = np.linalg.svd(xyz[heavy] - com)[2][0]

# head = N atom(s) (cysteine amine); fallback S; tail = heavy atom farthest from head
iN = np.where(elem == "N")[0]; iS = np.where(elem == "S")[0]
if len(iN):   head = xyz[iN].mean(0)
elif len(iS): head = xyz[iS].mean(0)
else:         sys.exit("Need an N or S atom to define the head group")
proj = (xyz[heavy] - head) @ axis
tail = xyz[heavy][np.argmax(np.abs(proj))]
if (tail - head) @ axis < 0:
    axis = -axis          # axis now points head -> tail

def rot_to_z(v):
    v = v / np.linalg.norm(v); z = np.array([0, 0, 1.0]); c = v @ z
    if c > 0.999999: return np.eye(3)
    if c < -0.999999: return np.diag([1, -1, -1.0])
    k = np.cross(v, z); s = np.linalg.norm(k); k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K

def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rtilt(tx, ty):
    cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

ref = xyz @ rot_to_z(axis).T
ref[:, :2] -= ref[heavy, :2].mean(0)
ref[:, 2] -= ref[:, 2].min()
length = ref[:, 2].max()
width = np.ptp(ref[heavy, :2], axis=0)

# ---------- lattice
d, nx, ny = a.d, a.nx, a.ny
if a.lattice == "square":
    bx, by = nx * d, ny * d
    pts = [(i * d, j * d) for j in range(ny) for i in range(nx)]
else:
    if ny % 2: sys.exit("hex lattice needs even --ny")
    dy = d * np.sqrt(3) / 2
    bx, by = nx * d, ny * dy
    pts = [(i * d + (0.5 * d if j % 2 else 0), j * dy) for j in range(ny) for i in range(nx)]
pts = np.array(pts) + d / 2
N = len(pts)

# ---------- place molecules
nres = len(set(resi))
rmap = {r: k for k, r in enumerate(sorted(set(resi)))}
lines = []; n = 0
for m, (px, py) in enumerate(pts):
    R = Rz(rng.uniform(0, 2 * np.pi))
    if a.tilt > 0:
        R = Rtilt(*np.radians(rng.uniform(-a.tilt, a.tilt, 2))) @ R
    p = ref @ R.T
    p[:, :2] += (px, py); p[:, 2] += a.z0
    for k in range(nat):
        n += 1
        rid = m * nres + rmap[resi[k]] + 1
        lines.append(f"{rid % 100000:5d}{resn[k]:<5s}{names[k]:>5s}{n % 100000:5d}"
                     f"{p[k,0]:8.3f}{p[k,1]:8.3f}{p[k,2]:8.3f}\n")
with open(a.out, "w") as f:
    f.write(f"18-MEA-Cys monolayer, {N} molecules, d={d} nm\n{n}\n")
    f.writelines(lines)
    f.write(f"{bx:10.5f}{by:10.5f}{a.lz:10.5f}\n")

# ---------- anchor restraint file (index inside one molecule)
if a.anchor:
    if a.anchor not in names: sys.exit(f"anchor '{a.anchor}' not found; atoms: {list(names)}")
    ai = list(names).index(a.anchor)
elif len(iS):
    ai = int(iS[0])
else:
    ai = int(iN[0])
with open("posre_S.itp", "w") as f:
    f.write("[ position_restraints ]\n; atom type fcx fcy fcz\n")
    f.write(f"{ai+1:5d} 1 1000 1000 1000   ; {names[ai]}\n")

info = f"""molecules: {N} ({nx} x {ny}, {a.lattice})   atoms/molecule: {nat}   total atoms: {n}
grafting distance d = {d} nm   area/molecule = {bx*by/N:.3f} nm^2
box = {bx:.3f} x {by:.3f} x {a.lz:.3f} nm
molecule length (z) = {length:.2f} nm, footprint = {width[0]:.2f} x {width[1]:.2f} nm
restraint atom: {names[ai]} (index {ai+1} in molecule)
head atom(s): N={[str(x) for x in names[iN]]}  S={[str(x) for x in names[iS]]}
"""
open("build_info.txt", "w").write(info); print(info)
