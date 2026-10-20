"""Helpers to build tiny synthetic PDB files for tests."""
from __future__ import annotations


def atom_line(serial, name, resname, chain, resseq, x, y, z, *, record="ATOM",
              altloc=" ", occ=1.0, element=None, icode=" "):
    element = element or name.strip()[0]
    nm = f" {name:<3}" if (len(name) < 4 and len(element) == 1) else f"{name:<4}"
    return (f"{record:<6}{serial:>5} {nm}{altloc}{resname:>3} {chain}{resseq:>4}{icode}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{20.0:6.2f}          {element:>2}\n")


# side-chain atoms relative to residue origin x0: (name, dx, dy, dz, element)
_SIDE = {
    "ALA": [("CB", 1.46, 1.5, 0.0, "C")],
    "MET": [("CB", 1.46, 1.5, 0.0, "C"), ("CG", 1.46, 2.9, 0.5, "C"),
            ("SD", 1.46, 4.5, 0.5, "S"), ("CE", 1.46, 6.2, 0.5, "C")],
    "MSE": [("CB", 1.46, 1.5, 0.0, "C"), ("CG", 1.46, 2.9, 0.5, "C"),
            ("SE", 1.46, 4.5, 0.5, "SE"), ("CE", 1.46, 6.2, 0.5, "C")],
    "SEP": [("CB", 1.46, 1.5, 0.0, "C"), ("OG", 1.46, 2.9, 0.5, "O"),
            ("P", 1.46, 4.4, 0.5, "P"), ("O1P", 1.46, 5.6, 0.5, "O")],
    "TYR": [("CB", 1.46, 1.5, 0.0, "C"), ("CG", 1.46, 2.9, 0.5, "C"),
            ("OH", 1.46, 5.9, 0.5, "O")],
}


def peptide(chain, specs, *, x_start=0.0, y=0.0, z=0.0, serial_start=1):
    """Build a straight peptide chain with real peptide-bond geometry.

    specs: list of (resname, resseq, record). Consecutive residues are 3.8 A
    apart along x so that C(i)-N(i+1) = 1.33 A. Returns (lines, next_serial).
    """
    lines, serial = [], serial_start
    for i, (resname, resseq, record) in enumerate(specs):
        x0 = x_start + 3.8 * i
        backbone = [("N", 0.0, "N"), ("CA", 1.46, "C"), ("C", 2.47, "C")]
        for name, dx, el in backbone:
            lines.append(atom_line(serial, name, resname, chain, resseq, x0 + dx, y, z,
                                   record=record, element=el)); serial += 1
        lines.append(atom_line(serial, "O", resname, chain, resseq, x0 + 2.47, y + 1.2, z,
                               record=record, element="O")); serial += 1
        for name, dx, dy, dz, el in _SIDE.get(resname, _SIDE["ALA"]):
            lines.append(atom_line(serial, name, resname, chain, resseq, x0 + dx, y + dy, z + dz,
                                   record=record, element=el)); serial += 1
    return lines, serial


def hetero(resname, chain, resseq, n_atoms, center, *, serial_start, element="C"):
    """A blob of n HETATM atoms (0.9 A spacing along x) centred on `center`."""
    cx, cy, cz = center
    lines = []
    for k in range(n_atoms):
        x = cx + (k - (n_atoms - 1) / 2.0) * 0.9
        lines.append(atom_line(serial_start + k, f"C{k+1}", resname, chain, resseq, x, cy, cz,
                               record="HETATM", element=element))
    return lines, serial_start + n_atoms


def write_pdb(path, lines):
    with open(path, "w") as fh:
        fh.writelines(lines)
        fh.write("END\n")
    return str(path)


def parse_atoms(path, records=("ATOM", "HETATM")):
    out = []
    for line in open(path):
        if line[:6].strip() in records:
            out.append({
                "record": line[:6].strip(), "serial": int(line[6:11]),
                "name": line[12:16].strip(), "altloc": line[16], "resname": line[17:20].strip(),
                "chain": line[21], "resseq": int(line[22:26]),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "element": line[76:78].strip(),
            })
    return out
