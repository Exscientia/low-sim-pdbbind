import itertools
import numpy as np
from openeye import oechem

# Define all residue types
all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']


def get_residue(residue):
    if residue in all_residues:
        return residue
    else:
        return 'OTH'


# Define all element types
all_elements = ['H', 'C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
Hal = ['F', 'Cl', 'Br', 'I']


def get_elementtype(e):
    if e in all_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'


keys = ["_".join(x) for x in list(itertools.product(all_residues, all_elements))]


def compute_features(complex_file, ncutoffs):
    ifs = oechem.oemolistream(complex_file)
    mol = next(ifs.GetOEGraphMols())

    residues = []
    res_num_tmp = None
    residue_atom_indices = []
    residue_atom_indices_tmp = []
    lig_indices = []
    all_ele = []
    lig_ele = []
    for i, atom in enumerate(mol.GetAtoms()):
        ele = str(atom).split(" ")[-1]
        all_ele.append(ele)

        res = oechem.OEAtomGetResidue(atom)
        res_name = res.GetName()
        res_num = res.GetResidueNumber()
        if res_name == "LIG":
            lig_indices.append(i)
            lig_ele.append(ele)

        if res_num_tmp is None:
            residues.append(res_name)
        elif res_num_tmp != res_num:
            residues.append(res_name)
            residue_atom_indices.append(residue_atom_indices_tmp)
            residue_atom_indices_tmp = []

        res_num_tmp = res_num
        residue_atom_indices_tmp.append(i)

    residue_atom_indices.append(residue_atom_indices_tmp)

    residues_indices = [i for i, res in enumerate(residues) if res != 'LIG']
    H_idxs = [i for i, x in enumerate(all_ele) if x == "H"]

    res_idxs_to_remove = []
    heavy_atom_indices = []

    for i in residues_indices:
        all_res_atom_idxs = residue_atom_indices[i]
        heavy_res_atom_idxs = [x for x in all_res_atom_idxs if not x in H_idxs]
        if len(heavy_res_atom_idxs) == 0:
            res_idxs_to_remove.append(i)
        else:
            heavy_atom_indices.append(heavy_res_atom_idxs)

    for idx in res_idxs_to_remove:
        residues_indices.remove(idx)

    residues = [residues[x] for x in residues_indices]
    coords = mol.GetCoords()
    xyz = np.array([coords[i] for i in range(len(coords))]) / 10

    distances = []
    for r_atoms in heavy_atom_indices:
        if len(r_atoms) == 0:
            continue
        for l_atom in lig_indices:
            ds = []
            for i in r_atoms:
                d = np.sqrt(np.sum(np.square(xyz[i] - xyz[l_atom])))
                ds.append(d)
            distances.append(min(ds))
    distances = np.array(distances)

    new_residue = list(map(get_residue, residues))
    new_lig = list(map(get_elementtype, lig_ele))

    residues_lig_atoms_combines = ["_".join(x) for x in list(itertools.product(new_residue, new_lig))]

    # calculate the number of contacts in different shells
    counts = []
    onion_counts = []
    for i, cutoff in enumerate(ncutoffs):
        counts_ = (distances <= cutoff) * 1
        if i == 0:
            onion_counts.append(counts_)
        else:
            onion_counts.append(counts_ - counts[-1])
        counts.append(counts_)

    results = []
    for n in range(len(ncutoffs)):
        d = {}
        d = d.fromkeys(keys, 0)
        for e_e, c in zip(residues_lig_atoms_combines, onion_counts[n]):
            d[e_e] += c
        results += d.values()

    return results
