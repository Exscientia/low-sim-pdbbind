from pathlib import Path
import torch
from openeye.oechem import (
    OEAddMols,
    OEMol,
    OEReadMolFromBytes,
    OESuppressHydrogens,
    OEWriteMolToBytes,
)
from openeye import oechem


def cut_off_pocket(mol1, mol2, cut_off):
    mol1_coords_dict = mol1.GetCoords()
    mol2_coords_dict = mol2.GetCoords()

    mol1_coords = torch.tensor(
        [list(mol1_coords_dict[idx]) for idx in range(len(mol1_coords_dict))]
    )
    mol2_coords = torch.tensor(
        [list(mol2_coords_dict[idx]) for idx in range(len(mol2_coords_dict))]
    )
    min_r_ij = (
        (mol1_coords.unsqueeze(1) - mol2_coords.unsqueeze(0)).norm(dim=-1).min(0)[0]
    )
    list_pocket_atoms = (min_r_ij <= cut_off).nonzero().squeeze().tolist()

    for atom in mol2.GetAtoms():
        if atom.GetIdx() not in list_pocket_atoms:
            assert mol2.DeleteAtom(atom)

    assert len(mol2.GetCoords()) == len(list_pocket_atoms)

    return mol1, mol2


def combine_mols(mol1, mol2):
    num_mol1 = len(mol1.GetCoords())

    OEAddMols(mol1, mol2)

    assert len(mol1.GetCoords()) == (num_mol1 + len(mol2.GetCoords()))

    return mol1


def add_bytes(
    x,
    cut_off,
    ligand_mol_column,
    receptor_path_column,
):
    receptor_path_column = Path(x[receptor_path_column])
    protein_mol = next(oechem.oemolistream(str(receptor_path_column)).GetOEGraphMols())

    ligand_mol = oechem.OEMol()
    oechem.OEReadMolFromBytes(ligand_mol, ".oeb", x[ligand_mol_column])

    ligand, protein = cut_off_pocket(ligand_mol, protein_mol, cut_off)

    for atom in ligand.GetAtoms():
        atom.SetType("0")

    for atom in protein.GetAtoms():
        atom.SetType("1")

    x["ligand_bytes"] = OEWriteMolToBytes(".oeb", ligand)
    x["pocket_bytes"] = OEWriteMolToBytes(".oeb", protein)

    ligand_pocket_combined = combine_mols(ligand, protein)
    x["ligand_pocket_bytes"] = OEWriteMolToBytes(".oeb", ligand_pocket_combined)

    return x


def suppress_hydrogens(
    x,
    which_hydrogens,
):
    ligand = OEMol()
    pocket = OEMol()
    ligand_pocket = OEMol()

    OEReadMolFromBytes(ligand, ".oeb", x["ligand_bytes"])
    OEReadMolFromBytes(pocket, ".oeb", x["pocket_bytes"])
    OEReadMolFromBytes(ligand_pocket, ".oeb", x["ligand_pocket_bytes"])
    if which_hydrogens == "explicit":
        pass
    elif which_hydrogens == "polar":
        # mol, retainPolar
        OESuppressHydrogens(ligand, True)
        OESuppressHydrogens(pocket, True)
        OESuppressHydrogens(ligand_pocket, True)
    elif which_hydrogens == "none":
        # mol, retainPolar
        OESuppressHydrogens(ligand, False)
        OESuppressHydrogens(pocket, False)
        OESuppressHydrogens(ligand_pocket, False)

    x["ligand_bytes"] = OEWriteMolToBytes(".oeb", ligand)
    x["pocket_bytes"] = OEWriteMolToBytes(".oeb", pocket)
    x["ligand_pocket_bytes"] = OEWriteMolToBytes(".oeb", ligand_pocket)

    return x
