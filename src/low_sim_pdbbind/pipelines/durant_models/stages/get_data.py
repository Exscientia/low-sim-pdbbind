import os
from pathlib import Path
import logging

import dvc.api
from datasets import disable_caching

from cloudpathlib import CloudPath
import molflux.datasets
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from openeye.oechem import OEReadMolFromBytes, OEMol, oemolostream, OEWriteMolecule, oemolistream

disable_caching()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def write_mol(mol, path):
    with oemolostream(path) as oss:
        mol_copy = mol.CreateCopy()
        return OEWriteMolecule(oss, mol_copy)

def read_struct_from_pdb(path):
    mol = next(oemolistream(path).GetOEGraphMols())
    return mol

def main() -> None:
    cfg = dvc.api.params_show()

    log.info("Loading dataset")

    dataset = molflux.datasets.load_dataset_from_store(cfg["dataset"]["path"])

    data_dir = get_pipeline_dir() / "data" / "data_dump"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "ligands").mkdir(parents=True, exist_ok=True)
    (data_dir / "proteins").mkdir(parents=True, exist_ok=True)
    def download_datapoint(x):
        try:
            # read and write ligand
            mol = OEMol()
            OEReadMolFromBytes(mol, ".oeb", x[cfg["dataset"]["ligand_mol_column"]])
            write_mol(mol, str(data_dir / "ligands" / f"{x[cfg['key']]}.sdf"))

            # read and write protein
            protein_path = Path(x[cfg["dataset"]["receptor_path_column"]])
            protein = read_struct_from_pdb(str(protein_path))
            write_mol(protein, str(data_dir / "proteins" / protein_path.parts[-1].replace(".gz", "")))
            x["success"] = True
        except Exception as e:
            x["success"] = False
            print(f"failed: {x['pdb_code']}, {e}")

        return x

    dataset = dataset.map(download_datapoint, num_proc=8)
    dataset = dataset.filter(lambda x: x["success"])

    log.info("saving fetched dataset")
    dataset_dir = get_pipeline_dir() / "data" / "dataset.parquet"
    molflux.datasets.save_dataset_to_store(dataset, str(dataset_dir), format="parquet")

if __name__ == "__main__":
    main()
