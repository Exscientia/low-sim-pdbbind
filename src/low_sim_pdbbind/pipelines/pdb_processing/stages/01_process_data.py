"""
Script to carry out data preprocessing.
"""

import logging
from pathlib import Path

import dvc.api
from openeye import oechem
from datasets import disable_caching
import molflux.datasets
from low_sim_pdbbind.utils.dir import get_pipeline_dir
from molflux.features.representations.openeye.canonical._utils import standardise_oemol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

disable_caching()


def add_canonical_smiles(x):
    mol = oechem.OEMol()
    oechem.OEReadMolFromBytes(mol, ".oeb", x["ligand_bytes"])
    x["canonical_smiles"] = oechem.OEMolToSmiles(standardise_oemol(mol))
    return x


def main() -> None:
    cfg = dvc.api.params_show()

    dataset = molflux.datasets.load_dataset_from_store(cfg["pdbbind_dataset_path"])

    def add_lig_bytes(x):
        try:
            lig_path = Path(cfg["ligands_path"]) / f"{x['pdb_code']}-ligand.sdf.gz"
            ligand = next(oechem.oemolistream(str(lig_path)).GetOEGraphMols())
            x["ligand_bytes"] = oechem.OEWriteMolToBytes(".oeb", ligand)
        except:
            x["ligand_bytes"] = None
        return x

    processed_dataset = dataset.map(add_lig_bytes)
    processed_dataset = processed_dataset.filter(
        lambda x: x["ligand_bytes"] is not None
    )

    def add_prot_paths(x):
        prot_path = Path(cfg["structures_path"]) / f"{x['pdb_code']}.pdb.gz"
        x["protein_path"] = str(prot_path)
        return x

    processed_dataset = processed_dataset.map(add_prot_paths)
    processed_dataset = processed_dataset.map(add_canonical_smiles)

    logger.info("Saving processed dataset")
    dataset_processed_dir = get_pipeline_dir() / "data" / "dataset_processed.parquet"
    molflux.datasets.save_dataset_to_store(
        processed_dataset, str(dataset_processed_dir), format="parquet"
    )


if __name__ == "__main__":
    main()
