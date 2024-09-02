import os
import torch
from train import OnionNet2, IMGDataset
from pathlib import Path
from tqdm.auto import tqdm
from openeye import oechem
from generate_features import compute_features
import numpy as np
import datasets
import pandas as pd
import itertools
from sklearn import preprocessing
import joblib
import lightning
from torch.utils.data.dataloader import DataLoader

shells = 62
outermost = 0.05 * (shells + 1)
ncutoffs = np.linspace(0.1, outermost, shells)


def make_complex_pdb(key, protein_file, ligand_file, csv_file):
    ifs = oechem.oemolistream(ligand_file)
    mol = next(ifs.GetOEGraphMols())
    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_PDB)
    ofs.openstring()
    oechem.OEWriteMolecule(ofs, mol)
    lig_lines = ofs.GetString().decode("utf-8").split("\n")

    with open(protein_file, "r") as file:
        prot_lines = file.readlines()

    for line in prot_lines:
        if not line.startswith("ATOM"):
            prot_lines.remove(line)

    with open(
        f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_complex.pdb", "w"
    ) as f:
        for line in prot_lines:
            if line.startswith("ATOM"):
                f.write(line)

        for i, line in enumerate(lig_lines):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                new_line = line[:17] + "LIG" + line[20:]
                f.write(new_line + "\n")


def make_all_complexes(x, csv_file):
    key, protein_file, ligand_file = x["key"], x["protein_file"], x["ligand_file"]
    make_complex_pdb(key, protein_file, ligand_file, csv_file)
    return x
def generate_features_per_complex(x, csv_file):
    key, protein_file, ligand_file = x["key"], x["protein_file"], x["ligand_file"]
    try:
        features = compute_features(
            f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_complex.pdb",
            ncutoffs,
        )
    except:
        features = None

    x["features"] = features
    return x


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def generate_all_features(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)

    data = datasets.Dataset.from_dict(
        {
            "protein_file": protein_files,
            "ligand_file": ligand_files,
            "key": keys,
            "pk": pks,
        }
    )
    os.makedirs(f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}")

    data = data.map(lambda x: make_all_complexes(x, csv_file), num_proc=4, batch_size=1)
    data = data.map(lambda x: generate_features_per_complex(x, csv_file), num_proc=4, batch_size=1)
    os.system(
        f"rm -r data/scratch/"
    )
    all_features = data["features"]
    print(len(all_features))
    keys = [x for x, y in zip(keys, all_features) if y is not None]
    pks = [x for x, y in zip(pks, all_features) if y is not None]
    all_features = [x for x in all_features if x is not None]
    print(len(all_features))
    all_features = np.array(all_features)
    all_elements = ["H", "C", "O", "N", "P", "S", "Hal", "DU"]
    all_residues = [
        "GLY",
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "PRO",
        "PHE",
        "TYR",
        "TRP",
        "SER",
        "THR",
        "CYS",
        "MET",
        "ASN",
        "GLN",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
        "HIS",
        "OTH",
    ]
    feat_keys = [
        "_".join(x) for x in list(itertools.product(all_residues, all_elements))
    ]
    columns = []
    for i, n in enumerate(feat_keys * len(ncutoffs)):
        columns.append(f"{n}_{i}")
    return pd.DataFrame(all_features, columns=columns), pd.DataFrame({"pk": pks, "key": keys})


def train_model(args, model_name):
    print("creating model")
    model = OnionNet2()
    full = pd.read_csv(
        f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv',
        index_col=0,
    )
    val = full.sample(1000, random_state=42)
    train = full.drop(val.index)
    n_features = 21 * 8 * 62
    X_train = train.values[:, :n_features]
    X_valid = val.values[:, :n_features]
    scaler = preprocessing.StandardScaler()
    X_train_std = scaler.fit_transform(X_train).reshape([-1] + args.shape)
    X_valid_std = scaler.transform(X_valid).reshape([-1] + args.shape)
    joblib.dump(scaler, f"data/models/{model_name}.scaler")
    all_y = pd.read_csv(
        f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_pks.csv',
    )["pk"].values
    y_train = all_y[train.index]
    y_valid = all_y[val.index]
    stop = lightning.pytorch.callbacks.EarlyStopping(
        monitor="val/loss",
        min_delta=0.001,
        patience=args.patience,
        mode="min",
    )
    bestmodel = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=f"data/models/{model_name}", save_top_k=1
    )
    trainer = lightning.Trainer(
        max_epochs=300,
        accelerator="auto",
        strategy="auto",
        callbacks=[stop, bestmodel]
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(IMGDataset(X_train_std, y_train), batch_size=128, shuffle=True, num_workers=os.cpu_count()),
        val_dataloaders=DataLoader(IMGDataset(X_valid_std, y_valid), batch_size=128, num_workers=os.cpu_count()),
    )


def predict(args):
    test = pd.read_csv(
        f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv',
        index_col=0,
    )
    test_index = pd.read_csv(
        f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_pks.csv',
    )["key"].values
    true_values = pd.read_csv(
        f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_pks.csv',
    )["pk"].values

    X_test = test.values

    scaler = joblib.load(f"data/models/{args.model_name}.scaler")
    X_test_std = scaler.transform(X_test).reshape([-1] + args.shape)

    paths = []
    for p in Path(f"data/models/{args.model_name}").iterdir():
        paths.append(str(p.parts[-1]))
    ckpt = torch.load(
        f"data/models/{args.model_name}/{paths[0]}", map_location="cpu"
    )
    model = OnionNet2()
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval()
    X_test_std = torch.tensor(X_test_std, dtype=torch.float32)
    outs = []
    for x in tqdm(X_test_std):
        x = x.unsqueeze(0)
        outs.append(float(model(x)))

    return pd.DataFrame({"key": test_index, "pred": outs, "pk": true_values})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str, default="val.csv")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--shape",
        type=int,
        default=[1, 84, 124],
        nargs="+",
        help="Input. Reshape the features.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Input. The learning rate."
    )
    parser.add_argument(
        "--batchs",
        type=int,
        default=64,
        help="Input. The number of samples processed per batch.",
    )
    parser.add_argument(
        "--rate", type=float, default=0.0, help="Input. The dropout rate."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Input. The alpha value in loss function.",
    )
    parser.add_argument(
        "--clipvalue",
        type=float,
        default=0.01,
        help="Input. The threshold for gradient clipping.",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=10416,
        help="Input. The number of features for each complex. \n"
        "When shells N=62, n_feautes=21*8*62.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Input. The number of times all samples in the training set pass the CNN model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Input. Number of epochs with no improvement after which training will be stopped.",
    )
    args = parser.parse_args()
    if args.train:
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        if not os.path.exists("data/features"):
            os.makedirs("data/features")
        if not os.path.exists(
            f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'
        ):
            df_features, df_pk = generate_all_features(args.csv_file, args.data_dir)
            df_features.to_csv(
                f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'
            )
            df_pk.to_csv(
                f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_pks.csv'
            )
        if args.gpus > 0:
            train_model(args, args.model_name)
        else:
            raise ValueError("Please use a GPU to train the model.")
    if args.predict:
        if not os.path.exists(
            f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'
        ):
            df_features, df_pk = generate_all_features(args.val_csv_file, args.val_data_dir)
            df_features.to_csv(
                f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'
            )
            df_pk.to_csv(
                f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_pks.csv'
            )
        df = predict(args)
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    if not args.train and not args.predict:
        raise ValueError("Please specify --train or --predict or both.")
