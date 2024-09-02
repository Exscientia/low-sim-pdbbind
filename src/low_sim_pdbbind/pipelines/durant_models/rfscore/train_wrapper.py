from oddt.toolkits import ob
from oddt.scoring import descriptors
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle


def load_csv(csv_file, data_dir):
    """
    Loads CSV file specfiying data and adds full path to protein and ligand files

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files

    Returns
    -------
    keys : list
        List of keys
    protein_files : list
        List of paths to protein files
    ligand_files : list
        List of paths to ligand files
    pks : list
        List of pK values
    """
    df = pd.read_csv(csv_file)
    keys = df["key"].values
    protein_files = [os.path.join(data_dir, file) for file in df["protein"].values]
    ligand_files = [os.path.join(data_dir, file) for file in df["ligand"].values]
    pks = df["pk"].values
    return keys, protein_files, ligand_files, pks


def generate_feature(protein_file, ligand_file, cutoff):
    """
    Generates RFScore features for a single protein-ligand complex

    Parameters
    ----------
    protein_file : str
        Path to protein PDB file
    ligand_file : str
        Path to ligand SDF file
    cutoff : float
        Distance cutoff for features

    Returns
    -------
    dict
        Dictionary with feature names and values
    """

    protein = next(ob.readfile("pdb", protein_file))
    protein.protein = True
    ligand = next(ob.readfile("sdf", ligand_file))
    rfscore_engine = descriptors.close_contacts_descriptor(
        protein=protein,
        cutoff=cutoff,
        ligand_types=[6, 7, 8, 9, 15, 16, 17, 35, 53],
        protein_types=[6, 7, 8, 16],
    )
    return {
        name: value
        for name, value in zip(rfscore_engine.titles, rfscore_engine.build(ligand)[0])
    }


def batch_generate_features(csv_file, data_dir, cutoff):
    """
    Generates RFScore features for a list of protein-ligand complexes

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features

    Returns
    -------
    features_df : pandas.DataFrame
        DataFrame with features for all data in CSV file
    pks : list
        List of pK values
    keys : list
        List of keys for each protein-ligand complex
    """
    keys, protein_files, ligand_files, pks = load_csv(csv_file, data_dir)
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    if not os.path.exists(
        f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv'
    ):
        with Parallel(n_jobs=-1) as parallel:
            features = parallel(
                delayed(generate_feature)(protein_files[i], ligand_files[i], cutoff)
                for i in tqdm(range(len(keys)))
            )
        features_df = pd.DataFrame(features)
        features_df.to_csv(
            f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv',
            index=False,
        )
    else:
        features_df = pd.read_csv(
            f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv'
        )
    return features_df, pks, keys


def train_model(csv_file, data_dir, cutoff):
    """
    Trains a Random Forest model on RFScore features of protein-ligand complexes

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    features_df, pks, keys = batch_generate_features(csv_file, data_dir, cutoff)
    print("Ready to train model")
    model = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    )
    model.fit(features_df, pks)
    return model


def predict(model, csv_file, data_dir, cutoff):
    """
    Predicts pK values for a list of protein-ligand complexes using a trained Random Forest model

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features

    Returns
    -------
    pandas.DataFrame
        DataFrame with predicted pK values
    """
    features_df, pks, keys = batch_generate_features(csv_file, data_dir, cutoff)
    pred_pK = model.predict(features_df)
    return pd.DataFrame({"key": keys, "pred": pred_pK, "pk": pks})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--csv_file", help="Training CSV file with protein, ligand and pk data"
    )
    parser.add_argument(
        "--data_dir", help="Training directory with protein and ligand files"
    )
    parser.add_argument(
        "--val_csv_file", help="Test CSV file with protein, ligand and pk data"
    )
    parser.add_argument(
        "--val_data_dir", help="Test directory with protein and ligand files"
    )
    parser.add_argument(
        "--model_name", help="Name of the model to be saved or to be loaded"
    )
    parser.add_argument(
        "--cutoff", help="Distance cutoff for features", default=12, type=int
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Predict pK")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache temporary files in data/scratch directory",
    )
    args = parser.parse_args()
    if args.train:
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        model = train_model(args.csv_file, args.data_dir, args.cutoff)
        with open(f"data/models/{args.model_name}.pkl", "wb") as handle:
            pickle.dump(model, handle)
    if args.predict:
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        with open(f"data/models/{args.model_name}.pkl", "rb") as handle:
            model = pickle.load(handle)
        results_df = predict(model, args.val_csv_file, args.val_data_dir, args.cutoff)
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        results_df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    if not args.train and not args.predict:
        print("Please specify --train and/or --predict")
