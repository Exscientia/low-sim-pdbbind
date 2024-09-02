datasets=(
    "pdb_bind_bespoke_ccdc"
)

higherSplits=(
    "[by_bespoke_5_fold_0,by_bespoke_5_fold_1,by_bespoke_5_fold_2]"
    "[by_bespoke_30_fold_0,by_bespoke_30_fold_1,by_bespoke_30_fold_2]"
    "[by_bespoke_80_fold_0,by_bespoke_80_fold_1,by_bespoke_80_fold_2]"
)

benchmrk_uniprots=(
    "P00918"
    "P56817"
    "Q9H2K2"
    "P00760"
    "P07900"
    "P24941"
    "P00734"
    "O60885"
)

features=(
    "mw"
    "ECFPMD"
    "ECFPMD_achiral"
    "FCFPMD_achiral"
    "atom_pairs_MD_achiral"
    "topo_torsion_MD_achiral"
)

models=(
    "random_forest_regressor"
    "svm_regressor"
    "catboost_regressor"
    "xg_boost_regressor"
)

for dataset in ${datasets[@]}; do
for uniprot in ${benchmrk_uniprots[@]}; do
for split in ${higherSplits[@]}; do
for feature in ${features[@]}; do
for model in ${models[@]}; do
CMD=$(cat <<-END
    dvc
    exp
    run
    -S dataset=$dataset
    -S filtering=uniprot_id
    -S filtering.0.value=$uniprot
    -S higher_split.presets.columns=$split
    -S featurisation=$feature
    -S train=$model
END

)
echo $CMD
eval $CMD
done
done
done
done
done

