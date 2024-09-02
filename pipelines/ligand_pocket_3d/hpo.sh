datasets=(
    "pdb_bind_bespoke_ccdc"
)

hydrogens=(
    "none"
    "polar"
    "explicit"
)

higherSplits=(
    "[by_bespoke_0_fold_0,by_bespoke_0_fold_1,by_bespoke_0_fold_2]"
    "[by_bespoke_5_fold_0,by_bespoke_5_fold_1,by_bespoke_5_fold_2]"
    "[by_bespoke_30_fold_0,by_bespoke_30_fold_1,by_bespoke_30_fold_2]"
    "[by_bespoke_80_fold_0,by_bespoke_80_fold_1,by_bespoke_80_fold_2]"
)

# multi is only for multi egnns
features=(
    "atomic_num_only"
#    "multi_graph_atomic_num_only"
)

models=(
    "egnn"
    "egnn_pre_trained_qm"
    "egnn_pre_trained_diffusion"
#    "multi_graph_egnn"
#    "multi_graph_egnn_pre_trained_qm"
#    "multi_graph_egnn_pre_trained_diffusion"
)

for dataset in ${datasets[@]}; do
for which_hydrogen in ${hydrogens[@]}; do
for split in ${higherSplits[@]}; do
for feature in ${features[@]}; do
for model in ${models[@]}; do
CMD=$(cat <<-END
    dvc
    exp
    run
    -S dataset=$dataset
    -S higher_split.presets.columns=$split
    -S featurisation=$feature
    -S featurisation.which_hydrogens=$which_hydrogen
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
