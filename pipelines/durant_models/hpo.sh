higherSplits=(
    "by_bespoke_0_fold_0"
    "by_bespoke_0_fold_1"
    "by_bespoke_0_fold_2"
    "by_bespoke_5_fold_0"
    "by_bespoke_5_fold_1"
    "by_bespoke_5_fold_2"
    "by_bespoke_30_fold_0"
    "by_bespoke_30_fold_1"
    "by_bespoke_30_fold_2"
    "by_bespoke_80_fold_0"
    "by_bespoke_80_fold_1"
    "by_bespoke_80_fold_2"
)

for split in ${higherSplits[@]}; do
CMD=$(cat <<-END
    dvc
    exp
    run
    -S dataset=pdb_bind_bespoke_ccdc
    -S model_repo=rfscore
    -S higher_split.presets.column=$split
END

)
echo $CMD
eval $CMD
done
