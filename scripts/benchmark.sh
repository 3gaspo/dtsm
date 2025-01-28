source .venv/bin/activate

output_dir="outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in persistence repeat lookback linear MLP patch_tst
do
    if [ "$model_name" = "patch_tst" ]; then
        for revin in 0 1
        do
            python3 benchmark.py \
                "model.name=$model_name" \
                "misc.outputdir=$output_dir" \
                "model.revin=$revin" \
                "training.loss=huber"
        done
    else
        python3 benchmark.py \
            "model.name=$model_name" \
            "misc.outputdir=$output_dir" \
            "training.loss=huber"
    fi
done

python3 -c "from src.visu.tables import print_nice_table; print_nice_table('${output_dir}/results.json')"
python3 losses.py misc.outputdir=$output_dir/benchmark/