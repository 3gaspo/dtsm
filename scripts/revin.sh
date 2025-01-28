source .venv/bin/activate

output_dir="outputs/revin/"
rm -rf "$output_dir"
mkdir -p "$output_dir"
model_name=MLP
for revin in 0 1
do
    for loss in mse huber
    do
    python3 benchmark.py \
        "model.name=$model_name" \
        "misc.outputdir=$output_dir" \
        "model.revin=$revin" \
        "training.loss=$loss"
    done
done

python3 -c "from src.visu.tables import print_nice_table; print_nice_table('${output_dir}/results.json')"