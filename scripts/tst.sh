source .venv/bin/activate

output_dir="outputs/deep/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

model_name=patch_tst
for revin in 0 1
do
    python3 main.py \
        "model.name=$model_name" \
        "misc.outputdir=$output_dir" \
        "model.revin=$revin"
done

python3 -c "from src.visu.tables import print_nice_table; print_nice_table('${output_dir}/results.json')"