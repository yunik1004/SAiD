COEFFS_DIR="../output-inference"
OUTPUT_DIR="../out_render"

BASEDIR=$(dirname "$0")

for rdx in {0..71}; do
    python "$BASEDIR/test_render.py" --coeffs_dir $COEFFS_DIR --repeat_regex "-${rdx}" --output_dir $OUTPUT_DIR
done
