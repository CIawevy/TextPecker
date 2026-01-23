SAMPLE_FOLDER=/path/to/your/generation_results
MODE=en # en or zh
OUTPUT_DIR=./eval_results

torchrun --nnodes=1 --node-rank=0 --nproc_per_node=8 \
    evaluate_text_reward.py \
    --sample_dir $SAMPLE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --mode $MODE

cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results.jsonl
rm $OUTPUT_DIR/results_chunk*.jsonl

python3 summary_scores.py $OUTPUT_DIR/results.json --mode $MODE


