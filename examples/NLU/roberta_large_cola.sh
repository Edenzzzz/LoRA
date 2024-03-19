r=8
if [ "$1" != "" ]; then
    r="$1"
fi
echo "Using rank = $r"

export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./cola_rank_${r}_model_init"
# python examples/text-classification/run_glue.py \
# --model_name_or_path roberta-large \
# --task_name cola \
# --do_eval \
# --max_seq_length 128 \
# --per_device_train_batch_size 4 \
# --learning_rate 2e-4 \
# --num_train_epochs 20 \
# --output_dir $output_dir/model \
# --logging_steps 100 \
# --logging_dir $output_dir/log \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --warmup_ratio 0.06 \
# --apply_lora \
# --lora_r $r \
# --seed 0 \
# --overwrite_output_dir \
# --weight_decay 0.1 \
# --do_tune

# Standalone training 
python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=5324 \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 4 \
--learning_rate 2e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--logging_steps 100 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r $r \
--seed 0 \
--save_total_limit 1 \
--overwrite_output_dir \
--weight_decay 0.1 