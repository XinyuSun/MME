# Set the path to save checkpoints
OUTPUT_DIR='exps/m3video/probing/ssv2/ssv2pt-400epo-lp-30epo'
# path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='data/csv/ssv2'
# path to pretrain model
MODEL_PATH='exps/m3video/pretrain/ssv2/checkpoint-399.pth'

# batch_size can be adjusted according to number of GPUs
OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port 12320 --nnodes=$2  --node_rank=$3 --master_addr=$4 \
    finetune/run.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --finetune ${MODEL_PATH} \
    --batch_size $5 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --update_freq 1 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 30 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --crop_min 0.25 \
    --linear_probing \
    ${@:6}