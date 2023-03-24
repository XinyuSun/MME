# Set the path to save checkpoints
OUTPUT_DIR='exps/m3video/finetune/ucf101/k400pt-1600epo'

# path to ucf101 set (train.csv/val.csv/test.csv)
DATA_PATH='data/csv/ucf101'

# path to pretrain model
MODEL_PATH='exps/m3video/pretrain/k400/checkpoint-1600.pth'
MODEL='vit_base_patch16_224'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port 12320 --nnodes=$2  --node_rank=$3 --master_addr=$4 \
    finetune/run.py \
    --model ${MODEL} \
    --data_set HMDB51 \
    --nb_classes 51 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $5 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 10 \
    --test_num_crop 3 \
    --enable_deepspeed \
    --update_freq 4 \
    --crop_min 0.25 \
    ${@:6}