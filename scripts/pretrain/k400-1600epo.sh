# Set the path to save checkpoints
OUTPUT_DIR='exps/m3video/pretrain/k400'

# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='data/csv/k400/train.csv'
MODEL='pretrain_videomae_base_patch16_224'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$1 \
        --master_port 12320 --nnodes=$2 --node_rank=$3 --master_addr=$4 \
        pretrain/run.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.7 \
        --model ${MODEL} \
        --decoder_num 1 \
        --decoder_depth 1 \
        --batch_size $5 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 50 \
        --epochs 1601 \
        --exit_epoch 1601 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --target_feature 'idt' \
        --traj_len 4 \
        --traj_max_len 4 \
        --traj_norm 'PatchStd' \
