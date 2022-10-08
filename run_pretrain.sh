export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
node_rank=$1
name=pretrain100_epoch1m0.975_lr1e-4_warmup40
all_dir=/nlp_group/wuxing/suzhenpeng/mae_rdrop/all_visible/${name}
mkdir ${all_dir}
nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.150.143 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 3243  \
    --use_env main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.5 \
    --epochs 100 \
    --accum_iter 2 \
    --output_dir ${all_dir} \
    --log_dir ${all_dir} \
    --warmup_epochs 40 \
    --blr 1.0e-4 --weight_decay 0.05 \
    --norm_pix_loss \
    --data_path /share/a100/vig_pytorch/imagenet-2012 \
    >${all_dir}/${name}_${node_rank}.log 2>&1 &
    # --start_epoch 41 \
    # --resume output_dir/resnet_pretrain_100_beta1_resdd2/checkpoint-40.pth \