export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

node_rank=$1
name=ft/pretrain100_epoch1m0.95_lr1e-4_warmup40
all_dir=/nlp_group/wuxing/suzhenpeng/mae_rdrop/output_dir/${name}
mkdir ${all_dir}


nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.152.11 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 23332  \
    --use_env main_finetune.py  \
    --finetune output_dir/pretrain100_epoch1m0.95_lr1e-4_warmup40/checkpoint-99.pth \
    --output_dir ${all_dir} \
    --log_dir ${all_dir} \
    --accum_iter 1 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path /share/a100/vig_pytorch/imagenet-2012 \
    # --start_epoch 41 \
    # --resume output_dir/resnet_pretrain_100_beta1_resdd2/checkpoint-40.pth \
    >${all_dir}/${node_rank}.log 2>&1 &