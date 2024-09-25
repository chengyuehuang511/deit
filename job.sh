#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,optimistprime,hk47,ig-88,omgwth,qt-1,sonny,robby,samantha,trublu,,synapse,dendrite,deebot
#SBATCH --mem-per-gpu=50G

cd /coc/testnvme/chuang475/projects/deit

# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main_finetune.py --arch clip_resnet50 --id SGDD_norm --opt sgdd --lr 1e-3 --mu 100 --data_dir ./datasets/domainnet --percent 100 --epoch 50 --gpu_per_node 4 --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt --batch_size 64 --wd 5e-4
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main_finetune.py --arch resnet50 --id moco_vanilla --percent 100 --lr 1e-2 --opt sgdd --mu 1 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir ./datasets/domainnet --load_pretrained ./pre_trained/mocov3_resnet50_pretrain.tar --wd 5e-4
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main_finetune.py --arch resnet50 --id moco_vanilla --percent 100 --lr 5e-2 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir ./datasets/domainnet --load_pretrained ./pre_trained/mocov3_resnet50_pretrain.tar --wd 5e-4
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main_finetune.py --arch clip_resnet50 --id FTP_clip --opt sgdp --lr 1e-3 --data_dir ./datasets/domainnet --percent 100 --epoch 50 --gpu_per_node 4 --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt --batch_size 64 --wd 5e-4
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python DomainNet_ResNet_Exp/main_finetune.py --arch clip_resnet50 --id clip_vanilla --percent 100 --lr 1e-3 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir ./datasets/domainnet --meta_dir ./datasets/ --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main.py --output_dir ./log/ImageNet --load_pretrained ./pre_trained_flash/clip_vitbase16_pretrain.pt --load_head ./pre_trained_flash/clip_vitbase16_pretrain_head.pt --data_path /srv/kira-lab/share4/datasets/ImageNet --batch_size 256 --gpu_per_node 2 --opt adam --lr 2e-5 --epoch 30 --wd 0.1
# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python main_finetune_imagenet.py --output_dir ./log/ImageNet --load_pretrained ./pre_trained_flash/clip_vitbase16_pretrain.pt --load_head ./pre_trained_flash/clip_vitbase16_pretrain_head.pt --data_path /srv/kira-lab/share4/datasets/ImageNet --batch_size 256 --gpu_per_node 8 --opt adam --lr 1e-4 --epoch 30 --wd 0.1

# {"id": "clip_vanilla_2e-5", "batch_size": "256", "epochs": "30", "bce_loss": "False", "unscale_lr": "False", "model": "clip_base_patch16_224", "input_size": "224", "drop": "0.0", "drop_path": "0.2", "model_ema": "False", "model_ema_decay": "0.99996", "model_ema_force_cpu": "False", "opt": "adamw", "opt_eps": "1e-08", "opt_betas": "None", "clip_grad": "None", "momentum": "0.9", "weight_decay": "0.1", "sched": "cosine", "lr": "2e-05", "lr_noise": "None", "lr_noise_pct": "0.67", "lr_noise_std": "1.0", "warmup_lr": "1e-06", "min_lr": "1e-05", "decay_epochs": "30", "warmup_epochs": "5", "cooldown_epochs": "10", "patience_epochs": "10", "decay_rate": "0.1", "color_jitter": "0.3", "aa": "rand-m9-mstd0.5-inc1", "smoothing": "0.1", "train_interpolation": "bicubic", "repeated_aug": "False", "train_mode": "True", "ThreeAugment": "False", "src": "False", "reprob": "0.0", "remode": "pixel", "recount": "1", "resplit": "False", "mixup": "0.8", "cutmix": "1.0", "cutmix_minmax": "None", "mixup_prob": "1.0", "mixup_switch_prob": "0.5", "mixup_mode": "batch", "teacher_model": "regnety_160", "teacher_path": "", "distillation_type": "none", "distillation_alpha": "0.5", "distillation_tau": "1.0", "finetune": "/srv/kira-lab/share4/jtian73/clip_vitbase16_pretrain.pt", "attn_only": "False", "load_head": "/srv/kira-lab/share4/jtian73/clip_vitbase16_pretrain_head.pt", "data_path": "/srv/kira-lab/share4/datasets/ImageNet/imagenet", "data_set": "IMNET", "inat_category": "name", "output_dir": "/srv/kira-lab/share4/jtian73/checkpoint/deit/experiments/clip_vanilla_2e-5/22_04_2023_23_39_29", "device": "cuda", "seed": "0", "resume": "", "start_epoch": "0", "eval": "False", "eval_crop_ratio": "1.0", "dist_eval": "False", "num_workers": "10", "pin_mem": "True", "num_nodes": "1", "gpu_per_node": "2", "node_rank": "0", "dist_url": "env://", "dist_backend": "NCCL", "world_size": "1", "global_pool": "True", "blr": "0.001", "layer_decay": "None", "accum_iter": "1", "k": "1.0"}

# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=<addr> --master_port=<port>  main.py \
#     --output_dir /coc/testnvme/chuang475/projects/deit/output/clip_vanilla_2e-5/ \
#     --data-path /srv/kira-lab/share4/datasets/ImageNet/imagenet \
#     --model deit_tiny_patch16_224 \
#     --batch-size 256 \
#     --opt adamw \
#     --lr 2e-5 \
#     --epoch 30 \
#     --weight-decay 0.1 \
#     --drop 0.0 \
#     --drop-path 0.2 \
#     --input-size 224 \
#     --model clip_base_patch16_224 \
#     --train-mode \
#     --sched cosine \
#     --warmup-lr 1e-6 \
#     --min-lr 1e-5 \
#     --decay-epochs 30 \
#     --warmup-epochs 5 \
#     --cooldown-epochs 10 \
#     --color-jitter 0.3 \
#     --aa rand-m9-mstd0.5-inc1 \
#     --smoothing 0.1 \
#     --train-interpolation bicubic \
#     --mixup 0.8 \
#     --cutmix 1.0 \
#     --mixup-prob 1.0 \
#     --mixup-switch-prob 0.5 \
#     --mixup-mode batch \
#     --teacher-model regnety_160 \
#     --distillation-type none \
#     --distillation-alpha 0.5 \
#     --distillation-tau 1.0 \
#     --device cuda \
#     --seed 0 \
#     --num_workers 10 \

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model clip_vit_base16 --batch-size 256 --data-path /srv/kira-lab/share4/datasets/ImageNet/imagenet --output_dir /coc/testnvme/chuang475/projects/deit/output/clip_vanilla_2e-5/  --load_pretrained /coc/pskynet4/chuang475/projects/zeroshot_finetune/pre_trained_flash/clip_vitbase16_pretrain.pt --load_head /coc/pskynet4/chuang475/projects/zeroshot_finetune/pre_trained_flash/clip_vitbase16_pretrain_head.pt