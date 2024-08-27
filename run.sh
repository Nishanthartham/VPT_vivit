# CUDA_VISIBLE_DEVICES=6,7 python finetune_vivit.py 10-class-SNR005 2>&1 1>log.simulated
echo $1
echo $2
CUDA_VISIBLE_DEVICES=$2 python finetune_vivit_224_224.py $1 2>&1 1>log-$1.video
