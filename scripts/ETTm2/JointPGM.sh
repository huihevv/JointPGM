model_name=JointPGM

python -u train.py \
  --seed 2024 \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --gc_deep 2 \
  --belta 0.01 \
  --embed_size 128 \
  --latent_size 256 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --gc_deep 2 \
  --belta 0.01 \
  --embed_size 128 \
  --latent_size 256 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --gc_deep 2 \
  --belta 0.01 \
  --embed_size 128 \
  --latent_size 256 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1


python -u train.py \
  --seed 2024 \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --gc_deep 2 \
  --belta 0.01 \
  --embed_size 128 \
  --latent_size 256 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1