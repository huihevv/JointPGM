model_name=JointPGM

python -u train.py \
  --seed 2024 \
  --data metr \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --gc_deep 2 \
  --belta 0 \
  --alpha 0.6 \
  --embed_size 128 \
  --latent_size 128 \
  --batch_size 128 \
  --lr 1e-3 \
  --max_epochs 15 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data metr \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --gc_deep 2 \
  --belta 0 \
  --alpha 0.6 \
  --embed_size 128 \
  --latent_size 128 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data metr \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --gc_deep 2 \
  --belta 0 \
  --alpha 0.6 \
  --embed_size 128 \
  --latent_size 128 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data metr \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --gc_deep 2 \
  --belta 1e-5 \
  --alpha 0.6 \
  --embed_size 128 \
  --latent_size 128 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1