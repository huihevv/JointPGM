model_name=JointPGM

python -u train.py \
  --seed 2024 \
  --data national_illness \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --gc_deep 2 \
  --belta 0.01 \
  --alpha 0.6 \
  --embed_size 512 \
  --latent_size 512 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data national_illness \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 36 \
  --gc_deep 2 \
  --belta 0.01 \
  --alpha 0.6 \
  --embed_size 512 \
  --latent_size 512 \
  --batch_size 128 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data national_illness \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 48 \
  --gc_deep 2 \
  --belta 0.01 \
  --alpha 0.6 \
  --embed_size 512 \
  --latent_size 512 \
  --batch_size 64 \
  --lr 1e-3 \
  --gpu 1

python -u train.py \
  --seed 2024 \
  --data national_illness \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 60 \
  --gc_deep 2 \
  --belta 0.01 \
  --alpha 0.6 \
  --embed_size 512 \
  --latent_size 512 \
  --batch_size 64 \
  --lr 1e-3 \
  --gpu 1