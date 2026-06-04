export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR" || exit 1

model_name=SECNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --num_channels 4 8 \
  --kernel_size 3 \
  --topk 8 \
  --patch_size 48 \
  --stride 24 \
  --dropout 0.1 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --num_workers 0 \
  --lradj type3 \
  --loss MAE \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --num_channels 4 8 \
  --kernel_size 3 \
  --topk 8 \
  --patch_size 48 \
  --stride 24 \
  --dropout 0.1 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_workers 0 \
  --lradj type3 \
  --loss MAE \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --num_channels 4 8 \
  --kernel_size 3 \
  --topk 8 \
  --patch_size 48 \
  --stride 24 \
  --dropout 0.1 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_workers 0 \
  --lradj type3 \
  --loss MAE \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --num_channels 4 8 \
  --kernel_size 3 \
  --topk 8 \
  --patch_size 48 \
  --stride 24 \
  --dropout 0.1 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --num_workers 0 \
  --lradj type3 \
  --loss MAE \
  --itr 1
