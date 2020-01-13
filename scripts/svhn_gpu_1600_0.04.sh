# coding=utf-8


task_name=svhn

python main_1600_0.04.py \
  --use_tpu=False \
  --do_eval_along_training=True\
  --do_train=True \
  --do_eval=True \
  --task_name=${task_name} \
  --sup_size=1000 \
  --unsup_ratio=5 \
  --data_dir=../data_svhn/proc_data/${task_name} \
  --model_dir=ckpt/svhn_gpu_1600_0.04_5 \
  --learning_rate=0.05  \
  --weight_decay_rate=7e-4   \
  --aug_copy=100  \
  --max_save=20   \
  --curr_step=0   \
  --iterations=10000 \
  --train_steps=200000 \
  $@
