# coding=utf-8


task_name=cifar10

python main_400_0.2.py \
  --use_tpu=False \
  --do_eval_along_training=True \
  --do_train=True \
  --do_eval=True \
  --task_name=${task_name} \
  --sup_size=4000 \
  --unsup_ratio=5 \
  --data_dir=../data/proc_data/${task_name} \
  --learning_rate=0.03  \
  --weight_decay_rate=5e-4   \
  --model_dir=ckpt/cifar10_gpu_400_0.2_5 \
  --aug_copy=100  \
  --max_save=20   \
  --curr_step=0   \
  --iterations=10000 \
  --train_steps=200000 \
  $@
