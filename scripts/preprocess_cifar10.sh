# coding=utf-8

task_name=cifar10

# preprocess supervised data
python preprocess.py \
  --data_type=sup \
  --sup_size=4000 \
  --task_name=${task_name} \
  --raw_data_dir=data/raw_data/${task_name} \
  --output_base_dir=data/proc_data/${task_name}

# preprocess unsupervised data
python preprocess.py \
  --data_type=unsup \
  --task_name=${task_name} \
  --raw_data_dir=data/raw_data/${task_name} \
  --output_base_dir=data/proc_data/${task_name} \
  $@

