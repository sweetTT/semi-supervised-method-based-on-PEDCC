# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import json
from absl import flags
from autoaugment import custom_ops as ops
import data
import utils
from autoaugment.wrn import build_wrn_model
from utils_pedcc import *
from mmd_tf import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# TPU related
flags.DEFINE_string(
    "master", default=None,
    help="the TPU address. This should be set when using Cloud TPU")
flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, "
    "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool(
    "use_tpu", default=False,
    help="Use TPUs rather than GPU/CPU.")
flags.DEFINE_enum(
    "task_name", "cifar10",
    enum_values=["cifar10", "svhn"],
    help="The task to use")

# UDA config:
flags.DEFINE_integer(
    "sup_size", default=4000,
    help="Number of supervised pairs to use. "
    "-1: all training samples. 4000: 4000 supervised examples.")
flags.DEFINE_integer(
    "aug_copy", default=0,
    help="Number of different augmented data generated.")
flags.DEFINE_integer(
    "unsup_ratio", default=0,
    help="The ratio between batch size of unlabeled data and labeled data, "
    "i.e., unsup_ratio * train_batch_size is the batch_size for unlabeled data."
    "Do not use the unsupervised objective if set to 0.")
flags.DEFINE_enum(
    "tsa", "",
    enum_values=["", "linear_schedule", "log_schedule", "exp_schedule"],
    help="anneal schedule of training signal annealing. "
    "tsa='' means not using TSA. See the paper for other schedules.")
flags.DEFINE_float(
    "uda_confidence_thresh", default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")
flags.DEFINE_float(
    "uda_softmax_temp", -1,
    help="The temperature of the Softmax when making prediction on unlabeled"
    "examples. -1 means to use normal Softmax")
flags.DEFINE_float(
    "ent_min_coeff", default=0,
    help="")
flags.DEFINE_integer(
    "unsup_coeff", default=1,
    help="The coefficient on the UDA loss. "
    "setting unsup_coeff to 1 works for most settings. "
    "When you have extermely few samples, consider increasing unsup_coeff")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string(
    "data_dir", default=None,
    help="Path to data directory containing `*.tfrecords`.")
flags.DEFINE_string(
    "model_dir", default=None,
    help="model dir of the saved checkpoints.")
flags.DEFINE_bool(
    "do_train", default=True,
    help="Whether to run training.")
flags.DEFINE_bool(
    "do_eval", default=False,
    help="Whether to run eval on the test set.")
flags.DEFINE_bool(
    "do_eval_along_training", default=True,
    help="Whether to run eval on the test set during training. "
    "This is only used to debug.")
flags.DEFINE_bool(
    "verbose", default=False,
    help="Whether to print additional information.")

# Training config
flags.DEFINE_integer(
    "train_batch_size", default=32,
    help="Size of train batch.")
flags.DEFINE_integer(
    "eval_batch_size", default=32,
    help="Size of evalation batch.")
flags.DEFINE_integer(
    "train_steps", default=100000,
    help="Total number of training steps.")
flags.DEFINE_integer(
    "curr_step", default=0,
    help="number of re-training steps.")
flags.DEFINE_integer(
    "iterations", default=10000,
    help="Number of iterations per repeat loop.")
flags.DEFINE_integer(
    "save_steps", default=10000,
    help="number of steps for model checkpointing.")
flags.DEFINE_integer(
    "max_save", default=30,
    help="Maximum number of checkpoints to save.")

# Model config
flags.DEFINE_enum(
    "model_name", default="wrn",
    enum_values=["wrn",
                 ],
    help="Name of the model")
flags.DEFINE_integer(
    "num_classes", default=10,
    help="Number of categories for classification.")
flags.DEFINE_integer("feature_dim",default=128,help="feature dimension of PEDCC")
flags.DEFINE_integer(
    "wrn_size", default=32,
    help="The size of WideResNet. It should be set to 32 for WRN-28-2"
    "and should be set to 160 for WRN-28-10")

# Optimization config
flags.DEFINE_float(
    "learning_rate", default=0.03,
    help="Maximum learning rate.")
flags.DEFINE_float(
    "weight_decay_rate", default=5e-4,
    help="Weight decay rate.")
flags.DEFINE_float(
    "min_lr_ratio", default=0.004,
    help="Minimum ratio learning rate.")
flags.DEFINE_integer(
    "warmup_steps", default=20000,
    help="Number of steps for linear lr warmup.")



FLAGS = tf.flags.FLAGS
arg_scope = tf.contrib.framework.arg_scope

def setup_arg_scopes(is_training):
  """Sets up the argscopes that will be used when building an image model.

  Args:
    is_training: Is the model training or not.

  Returns:
    Arg scopes to be put around the model being constructed.
  """

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "scale": True,
      # collection containing the moving mean and moving variance.
      "is_training": is_training,
  }

  scopes = []

  scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
  return scopes


def build_model(inputs, num_classes, feature_dim,is_training, update_bn, hparams,):
  """Constructs the vision model being trained/evaled.

  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.

  Returns:
    The logits of the image model.
  """
  scopes = setup_arg_scopes(is_training)
  with contextlib.nested(*scopes):
    if hparams.model_name == "pyramid_net":
      logits = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == "wrn":
      logits= build_wrn_model(
          inputs, num_classes, feature_dim,hparams.wrn_size, update_bn)


    elif hparams.model_name == "shake_shake":
      logits = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)
  return logits



def _kl_divergence_with_logits(p_logits, q_logits):
  p = tf.nn.softmax(p_logits)
  log_p = tf.nn.log_softmax(p_logits)
  log_q = tf.nn.log_softmax(q_logits)

  kl = tf.reduce_sum(p * (log_p - log_q), -1)
  return kl


def get_model_fn(hparams):
  def model_fn(features, labels, mode, params):
    sup_labels = tf.reshape(features["label"], [-1])

    #### Configuring the optimizer
    global_step = tf.train.get_global_step()
    metric_dict = {}
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if FLAGS.unsup_ratio > 0 and is_training:
      all_images = tf.concat([features["image"],
                              features["ori_image"],
                              features["aug_image"]], 0)
    else:
      all_images = features["image"]

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      all_logits = build_model(
          inputs=all_images,
          num_classes=FLAGS.num_classes,
          feature_dim =128,
          is_training=is_training,
          update_bn=True and is_training,
          hparams=hparams,
      )
      sup_bsz = tf.shape(features["image"])[0]

      sup_logits = all_logits[0][:sup_bsz]
      print('sup_buz')
      print(sup_bsz)
      sup_features = all_logits[1][:sup_bsz]


      map_dict = read_pkl()
      tmp_list = [x.numpy() for x in map_dict.values()]
      pedcc_features_all = np.concatenate(tmp_list)

      def f0():
          return tmp_list[0]
      def f1():
          return tmp_list[1]
      def f2():
          return tmp_list[2]
      def f3():
          return tmp_list[3]
      def f4():
          return tmp_list[4]
      def f5():
          return tmp_list[5]
      def f6():
          return tmp_list[6]
      def f7():
          return tmp_list[7]
      def f8():
          return tmp_list[8]
      def f9():
          return tmp_list[9]
      def f10():
          pass

      for i in range(FLAGS.train_batch_size):
          tmp = sup_labels[i]
          test = tf.case({
              tf.equal(tmp,0):  f0,
              tf.equal(tmp, 1): f1,
              tf.equal(tmp, 2): f2,
              tf.equal(tmp, 3): f3,
              tf.equal(tmp, 4): f4,
              tf.equal(tmp, 5): f5,
              tf.equal(tmp, 6): f6,
              tf.equal(tmp, 7): f7,
              tf.equal(tmp, 8): f8,
              tf.equal(tmp, 9): f9
          },exclusive=True)
          if i==0:
              feature_label=test
          else:
              feature_label=tf.concat([feature_label,test], axis=0)

      pedcc_features = tf.cast(feature_label, dtype=tf.float32)

      mse_loss = tf.reduce_mean(tf.square(sup_features- pedcc_features))
      loss_2 = AM_loss (sup_logits,sup_labels)
      sup_loss = mse_loss + loss_2
      sup_prob = tf.nn.softmax(sup_logits, axis=-1)
      metric_dict["sup/pred_prob"] = tf.reduce_mean(
          tf.reduce_max(sup_prob, axis=-1))

    avg_sup_loss = tf.reduce_mean(sup_loss)
    total_loss = avg_sup_loss

    if FLAGS.unsup_ratio > 0 and is_training:
      aug_bsz = tf.shape(features["ori_image"])[0]

      ori_logits = all_logits[0][sup_bsz : sup_bsz + aug_bsz]
      ori_features = all_logits[1][sup_bsz: sup_bsz + aug_bsz]
      aug_logits = all_logits[0][sup_bsz + aug_bsz:]

      ori_logits_tgt = ori_logits
      ori_prob = tf.nn.softmax(ori_logits, axis=-1)
      aug_prob = tf.nn.softmax(aug_logits, axis=-1)
      metric_dict["unsup/ori_prob"] = tf.reduce_mean(
          tf.reduce_max(ori_prob, axis=-1))
      metric_dict["unsup/aug_prob"] = tf.reduce_mean(
          tf.reduce_max(aug_prob, axis=-1))

      for i in range(0,int(FLAGS.train_batch_size*FLAGS.unsup_ratio/10-1)):  ##
          # print(i)
          if i==0:
              pedcc_features_sum = tf.concat([pedcc_features_all, pedcc_features_all], axis=0)
          else:
              pedcc_features_sum = tf.concat([pedcc_features_sum,pedcc_features_all], axis=0)
      pedcc_features_sum = tf.cast(pedcc_features_sum, dtype=tf.float32)


      mmd_loss = mmd_rbf(ori_features,pedcc_features_sum)
      mmd_loss = mmd_loss * 0.04
      aug_loss = _kl_divergence_with_logits(
          p_logits=tf.stop_gradient(ori_logits_tgt),
          q_logits=aug_logits)

      avg_unsup_loss = tf.reduce_mean(aug_loss)
      avg_unsup_loss = avg_unsup_loss*1600
      total_loss += FLAGS.unsup_coeff * avg_unsup_loss
      total_loss += mmd_loss
      metric_dict["unsup/mmd_loss"] = mmd_loss
      metric_dict["unsup/loss"] = avg_unsup_loss

    total_loss = utils.decay_weights(
        total_loss,
        FLAGS.weight_decay_rate)



    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))


    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      #### Metric function for classification
      def metric_fn(per_example_loss, label_ids, logits):
        # classification loss & accuracy
        loss = tf.metrics.mean(per_example_loss)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)

        ret_dict = {
            "eval/classify_loss": loss,
            "eval/classify_accuracy": accuracy
        }

        return ret_dict

      eval_metrics = (metric_fn, [sup_loss, sup_labels, sup_logits])

      #### Constucting evaluation TPUEstimatorSpec.
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics)

      return eval_spec

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
      warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                  * FLAGS.learning_rate
    else:
      warmup_lr = 0.0

    # decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step-FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)


    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9,
        use_nesterov=True)

    #### use_tpu =false  ###
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(
          zip(gradients, variables), global_step=tf.train.get_global_step())

    #### Creating training logging hook
    # compute accuracy
    sup_pred = tf.argmax(sup_logits, axis=-1, output_type=sup_labels.dtype)
    is_correct = tf.to_float(tf.equal(sup_pred, sup_labels))
    acc = tf.reduce_mean(is_correct)
    metric_dict["sup/sup_loss"] = avg_sup_loss
    metric_dict["training/loss"] = total_loss
    metric_dict["sup/acc"] = acc
    metric_dict["training/lr"] = learning_rate
    metric_dict["training/step"] = global_step


    if not FLAGS.use_tpu:
      log_info = ("step [{training/step}] lr {training/lr:.6f} "
                  "loss {training/loss:.4f} "
                  "sup/acc {sup/acc:.4f} sup/loss {sup/sup_loss:.6f} ")
      if FLAGS.unsup_ratio > 0:
        log_info += "unsup/loss {unsup/loss:.6f} "
        log_info += "unsup/mmd_loss {unsup/mmd_loss:.6f} "
      formatter = lambda kwargs: log_info.format(**kwargs)
      logging_hook = tf.train.LoggingTensorHook(
          tensors=metric_dict,
          every_n_iter=FLAGS.iterations,
          formatter=formatter)
      training_hooks = [logging_hook]
      #### Constucting training TPUEstimatorSpec.
      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op,
          training_hooks=training_hooks)
    else:
      #### Constucting training TPUEstimatorSpec.
      host_call = utils.construct_scalar_host_call(
          metric_dict=metric_dict,
          model_dir=params["model_dir"],
          prefix="",
          reduce_fn=tf.reduce_mean)
      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op,
          host_call=host_call)

    return train_spec

  return model_fn


def train(hparams):
  ##### Create input function

  ######read data#######
  if FLAGS.do_train:
    train_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="train",
        task_name=FLAGS.task_name,
        sup_size=FLAGS.sup_size,
        unsup_ratio=FLAGS.unsup_ratio,
        aug_copy=FLAGS.aug_copy,
    )

  if FLAGS.do_eval:
    eval_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="test",
        task_name=FLAGS.task_name,
        sup_size=-1,
        unsup_ratio=0,
        aug_copy=0)
    if FLAGS.task_name == "cifar10":
      eval_size = 10000
    elif FLAGS.task_name == "svhn":
      eval_size = 26032
    else:
      raise ValueError, "You need to specify the size of your test set."
    eval_steps = eval_size // FLAGS.eval_batch_size

  ##### Get model function
  model_fn = get_model_fn(hparams)
  estimator = utils.get_TPU_estimator(FLAGS, model_fn)

  #### Training
  if FLAGS.do_eval_along_training:
    tf.logging.info("***** Running training & evaluation *****")
    tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Unsupervised batch size = %d",
                    FLAGS.train_batch_size * FLAGS.unsup_ratio)
    tf.logging.info("  Num train steps = %d", FLAGS.train_steps)

    while True:
      if FLAGS.curr_step >= FLAGS.train_steps:
        break
      tf.logging.info("Current step {}".format(FLAGS.curr_step))
      train_step = min(FLAGS.save_steps, FLAGS.train_steps - FLAGS.curr_step)
      estimator.train(input_fn=train_input_fn, steps=train_step)
      estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      FLAGS.curr_step += FLAGS.save_steps
  else:
    if FLAGS.do_train:
      tf.logging.info("***** Running training *****")
      tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
      tf.logging.info("  Unsupervised batch size = %d",
                      FLAGS.train_batch_size * FLAGS.unsup_ratio)
      estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    if FLAGS.do_eval:
      tf.logging.info("***** Running evaluation *****")
      results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      tf.logging.info(">> Results:")
      for key in results.keys():
        tf.logging.info("  %s = %s", key, str(results[key]))
        results[key] = results[key].item()
      acc = results["eval/classify_accuracy"]
      with tf.gfile.Open("{}/results.txt".format(FLAGS.model_dir), "w") as ouf:
        ouf.write(str(acc))


def main(_):

  if FLAGS.do_train:
    tf.gfile.MakeDirs(FLAGS.model_dir)
    flags_dict = tf.app.flags.FLAGS.flag_values_dict()
    with tf.gfile.Open(os.path.join(FLAGS.model_dir, "FLAGS.json"), "w") as ouf:
      json.dump(flags_dict, ouf)
  hparams = tf.contrib.training.HParams()

  if FLAGS.model_name == "wrn":
    hparams.add_hparam("model_name", "wrn")
    hparams.add_hparam("wrn_size", FLAGS.wrn_size)
  else:
    raise ValueError("Not Valid Model Name: %s" % FLAGS.model_name)

  train(hparams)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
