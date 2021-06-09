"""
The code is adapted from the MadryLab's repo:
https://github.com/MadryLab/mnist_challenge/eval.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack


parser = argparse.ArgumentParser(description='adv_training')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--eps', type=float, default=0.4)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--restarts', type=int, default=1)
parser.add_argument('--loss', type=str, default="xent",
        help="choose loss function, [xent, cw]")
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd, clean]")


args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.device)

# Global constants
with open('config.json') as config_file:
    config = json.load(config_file)
config["epsilon"] = args.eps
config["k"] = args.k
config["loss_func"] = args.loss

if args.method=="pgd":
    config["model_dir"] = "models/pgd_0.4"
elif args.method=="clean":
    config["model_dir"] = "models/pgd_0.0"
else:
    config["model_dir"] = "models/pgd-{}_0.2_0.4".format(args.method)

num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
model_dir = config['model_dir']
eval_on_cpu = args.cpu

# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    attack = LinfPGDAttack(model,
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model()
  attack = LinfPGDAttack(model,
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]
      eps_batch = (args.eps * np.ones(bend-bstart)).reshape(-1,1)

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      best_batch_adv = np.copy(x_batch)
      dict_adv = {model.x_input: best_batch_adv, model.y_input: y_batch}
      cur_corr, best_loss = sess.run([model.num_correct, model.xent],
                                                    feed_dict=dict_adv)
      for _ in range(args.restarts):
          x_batch_adv = attack.perturb(x_batch, y_batch, sess, eps_batch)
          dict_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}
          cur_corr, this_loss = sess.run([model.num_correct, model.xent],
                                                      feed_dict=dict_adv)
          bb = best_loss >= this_loss
          bw = best_loss < this_loss
          best_batch_adv[bw, :] = x_batch_adv[bw, :]
          best_corr, best_loss = sess.run([model.num_correct, model.xent],
                                feed_dict={model.x_input: best_batch_adv, model.y_input: y_batch})

      x_batch_adv = np.copy(best_batch_adv)

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_adv)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))

cur_checkpoint = tf.train.latest_checkpoint(model_dir)
evaluate_checkpoint(cur_checkpoint)
