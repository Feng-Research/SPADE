from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pgd_attack import LinfPGDAttack

from tensorflow.python import pywrap_tensorflow

from model import Model
# from pgd_attack import LinfPGDAttack

print("acc"+"b")

with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model = Model()
eps = config['epsilon']
model_dir = config['model_dir']

# ============= set output file path ==============
path_folder = "./mnist_data_"+str(eps)+"_dl/";

saver = tf.train.Saver()

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
use_set = mnist.train

if eval_on_cpu:
  with tf.device("/cpu:0"):
    attack = LinfPGDAttack(model, 
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])


global_step = tf.contrib.framework.get_or_create_global_step()



with tf.Session() as sess:
    print(sess.list_devices())  

    # model_dir = 'models/mnist-'+str(eps)
    model_dir = config['model_dir']

    # for eps in np.arange(0.4, 0.6, 0.1):
    cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    # cur_checkpoint = 'models/mnist-0.3-dl/checkpoint-99900'
    print('restoring '+ cur_checkpoint)
    saver = tf.train.Saver()
    saver.restore(sess, cur_checkpoint)


    num_batches = int(math.ceil(use_set.labels.shape[0] / eval_batch_size))
    if not os.path.exists(path_folder):
      os.mkdir(path_folder)

    batch_index = 0
    op_data = np.zeros((1,10))
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, use_set.labels.shape[0])

      x_batch = use_set.images[bstart:bend, :]
      y_batch = use_set.labels[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}
      # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      # dict_adv = {model.x_input: x_batch_adv,
      #             model.y_input: y_batch}

      # h_conv1, h_conv2, h_fc1, pred_softmax = sess.run([model.h_conv1, model.h_conv2, model.h_fc1, model.pre_softmax],
                                      # feed_dict = dict_nat)
      h_fc1, y_xent, xent, pred_softmax = sess.run([model.h_fc1, model.y_xent, model.xent, model.pre_softmax],
                                      feed_dict = dict_nat)
      
      # shp_conv1 = h_conv1.shape
      # shp_conv2 = h_conv2.shape
      shp_fc1 = h_fc1.shape

      print(y_xent.shape)
      print(xent.shape)
      
      op_data = np.vstack((op_data, pred_softmax))

      # pred_softmax = sess.run(model.pre_softmax,
      #                                 feed_dict = dict_nat)

      # print('conv1')
      # print(np.reshape(h_conv1, (eval_batch_size, shp_conv1[1]*shp_conv1[2]*shp_conv1[3])).shape)
      
      
      # fp_out = open("I:/process/mltest/mnist_data_0.3/gbt_output_conv1.csv","ab");
      # np.savetxt(fp_out, np.reshape(h_conv1, (eval_batch_size, shp_conv1[1]*shp_conv1[2]*shp_conv1[3])))
      # fp_out.close()

      # fp_out = open("I:/process/mltest/mnist_data_0.3/gbt_output_conv2.csv","ab");
      # np.savetxt(fp_out, np.reshape(h_conv2, (eval_batch_size, shp_conv2[1]*shp_conv2[2]*shp_conv2[3])))
      # fp_out.close()

      # fp_out = open(path_folder+"/gbt_output_fc1.csv","ab");
      # np.savetxt(fp_out, h_fc1, fmt="%.3f"); 
      # fp_out.close(); 

      fp_out = open(path_folder+"gbt_output_out.csv","ab");
      np.savetxt(fp_out, pred_softmax, fmt="%.4f")
      fp_out.close()

      fp_in = open(path_folder+"gbt_input.csv", "ab")
      np.savetxt(fp_in, x_batch, fmt="%.4f")
      fp_in.close()

      fp_label = open(path_folder+"gbt_label.csv", "ab")
      np.savetxt(fp_label, y_batch, fmt="%d")
      fp_label.close()

      # break;
      # batch_index=batch_index+1
      # print(batch_index)
      # if batch_index>=2:
      #   break
      print(ibatch)
    # op_data = op_data[1:]
    # print(np.sum(np.argmax(op_data, axis=1) == mnist.test.labels)/mnist.test.labels.shape[0])
    # print(mnist.test.images[0])
    

    # =========for debug===========
    # ckpt = tf.train.get_checkpoint_state('models/mnist-0')
    # print(ckpt.model_checkpoint_path)
    # 'models/mnist-0\checkpint-19800'
    # reader = pywrap_tensorflow.NewCheckpointReader(cur_checkpoint)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #   print(key)

    
