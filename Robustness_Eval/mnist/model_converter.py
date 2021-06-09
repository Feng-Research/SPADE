import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from model import Model

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam

from tensorflow.python import pywrap_tensorflow

# read checkpoint
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

#   set path of model in and out
  model_name = 'mnist-0.3-selftrained'
  saver = tf.train.import_meta_graph('models/mnist-0.3/checkpoint-19800.meta')
  saver.restore(sess, 'models/mnist-0.3/checkpoint-19800')

  vars_global = tf.global_variables()
  model_vars = {}
  var_shapes = []
  var_names = []

  for var in vars_global:
    # try:
    model_vars[var.name] = var.eval()
    var_shapes.append((model_vars[var.name].shape))
    var_names.append(var.name)
    # except:
    #     print("For var={}, an exception occurred".format(var.name))

labels = tf.placeholder(tf.int64, shape=[None], name='labels')
logits = tf.placeholder(tf.float32, shape=[None, 10], name='logits')

# create keras model according to the model structure
model = Sequential()
conv1 = Conv2D(32, (5,5), input_shape=(28,28,1), activation='relu', padding='SAME', 
use_bias=True
)
model.add(conv1)
model.add(MaxPooling2D(pool_size=(2,2),
                            padding='SAME'))
conv2 = Conv2D(64, (5,5), activation='relu', padding='SAME', 
use_bias=True
)                            
model.add(conv2)
model.add(MaxPooling2D(pool_size=(2,2),
                            padding='SAME'))
model.add(Flatten())

fc1 = Dense(1024, activation='relu', use_bias=True)
model.add(fc1)
fc2 = Dense(10, activation='softmax', use_bias=True)
model.add(fc2)

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                logits=predicted)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=fn, metrics=['accuracy'])

# match layers of two models
model.layers[0].set_weights([model_vars['Variable:0'], model_vars['Variable_1:0']])
model.layers[2].set_weights([model_vars['Variable_2:0'], model_vars['Variable_3:0']])
model.layers[5].set_weights([model_vars['Variable_4:0'], model_vars['Variable_5:0']])
model.layers[6].set_weights([model_vars['Variable_6:0'], model_vars['Variable_7:0']])


model.save('models/'+model_name+'.h5', overwrite=True)

# validate the accuracy of models

dataset = keras.datasets.mnist.load_data()

# mnist
test_data = dataset[1][0]
test_labels = dataset[1][1]

extra_bias = 0

batch_size = 100
op_data = np.zeros((1,10))
for i in range(int(test_data.shape[0]/batch_size)):
        # x = test_data[i*batch_size:(i+1)*batch_size]
        x = test_data[i*batch_size:(i+1)*batch_size]/255 - extra_bias
        x = np.reshape(x, (batch_size, test_data.shape[1], test_data.shape[2], 1))
        # op = sess.run(model.output, feed_dict = {model.input: x})
        op = model.predict(x, batch_size=batch_size)
        op_data = np.vstack((op_data, op))
op_data = op_data[1:]
print('accuracy is:')
print(np.sum(np.argmax(op_data, axis=1) == test_labels)/op_data.shape[0])

