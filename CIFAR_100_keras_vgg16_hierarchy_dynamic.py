#------------------------------------
# Author: Xinqi Zhu
# Please cite paper https://arxiv.org/abs/1709.09890 if you use this code
#------------------------------------
import keras
import numpy as np
import pickle
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras import backend as K

#-----data dir----
data_dir = "./data"
#-----------------

def scheduler(epoch):
  learning_rate_init = 0.001
  if epoch > 55:
    learning_rate_init = 0.0002
  if epoch > 70:
    learning_rate_init = 0.00005
  return learning_rate_init

def unpickle(filename):
  file = os.path.join(data_dir, filename)
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

class LossWeightsModifier(keras.callbacks.Callback):
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    # customize your behavior
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 13:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.8)
      K.set_value(self.gamma, 0.1)
    if epoch == 23:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.2)
      K.set_value(self.gamma, 0.7)
    if epoch == 33:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 0)
      K.set_value(self.gamma, 1)

#-------- dimensions ---------
height, width = 32, 32
channel = 3
if K.image_data_format() == 'channels_first':
    input_shape = (channel, height, width)
else:
    input_shape = (height, width, channel)
#-----------------------------

train_size = 50000
test_size = 10000

#--- coarse 1 classes ---
coarse1_classes = 8
#--- coarse 2 classes ---
coarse2_classes = 20
#--- fine classes ---
num_classes  = 100

batch_size   = 128
epochs       = 80

#--- file paths ---
log_filepath = './tb_log_vgg16_hierarchy_dynamic/'
weights_store_filepath = './vgg16_weights_hierarchy_dynamic/'
retrain_id = '101'
model_name = 'weights_vgg16_dynamic_cifar_100_'+retrain_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)

#----------get VGG16 pre-trained weights--------
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                         WEIGHTS_PATH,
                         cache_subdir='models')

#---------get data---------
meta = unpickle("meta")
test = unpickle("test")
train = unpickle("train")

#-------------------- data loading ----------------------
x_train = np.reshape(train[b'data'], (train_size, channel, height, width)).transpose(0, 2, 3, 1).astype("float32")
x_train = (x_train-np.mean(x_train)) / np.std(x_train)

x_test = np.reshape(test[b'data'], (test_size, channel, height, width)).transpose(0, 2, 3, 1).astype("float32")
x_test = (x_test-np.mean(x_test)) / np.std(x_test)

y_train = np.zeros((train_size, num_classes)).astype('float32')
y_c2_train = np.zeros((train_size, coarse2_classes)).astype('float32')

y_test = np.zeros((test_size, num_classes)).astype('float32')
y_c2_test = np.zeros((test_size, coarse2_classes)).astype('float32')

y_train[np.arange(train_size), train[b'fine_labels']] = 1
y_c2_train[np.arange(train_size), train[b'coarse_labels']] = 1

y_test[np.arange(test_size), test[b'fine_labels']] = 1
y_c2_test[np.arange(test_size), test[b'coarse_labels']] = 1

c2_to_f = np.zeros((coarse2_classes, num_classes)).astype('float32')
fine_unique, fine_unique_indices = np.unique(train[b'fine_labels'], return_index=True)
for i in fine_unique_indices:
  c2_to_f[train[b'coarse_labels'][i]][train[b'fine_labels'][i]] = 1

parent_c2 = {
  0:0, 1:0, 2:1, 3:2, 
  4:1, 5:2, 6:2, 7:3, 
  8:4, 9:5, 10:5, 11:4, 
  12:4, 13:3, 14:6, 15:4, 
  16:4, 17:1, 18:7, 19:7
}

y_c1_train = np.zeros((y_c2_train.shape[0], coarse1_classes)).astype("float32")
y_c1_test = np.zeros((y_c2_test.shape[0], coarse1_classes)).astype("float32")
for i in range(y_c1_train.shape[0]):
  y_c1_train[i][parent_c2[np.argmax(y_c2_train[i])]] = 1.0
for i in range(y_c1_test.shape[0]):
  y_c1_test[i][parent_c2[np.argmax(y_c2_test[i])]] = 1.0

del(train)
del(test)
#---------------------------

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
print("y_c1_train shape: ", y_c1_train.shape)
print("y_c1_test shape: ", y_c1_test.shape)
print("y_c2_train shape: ", y_c2_train.shape)
print("y_c2_test shape: ", y_c2_test.shape)

#----------------------- model definition ---------------------------
alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

img_input = Input(shape=input_shape, name='input')
#--- block 1 ---
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#--- block 2 ---
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

#--- coarse 1 branch ---
c_1_bch = Flatten(name='c1_flatten')(x)
c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
c_1_bch = BatchNormalization()(c_1_bch)
c_1_bch = Dropout(0.5)(c_1_bch)
c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
c_1_bch = BatchNormalization()(c_1_bch)
c_1_bch = Dropout(0.5)(c_1_bch)
c_1_pred = Dense(coarse1_classes, activation='softmax', name='c1_predictions_cifar10')(c_1_bch)

#--- block 3 ---
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#--- coarse 2 branch ---
c_2_bch = Flatten(name='c2_flatten')(x)
c_2_bch = Dense(1024, activation='relu', name='c2_fc_cifar100_1')(c_2_bch)
c_2_bch = BatchNormalization()(c_2_bch)
c_2_bch = Dropout(0.5)(c_2_bch)
c_2_bch = Dense(1024, activation='relu', name='c2_fc2')(c_2_bch)
c_2_bch = BatchNormalization()(c_2_bch)
c_2_bch = Dropout(0.5)(c_2_bch)
c_2_pred = Dense(coarse2_classes, activation='softmax', name='c2_predictions_cifar100')(c_2_bch)

#--- block 4 ---
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


#--- block 5 ---
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = BatchNormalization()(x)

#--- fine block ---
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc_cifar100_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc_cifar100_2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
fine_pred = Dense(num_classes, activation='softmax', name='predictions_cifar100')(x)

model = Model(img_input, [c_1_pred, c_2_pred, fine_pred], name='vgg16_hierarchy')
model.load_weights(weights_path, by_name=True)

#----------------------- compile and fit ---------------------------
sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              loss_weights=[alpha, beta, gamma], 
              # optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
change_lw = LossWeightsModifier(alpha, beta, gamma)
cbks = [change_lr, tb_cb, change_lw]

model.fit(x_train, [y_c1_train, y_c2_train, y_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(x_test, [y_c1_test, y_c2_test, y_test]))

#---------------------------------------------------------------------------------
# The following compile() is just a behavior to make sure this model can be saved.
# We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
#---------------------------------------------------------------------------------
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adadelta(),
              optimizer=sgd, 
              metrics=['accuracy'])

model.save(model_path)
score = model.evaluate(x_test, [y_c1_test, y_c2_test, y_test], verbose=0)
print('score is: ', score)