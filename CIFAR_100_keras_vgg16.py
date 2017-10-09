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

#------unpickle data------
def unpickle(filename):
  file = os.path.join(data_dir, filename)
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


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

#--- fine classes ---
num_classes  = 100

batch_size   = 128
epochs       = 80

#--- file paths ---
log_filepath = './tb_log_vgg16/'
weights_store_filepath = './vgg16_weights/'
retrain_id = '101'
model_name = 'weights_vgg16_cifar_100_'+retrain_id+'.h5'
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

y_test = np.zeros((test_size, num_classes)).astype('float32')

y_train[np.arange(train_size), train[b'fine_labels']] = 1

y_test[np.arange(test_size), test[b'fine_labels']] = 1

del(train)
del(test)
#---------------------------

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

#----------------------- model definition ---------------------------
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

#--- block 3 ---
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

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

model = Model(img_input, fine_pred, name='vgg16')
model.load_weights(weights_path, by_name=True)

#----------------------- compile and fit ---------------------------
sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              # optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
model.save(model_path)
print('score is: ', score)