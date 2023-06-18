
from __future__ import print_function
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import importlib
import sys

from tensorflow import keras

from SE import RUN_SE

importlib.reload(sys)
import numpy as np
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.pyplot import savefig
import random
# random.seed(2345)
import pandas as pd
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config=tf.compat.v1.ConfigProto()
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.83)
config.gpu_options.allow_growth=True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import concatenate
from keras import regularizers
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from ucr_reading import *

dir_path = 'Data'
list_dir = os.listdir(dir_path)
index1 = list_dir.index('Processure1')
index2 = list_dir.index('Processure2')
index3 = list_dir.index('Processure3')

number_res=32
IS=1
SR=0.9
SP=0.7
nb_epoch =30
batch_size=25
nb_filter=120
ratio=[0.5,0.7]

print('Loading data...')
train_echoes1, train_y1, test_echoes1, test_y1, dataset_name1, n_res, len_series, IS, SR, SP = run_loading(index=index1, n_res =number_res,IS=IS,SR=SR,SP=SP)
train_echoes2, train_y2, test_echoes2, test_y2, dataset_name2, n_res, len_series, IS, SR, SP = run_loading(index=index2, n_res =number_res,IS=IS,SR=SR,SP=SP)
train_echoes3, train_y3, test_echoes3, test_y3, dataset_name3, n_res, len_series, IS, SR, SP = run_loading(index=index3, n_res =number_res,IS=IS,SR=SR,SP=SP)

print('Transfering label...')
train_y1, nums_class = transfer_labels(train_y1)
train_y2, nums_class = transfer_labels(train_y2)
train_y3, nums_class = transfer_labels(train_y3)

test_y1, _ = transfer_labels(test_y1)
test_y2, _ = transfer_labels(test_y2)
test_y3, _ = transfer_labels(test_y3)
#这是在进行独热编码

train_y1 = np_utils.to_categorical(train_y1, nums_class)
train_y2 = np_utils.to_categorical(train_y2, nums_class)
train_y3 = np_utils.to_categorical(train_y3, nums_class)

test_y1 = np_utils.to_categorical(test_y1, nums_class)
test_y2 = np_utils.to_categorical(test_y2, nums_class)
test_y3 = np_utils.to_categorical(test_y3, nums_class)
nb_class = nums_class

nb_sample_train1 = np.shape(train_echoes1)[0]
nb_sample_train2 = np.shape(train_echoes2)[0]
nb_sample_train3 = np.shape(train_echoes3)[0]
nb_sample_test1 = np.shape(test_echoes1)[0]
nb_sample_test2 = np.shape(test_echoes2)[0]
nb_sample_test3 = np.shape(test_echoes3)[0]


test_data1 = np.reshape(test_echoes1,(nb_sample_test1, 1, len_series, n_res))
test_data2 = np.reshape(test_echoes2,(nb_sample_test2, 1, len_series, n_res))
test_data3 = np.reshape(test_echoes3,(nb_sample_test3, 1, len_series, n_res))

test_labels1 = test_y1
test_labels2 = test_y2
test_labels3 = test_y3

L_train1 = [x_train1 for x_train1 in range(nb_sample_train1)]
np.random.shuffle(L_train1)

train_data1 = np.zeros((nb_sample_train1, 1, len_series, n_res))
train_data2 = np.zeros((nb_sample_train2, 1, len_series, n_res))
train_data3 = np.zeros((nb_sample_train3, 1, len_series, n_res))

train_label1 = np.zeros((nb_sample_train1, nb_class))
train_label2 = np.zeros((nb_sample_train2, nb_class))
train_label3 = np.zeros((nb_sample_train3, nb_class))


for m in range(nb_sample_train1):

    train_data1[m,0,:,:] = train_echoes1[L_train1[m],:,:]
    train_label1[m,:] = train_y1[L_train1[m],:]
    train_data2[m,0,:,:] = train_echoes2[L_train1[m],:,:]
    train_label2[m,:] = train_y2[L_train1[m],:]
    train_data3[m, 0, :, :] = train_echoes3[L_train1[m], :, :]
    train_label3[m, :] = train_y3[L_train1[m], :]

Data_train=np.array([train_data1,train_data2,train_data3])
# print(Data_train.shape())

Data_test=np.array([test_data1,test_data2,test_data3])
Data_train=np.reshape(Data_train,(nb_sample_train1,3,1,len_series, n_res))
Data_test=np.reshape(Data_test,(nb_sample_test1,3,1,len_series, n_res))

i=0
while i<nb_sample_train1:
    Data_train[i]=RUN_SE(Data_train[i])
    i+=1
j=0
while j<nb_sample_test1:
    Data_test[i]=RUN_SE(Data_test[i])
    j+=1













# input_shape = (1, len_series, n_res)
input_shape = (1, len_series, n_res)
nb_row=[np.int(ratio[0]*len_series),np.int(ratio[1]*len_series)]
nb_col = input_shape[2]
#kernel_initializer权重初始化方法，LeCun 均匀初始化器。它从[−limit,limit]中的均匀分布中抽取样本，
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format='channels_last'
optimizer = 'adam'
loss = ['binary_crossentropy', 'categorical_crossentropy']
verbose = 1
#model


# train_data,test_data=RUN_SE(train_data,test_data)




print(input_shape[0])


input = Input(shape = input_shape)
input1 = Input(shape = input_shape)
input2 = Input(shape = input_shape)
input3 = Input(shape = input_shape)

convs1 = []
for j in range(len(nb_row)):
    #strides步长是（1,1），卷积核个数是120
	conv1 = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = 'same', strides = strides, data_format = data_format)(input1)
	conv1 = GlobalMaxPooling2D(data_format = data_format)(conv1)
	convs1.append(conv1)

body_feature1 = concatenate(convs1,name='concat_layer1')

convs2 = []
for j in range(len(nb_row)):
    #strides步长是（1,1），卷积核个数是120
	conv2 = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = 'same', strides = strides, data_format = data_format)(input2)
	conv2 = GlobalMaxPooling2D(data_format = data_format)(conv2)
	convs2.append(conv2)

body_feature2 = concatenate(convs2,name='concat_layer2')

convs3 = []
for j in range(len(nb_row)):
    #strides步长是（1,1），卷积核个数是120
	conv3 = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = 'same', strides = strides, data_format = data_format)(input3)
	conv3 = GlobalMaxPooling2D(data_format = data_format)(conv3)
	convs3.append(conv3)

body_feature3 = concatenate(convs3,name='concat_layer3')
body_feature = keras.layers.Concatenate()([body_feature1,body_feature2,body_feature3])
#body_feature = Dense(64, kernel_initializer = kernel_initializer, activation = activation)(body_feature)
#body_feature = Dense(128, kernel_initializer = kernel_initializer, activation = activation)(body_feature)
body_feature = Dropout(0.25)(body_feature)
output = Dense(nb_class, kernel_initializer = kernel_initializer, activation = 'softmax',name = 'dense_output')(body_feature)



model = Model(input1,input2,input3, output)
# model.summary()
callbacks = [
        keras.callbacks.ModelCheckpoint(
            "Model_save/EMN.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])
history = model.fit(Data_train, train_label1, batch_size = batch_size,
	epochs = nb_epoch, callbacks=callbacks,verbose = verbose,
	validation_split=0.2)
# print(history.history.keys())
log = pd.DataFrame(history.history)
minloss_acc=log.loc[log['loss'].idxmin]['val_accuracy']

plt.figure(figsize=(9,3))
plt.plot(history.history['accuracy'],linewidth=0.5)
plt.plot(history.history['val_accuracy'],linewidth=0.5)

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.figure(figsize=(9,3))
plt.plot(history.history['loss'],linewidth=0.5)
plt.plot(history.history['val_loss'],linewidth=0.5)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print('batch:',batch_size)
print('filter:,',nb_filter)
print('nb_row:',ratio)
print('IS :', IS)
print('SR:', SR)
print('SP:', SP)
print('Size:', number_res)
print('accuracy:',minloss_acc)
test_loss, test_acc = model.evaluate(test_data1,test_data2,test_data3, test_labels1)
print("test_loss",test_loss)
print("test_acc",test_acc)


model2=keras.models.load_model("Model_save/EMN.h5")
train_loss, train_acc = model2.evaluate(test_data1,test_data2,test_data3, test_labels1)
test_loss, test_acc = model2.evaluate(test_data1,test_data2,test_data3, test_labels1)
print("train_loss",train_loss)
print("train_acc",train_acc)
print("test_loss",test_loss)
print("test_acc",test_acc)


Y_predictions = model2.predict(test_data1,test_data2,test_data3, test_labels1)
i = 0
Y_label = list()
Y_test_label = list()
count=0
while i < len(test_labels1):
    label = list(Y_predictions[i]).index(max(Y_predictions[i]))
    print("预测：",label)
    Y_label.append(label)
    print(test_labels1[i])
    print("真实：",np.argmax(test_labels1[i], axis=-1))
    if(label==np.argmax(test_labels1[i], axis=-1)):
        count+=1
    Y_test_label.append(np.argmax(test_labels1[i], axis=-1))
    i += 1
print(Y_label)

print("ACCURACY:",count/len(Y_label))

confusion = tf.compat.v1.confusion_matrix(labels=Y_test_label, predictions=Y_label, num_classes=5)
print(confusion)
confusion = np.array(confusion)
df_cm = pd.DataFrame(confusion)
import seaborn as sn
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, fmt='.0f', cmap="Blues", annot_kws={"size": 16})  # font size
plt.show()


