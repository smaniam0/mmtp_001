###################################################################
##                                                               ##
##    The Final Model for Training a Tamil Isolated Handwritten  ##
##    Character Recognition system developed as part of Project  ##
##    for MSCMACS Course of IGNOU                                ##
##                                                               ##
###################################################################



import tensorflow as tf
##
###  Tensor Flow GPU settings 
##
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth=True
ess = tf.Session(config=cfg)

##
##   Import components for Building the Network
##
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD, Nadam, Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2

##
##  Import Utilities for unpacking saved training and
##  testing data sets
##
from trainutils import pickleData, unpickleData
import numpy as np


##
##  Validate Commandline  Input arguments
##

import sys
if (len(sys.argv) != 3):
    print('Usage: ', sys.argv[0], ' max-epochs learn-rate')
    sys.exit(1)

max_epochs = int(sys.argv[1])
learning_rate=float(sys.argv[2])

##
##   Build the Model
##


imgshape = (48, 48, 1)
LABELCNT=156

model = Sequential()

model.add(Input(shape=imgshape))

model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
    input_shape=imgshape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
    activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.3))
model.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
    activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
    activation='relu', padding='same'))

model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=LABELCNT, activation='softmax', use_bias=True))

optimizer = Nadam(lr=learning_rate)

model.compile(loss=categorical_crossentropy, optimizer=optimizer,
    metrics=['accuracy'])

##
##  Train the Model
##



TRG_DATA='../data/trg_dataset_augmented_twice.pkl.gz'   ## Augmented data
TST_DATA='../data/tst_dataset.pkl.gz'

batchsz = 64


train_set = unpickleData(TRG_DATA)[0]   ## Pickled data is stored as an array
test_set  = unpickleData(TST_DATA)[0]   ##    element even for one item

train_set.data=np.expand_dims(train_set.data, axis=3)
test_set.data=np.expand_dims(test_set.data, axis=3)

history = model.fit(train_set.data, train_set.labels,
                  batch_size = batchsz,
                  verbose = 1, epochs = max_epochs,
                  validation_data=(test_set.data, test_set.labels))

basefname = sys.argv[0].rsplit('.', 1)[0]
pickleData(history.history, basefname + '.hst')
model.save(basefname + '.mdl')
