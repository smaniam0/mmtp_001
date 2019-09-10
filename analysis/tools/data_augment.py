from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
from trainutils import DataSet, plotTrainingHistory, pickleData, unpickleData

import sys
if (len(sys.argv) != 3):
    print('Usage: ', sys.argv[0], ' aug-fname src-fname')
    sys.exit(1)


DATA_FILE=sys.argv[2]
data_set  = unpickleData(DATA_FILE)[0]

datagen = ImageDataGenerator(
        width_shift_range=[-2, -1, 1, 2], fill_mode='constant', cval=0,
        height_shift_range=[-2, -1, 1, 2], shear_range=15, rotation_range=15,
        validation_split=0, horizontal_flip=False, vertical_flip=False,
        #zca_whitening=True, samplewise_center=True, samplewise_std_normalization=True,
        zoom_range=0.15)

X = np.expand_dims(data_set.data, axis=3)
datagen.fit(X, augment=True, rounds=1)
batchsz = 256

n = 0
for x_batch, y_batch in datagen.flow(X, data_set.labels, batch_size=batchsz):
    #print(y_batch[0])
    #cv.imshow('img', x_batch[0].reshape(48, 48))
    #cv.waitKey(0)
    x_batch = x_batch.reshape(x_batch.shape[0], 
            data_set.data.shape[1], data_set.data.shape[2])
    x_batch = np.float16(x_batch)
    if n == 0:
        aug_data = x_batch
        aug_labels = y_batch
    else:
        aug_data = np.append(aug_data, x_batch, axis=0)
        aug_labels = np.append(aug_labels, y_batch, axis=0)
    n = n + 1
    if n > data_set.data.shape[0]/batchsz:
        break

aug_set = DataSet()
aug_set.data = aug_data
aug_set.labels = aug_labels
pickleData(aug_set, sys.argv[1])


