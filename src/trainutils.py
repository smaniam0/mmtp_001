# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 00:49:59 2019

@author: SS
"""

import sys
import os
import gzip
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

#from tensorflow.keras.utils import to_categorical

class DataSet:
    def __init__(self, fname=None, labelcnt=None, flatten=True):
        self.fname = fname
        self.labelcnt = labelcnt
        if fname is None:
            self.data = None
            self.labels = None
            return
        if labelcnt == None:
            raise ValueError('lablecnt has to be specified')
        if not os.path.isfile(fname):
            raise FileNotFoundError('Invalid Dataset file: '+ fname)
        
        (self.data, self.labels) = self.loadData(fname, labelcnt)
        if flatten:
            inputsz = self.data.shape[1]*self.data.shape[2]  ##  (w=64, h=48)
            cnt = self.data.shape[0]
            self.data = self.data.reshape(cnt, inputsz)

    def loadData(self, fname, label_vector_size=0):
        f = gzip.open(fname)
        u = pickle.Unpickler(f)

        data = []
        label = []
        try:
            while True:
                rec = u.load()
                data.append(rec[0])
                if label_vector_size > 0:
                    lbl_vector = np.zeros(label_vector_size)
                    lbl_vector[rec[1]] = 1
                    label.append(lbl_vector)
                else:
                    label.append(rec[1])
        except EOFError:
            pass
        return [np.array(data, dtype=rec[0].dtype), np.array(label)]


class TrainingDataUtil:
    PKL_EXTN='.pkl.gz'
    def __init__(self, traindata_extn='_train', testdata_extn='_test',
                 model_name=None, label_cnt=156):
        if model_name is None:
            if (len(sys.argv) != 2):
                print('Usage: ', sys.argv[0], ' model-name')
                raise TypeError('Missing argument: model_name')
            model_name=sys.argv[1]
        self.model_name = model_name
        self.train_data_file = self.model_name + traindata_extn + self.PKL_EXTN
        self.test_data_file = self.model_name + testdata_extn + self.PKL_EXTN

        self.training_dataset = DataSet(fname=self.train_data_file, labelcnt=label_cnt)
        self.validation_dataset = DataSet(fname=self.test_data_file, labelcnt=label_cnt)
        
    
    def pickleData(self, data, fname=None):
        if fname is None:
            fname = self.model_name + '_history.pkl.gz'
        f = gzip.open(fname, 'wb')
        p = pickle.Pickler(f)
        p.dump(data)
        f.close()

def pickleData(data, fname, unbunch=False):
    f = gzip.open(fname, 'wb')
    p = pickle.Pickler(f)
    if unbunch:
        for item in data:
            p.dump(item)
    else:
        p.dump(data)
    f.close()

 
def unpickleData(fname):
    f = gzip.open(fname)
    u = pickle.Unpickler(f)
    data = []
    try:
        while True:
            rec = u.load()
            data.append(rec)
    except EOFError:
        pass
    f.close()
    return data


def plotTrainingHistory(history, fname=None):
    epochs = len(history["acc"])
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history["loss"], label="train_loss")
    plt.plot(N, history["val_loss"], label="val_loss")
    plt.plot(N, history["acc"], label="train_acc")
    plt.plot(N, history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    if fname is not None:
        plt.savefig(fname)
