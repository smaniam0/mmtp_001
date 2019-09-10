import tensorflow as tf
### Set this Or Else reinstall CUDA from scratc ###
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth=True
ess = tf.Session(config=cfg)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models as Models


from trainutils import DataSet, unpickleData, pickleData
import numpy as np

def save_stats(prediction, fname):
    predidx = np.argmax(prediction, 1)
    actidx = np.argmax(test_set.labels, 1)
    mismatch = np.nonzero(predidx != actidx)
    missed = [mismatch, actidx[mismatch], prediction[mismatch]]
    pickleData(missed, fname)

import sys
if len(sys.argv) != 2:
    print('Usage: ', sys.argv[0], ' model-name')
    sys.exit(1)

TST_DATA='../../data/tst_dataset.pkl.gz'

print('Loading Data...')
test_set = unpickleData(TST_DATA)[0]

## Change Dim for Conv nets: Basically add channel  ##
test_set.data=np.expand_dims(test_set.data, axis=3)


print('Loading Model...')
model = Models.load_model(sys.argv[1])

print('Generating Raw statistics...')

prediction = model.predict(test_set.data)
basefname = sys.argv[1].rsplit('.', 1)[0]
fname = basefname + '.mis'
save_stats(prediction, fname)

