import logging
import mxnet as mx
import numpy as np
from symbol import get_shufflenet

logging.getLogger().setLevel(logging.INFO)

mnist = mx.test_utils.get_mnist()

print mnist['train_data'].shape


batch_size = 500

train_data = np.concatenate((mnist['train_data'], mnist['train_data'], mnist['train_data']), 
	                        axis=1)
val_data = np.concatenate((mnist['test_data'], mnist['test_data'], mnist['test_data']), 
	                       axis=1)

train_iter = mx.io.NDArrayIter(train_data, mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, mnist['test_label'], batch_size)

shufflenet = get_shufflenet()

shufflenet_mod = mx.mod.Module(symbol=shufflenet, context=[mx.gpu(0), mx.gpu(1)])

shufflenet_mod.fit(train_iter, 
              eval_data=val_iter, 
              optimizer='sgd',  
              optimizer_params={'learning_rate':0.01},  
              eval_metric='acc',  
              batch_end_callback = mx.callback.Speedometer(batch_size, 20), 
              num_epoch=10) 


