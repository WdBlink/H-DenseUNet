
# coding: utf-8

# In[10]:


"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
import keras
#sys.path.insert(0,'Keras-2.0.8')
from multiprocessing.dummy import Pool as ThreadPool
from medpy.io import load
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from hybridnet import dense_rnn_net
from denseunet3d import denseunet_3d
import keras.backend as K
import os
import time
from loss import weighted_crossentropy
from skimage.transform import resize
import argparse
K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
#  data folder
parser.add_argument('-data', type=str, default='../data/liver', help='test images')
parser.add_argument('-save_path', type=str, default='./test_result/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./Experiments/H_model/weights.04-0.13.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
parser.add_argument('-mode', type=str, default='end2end')

#  data augment
parser.add_argument('-mean', type=int, default=48)
args = parser.parse_args()
      
def train_and_predict():    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = dense_rnn_net(args)
    model.load_weights(args.model_weight, by_name=True)
    print (model.summary())

    #  liver tumor LITS
    trainidx = list(range(28))
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    for idx in range(28):
        img, img_header = load(args.data + '/TestData/volume-' + str(idx) + '.nii' )
        tumor, tumor_header = load(args.data + '/TestData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)
    print('-'*30)
    print('Fitting model......')
    print('-'*30)
    loss_and_metrics = model.evaluate(img_list,tumor_list, batch_size=1)
    print(loss_and_metrics)
#     model.fit_generator(generate_arrays_from_file(args.b, trainidx, img_list, tumor_list, tumorlines, liverlines,
#                                                   tumoridx, liveridx, minindex_list, maxindex_list),
#                         steps_per_epoch=steps,
#                         epochs= 5, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10, workers=1, use_multiprocessing=False)
    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict(args)


# In[ ]:




