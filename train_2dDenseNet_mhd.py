
# coding: utf-8

# In[8]:


from __future__ import print_function
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
import random
from medpy.io import load
import numpy as np
import argparse
import keras
from keras.layers import concatenate
from keras.layers import Lambda
from keras.models import Model
import keras.backend as K
from loss import weighted_crossentropy_2ddense
import os
from skimage.transform import resize
import sys
import nibabel as nib
import SimpleITK as sitk


# In[9]:


keras.__version__


# In[10]:


K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras 2d denseunet Training')
#  data folder
data='../data/prostate'
save_path='./ProstateExperiments/'
image_path='../data/prostate/TrainingData_Part1'
#  other paras
batch_size =5
input_size = 224
model_weight = '../model/densenet161_weights_tf.h5'
input_cols =3

#  data augment
MEAN = 48
thread_num =14


#liverlist = [32,34,38,41,47,87,89,91,102]


# In[11]:


from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from lib.custom_layers import Scale

def DenseUNet(nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, weights_path=None,
              batch_size=None, input_size=None):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(batch_shape=(batch_size, input_size, input_size, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161
    box = []
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    box.append(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        box.append(x)
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    box.append(x)

    up0 = UpSampling2D(size=(2,2))(x)
    line0 = Conv2D(2208, (1, 1), padding="same", kernel_initializer="normal", name="line0")(box[3])
    up0_sum = add([line0, up0])
    conv_up0 = Conv2D(768, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up0")(up0_sum)
    bn_up0 = BatchNormalization(name = "bn_up0")(conv_up0)
    ac_up0 = Activation('relu', name='ac_up0')(bn_up0)

    up1 = UpSampling2D(size=(2,2))(ac_up0)
    up1_sum = add([box[2], up1])
    conv_up1 = Conv2D(384, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up1")(up1_sum)
    bn_up1 = BatchNormalization(name = "bn_up1")(conv_up1)
    ac_up1 = Activation('relu', name='ac_up1')(bn_up1)

    up2 = UpSampling2D(size=(2,2))(ac_up1)
    up2_sum = add([box[1], up2])
    conv_up2 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up2")(up2_sum)
    bn_up2 = BatchNormalization(name = "bn_up2")(conv_up2)
    ac_up2 = Activation('relu', name='ac_up2')(bn_up2)

    up3 = UpSampling2D(size=(2,2))(ac_up2)
    up3_sum = add([box[0], up3])
    conv_up3 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up3")(up3_sum)
    bn_up3 = BatchNormalization(name = "bn_up3")(conv_up3)
    ac_up3 = Activation('relu', name='ac_up3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name="conv_up4")(up4)
    conv_up4 = Dropout(rate=0.3)(conv_up4)
    bn_up4 = BatchNormalization(name="bn_up4")(conv_up4)
    ac_up4 = Activation('relu', name='ac_up4')(bn_up4)

    x = Conv2D(3, (1,1), padding="same", kernel_initializer="normal", name="dense167classifer")(ac_up4)

    model = Model(img_input, x, name='denseu161')


    return model

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


# In[12]:


def make_parallel(model, gpu_count, mini_batch):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        # print ("data",data)
        # print ("shape",shape[:1])
        # print (shape[1:])
        
        # size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        # print (size)
        # print ('1',shape[:1] // parts)
        # print ('2',shape[1:]*0)
        # stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        # print (stride)
        # start = stride * idx
        # print (start)
        # print ('return',tf.slice(data,start,size))
        # # exit(0)
        # print ('idx', idx*mini_batch,(idx+1)*mini_batch )
        return data[idx*mini_batch:(idx+1)*mini_batch,:, :,:]
        # data[25:50, :, :, :]
        # return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])
    # print (outputs_all)
    #Place a copy of the model on each GPU, each getting a slice of the batch

    for i in range(gpu_count):
        id = i
        # print ('loading'+str(id))
        with tf.device('/gpu:%d' % id):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # print ('ssssssssssss')
                # print ('rr',model.inputs)
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    # print ('x', x)
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    # print (input_shape)
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    # print ('slice_n', slice_n)
                    inputs.append(slice_n)

                # print ('ii',inputs)
                outputs = model(inputs)
                # print ('xx',outputs)

                # print ('ssdadsa')
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # print ('ssd')
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
                # print ('hard')
    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
        return Model( outputs=merged, inputs= model.inputs)


# In[ ]:


def generate_arrays_from_file(batch_size, trainidx, img_list, prostate_list, prostatelines, prostateidx,  minindex_list, maxindex_list):
    while 1:
        X = np.zeros((batch_size, input_size, input_size, input_cols), dtype='float32')
        Y = np.zeros((batch_size, input_size, input_size, 1), dtype='int16')
        result_list = []
        for idx in range(batch_size):
            count = random.choice(trainidx)
            img = img_list[count]
            prostate = prostate_list[count]
            minindex = minindex_list[count]
            maxindex = maxindex_list[count]
            
            lines = prostatelines[count]
            numid = prostateidx[count]
                
#             if len(lines)==1:
#                 lines[0] = "0 0 0"
            #  randomly scale
#             scale = np.random.uniform(0.8,1.2)
#             deps = int(input_size * scale)
#             rows = int(input_size * scale)
#             cols = 3

#             sed = np.random.randint(1,numid)
#             cen = lines[sed-1]
#             cen = np.fromstring(cen, dtype=int, sep=' ')

#             a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
#             b = min(max(minindex[1] + rows/2, cen[0]), maxindex[1]- rows/2-1)
#             c = min(max(minindex[2] + cols/2, cen[0]), maxindex[2]- cols/2-1)

#             cropp_img = img[int(a - deps / 2):int(a + deps / 2), int(b - rows / 2):int(b + rows / 2),
#                         int(c - cols / 2): int(c + cols / 2 + 1)].copy()
#             cropp_prostate = prostate[int(a - deps / 2):int(a + deps / 2), int(b - rows / 2):int(b + rows / 2),
#                           int(c - cols / 2):int(c + cols / 2 + 1)].copy()

#             cropp_img -= MEAN
#              # randomly flipping
#             flip_num = np.random.randint(0, 8)
#             if flip_num == 1:
#                 cropp_img = np.flipud(cropp_img)
#                 cropp_prostate = np.flipud(cropp_prostate)
#             elif flip_num == 2:
#                 cropp_img = np.fliplr(cropp_img)
#                 cropp_prostate = np.fliplr(cropp_prostate)
#             elif flip_num == 3:
#                 cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
#                 cropp_prostate = np.rot90(cropp_prostate, k=1, axes=(1, 0))
#             elif flip_num == 4:
#                 cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
#                 cropp_prostate = np.rot90(cropp_prostate, k=3, axes=(1, 0))
#             elif flip_num == 5:
#                 cropp_img = np.fliplr(cropp_img)
#                 cropp_prostate = np.fliplr(cropp_prostate)
#                 cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
#                 cropp_prostate = np.rot90(cropp_prostate, k=1, axes=(1, 0))
#             elif flip_num == 6:
#                 cropp_img = np.fliplr(cropp_img)
#                 cropp_prostate = np.fliplr(cropp_prostate)
#                 cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
#                 cropp_prostate = np.rot90(cropp_prostate, k=3, axes=(1, 0))
#             elif flip_num == 7:
#                 cropp_img = np.flipud(cropp_img)
#                 cropp_prostate = np.flipud(cropp_prostate)
#                 cropp_img = np.fliplr(cropp_img)
#                 cropp_prostate = np.fliplr(cropp_prostate)

#             cropp_prostate = resize(cropp_prostate, (input_size,input_size,input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
#             cropp_img   = resize(cropp_img, (input_size,input_size,input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
            
#             result_list.append([cropp_img, cropp_prostate[:,:,1]])
            
#         for idx in range(len(result_list)):
#             X[idx, :, :, :] = result_list[idx][0]
#             Y[idx, :, :, 0] = result_list[idx][1]
        yield (X,Y)
    


def load_fast_files():
    trainidx = list(range(50))
    img_list = []
    prostate_list = []
    minindex_list = []
    maxindex_list = []
    prostatelines = []
    prostateidx = []
    for idx in range(50):
        subject_name = 'Case%02d' % idx
        mhd = os.path.join(image_path, subject_name+'.mhd')
        img = sitk.ReadImage(mhd)
        img = sitk.GetArrayFromImage(img)
        #img, img_header = load(data+ '/myTrainingData/volume-' + str(idx) + '.nii')
        
        subject_name = 'Case%02d' % idx
        mhd = os.path.join(image_path, subject_name+'_segmentation.mhd')
        prostate = sitk.ReadImage(mhd)
        prostate = sitk.GetArrayFromImage(prostate)
        
        img_list.append(img)
        prostate_list.append(prostate)

        maxmin = np.loadtxt(data + '/myTrainingDataTxt/ProstateBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0] + 3)
        maxindex[1] = min(img.shape[1], maxindex[1] + 3)
        maxindex[2] = min(img.shape[2], maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        
        f2 = open(data + '/myTrainingDataTxt/ProstatePixels/prostate_' + str(idx) + '.txt', 'r')
        prostateline = f2.readlines()   #prostate的分割线像素点集合
        prostatelines.append(prostateline)
        prostateidx.append(len(prostateline))
        f2.close()
    return trainidx, img_list, prostate_list, prostatelines, prostateidx, minindex_list, maxindex_list

def train_and_predict():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = DenseUNet(reduction=0.5, batch_size=batch_size,input_size=input_size)
    model.load_weights(model_weight, by_name=True)
    #model = make_parallel(model, batch_size//10, mini_batch=10)
    sgd = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy_2ddense])

    trainidx, img_list, prostate_list, prostatelines, prostateidx, minindex_list, maxindex_list = load_fast_files()

    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_path + "/model"):
        os.mkdir(save_path + '/model')
        os.mkdir(save_path + '/history')
    else:
        if os.path.exists(save_path+ "/history/lossbatch.txt"):
            os.remove(save_path + '/history/lossbatch.txt')
        if os.path.exists(save_path + "/history/lossepoch.txt"):
            os.remove(save_path + '/history/lossepoch.txt')

    model_checkpoint = keras.callbacks.ModelCheckpoint(save_path + '/model/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)

    steps = 500/batch_size
    
    model.fit_generator(generate_arrays_from_file(batch_size, trainidx, img_list, prostate_list, prostatelines,  prostateidx,
                                                   minindex_list, maxindex_list),steps_per_epoch=steps,
                                                    epochs= 1000, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10,
                                                    workers=1, use_multiprocessing=False)

    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict()


# In[ ]:




