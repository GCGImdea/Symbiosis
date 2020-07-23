"""
Created on 2019-08-30
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
from __future__ import division

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

#from keras import applications
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

import math

import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil #for moving files
import numpy as np
import time
import datetime
import sys

import pandas as pd


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
#import traceback
###export CUDA_VISIBLE_DEVICES='' #in console
###import os
###os.environ["CUDA_VISIBLE_DEVICES"]='0'

# functions

def get_session():
    #input_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.10 # too try to avoid the warning  CUDA_ERROR_OUT_OF_MEMORY (with 0.5 gives it, with 0.30 or less no, and the performace is increased)
    #config.allow_soft_placement = True #no needed here
    #config.log_device_placement=True
    config.gpu_options.allow_growth = True
    return tf.Session(config=config) ##graph=input_graph,config=config


def clean_gpu_session():
    #bck_sess = keras.backend.tensorflow_backend.get_session()
    sess = tf.Session()
    sess.close()



def getBinaryVGG16Inference(df, folder_path):

    # return array(inference_c1, inference_c2, inference_c3, inference_c4, inference_c5, inference_c6)

    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #clean previous gpu memory
    ###clean_gpu_session()  ## if called --> Internal: CUDA runtime implicit initialization on GPU:0 failed. Status: unknown error.  ....  Failed to create session.

    # set the modified tf session as backend in keras
    # try:
    #     keras.backend.tensorflow_backend.set_session(get_session())
    # except Exception as e:
    #     traceback.print_exc()
    keras.backend.tensorflow_backend.set_session(get_session())


    # CONFING

    # dimensions of our images.
    img_width, img_height = 150, 150

    model_weights_path = './weights/VGG16froz_lay256_x150_binary_fish_0.9809tstacc_20190507-1833.h5'

    #if(debug):
        # print("Starting binary inference...")
        # t_start_new = datetime.datetime.now()
        # t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
        # print( t_start_str_new )



    # build the VGG16 network

    #model_vgg = VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
    model_vgg = VGG16(weights=None, include_top=False, input_shape = (img_width, img_height, 3))
    #print('Model loaded')
    #print('')

    model = Sequential()

    for layer in model_vgg.layers:
        model.add(layer)#top_model.add(Flatten(input_shape=(7, 7, 512)))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))


    model.add(top_model)

    model.load_weights(model_weights_path)
    #print("Model loaded")
    #print("")


    #vars
    count_imgs = 0

    inference_results = []
    df_ii_inferences = pd.DataFrame( columns=('ii', 'bin-acc-neg', 'bin-acc-pos') ) #to link ii with bin-acc-X to can make a merge


    #inference for each file
    for index, row in df.iterrows():
        file_name = row['filename']
        img_path = join( folder_path, file_name )

        # print("File: %s" % img_path)

        if isfile( img_path ):

            #mac files
            if file_name.startswith('.DS_Store') or file_name.startswith('._'):
                continue

            img = load_img( img_path, target_size = (img_width, img_height) )

            img = img_to_array( img )
            img = img / 255
            img = np.expand_dims( img, axis=0 )

            inference = model.predict( img )
            # POSITIVE : inference[0][1]
            # NEGATIVE : inference[0][0]


            inference_results.append((inference[0][0], inference[0][1]))

            df_ii_inferences.loc[ 0 if pd.isnull( df_ii_inferences.index.max() ) else df_ii_inferences.index.max() + 1 ] = [int(row['ii'])] + [float(inference[0][0])] + [float(inference[0][1])]

            count_imgs+=1


    #release gpu memory
    clean_gpu_session()


    return df_ii_inferences
    #return np.array( inference_results )


def getBinaryArtifactClassification(df):

    artifact_list_ids = df['artifact-id'].unique()
    df_artifacts_inferences = pd.DataFrame( columns=('artifact-id', 'artifact-bin-acc-neg', 'artifact-bin-acc-pos', 'artifact-bin-decision') ) #to link artifact-id with artifact-bin-acc-X and decision to can make a merge


    ##for index, artifact_id in artifact_list_ids: ##nope --> TypeError: cannot unpack non-iterable numpy.int64 object

    for artifact_id in range(len(artifact_list_ids) ):
        df_artifact = df[ df['artifact-id'] == artifact_list_ids[artifact_id] ]


        list_bin_acc_columns = np.array(['bin-acc-neg','bin-acc-pos'])
        artifact_inferences_means = df_artifact[ list_bin_acc_columns ].mean()

        #print(artifact_inferences_means.max())
        # also with:  np.argmax( artifact_inferences_means.to_numpy() )
        #artifact_inferences_means.idxmax() --> bin-acc-pos (I need to vonver into 0-X index)

        specie_index = np.where( list_bin_acc_columns == artifact_inferences_means.idxmax() )[0][0]

        df_artifacts_inferences.loc[ 0 if pd.isnull( df_artifacts_inferences.index.max() ) else df_artifacts_inferences.index.max() + 1 ] = \
        np.array( [int( artifact_id + 1 )] + (artifact_inferences_means.to_numpy()).tolist() + [int( specie_index )] )

    #convert from float to int
    df_artifacts_inferences['artifact-id'] = df_artifacts_inferences['artifact-id'].astype(int)
    df_artifacts_inferences['artifact-bin-decision'] = df_artifacts_inferences['artifact-bin-decision'].astype(int)

    return df_artifacts_inferences
