"""
Created on 2019-08-30
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
from __future__ import division

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
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
import shutil #form moving files
import numpy as np
import time
import datetime
import sys

import pandas as pd

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf



# functions

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.10
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def clean_gpu_session():
    sess = tf.Session()
    sess.close()


def getMulticlassPreloadCholletInference(models, df, folder_path):

    # return array(inference_c1, inference_c2, inference_c3, inference_c4, inference_c5, inference_c6)

    # print("Starting multiclass inference...")
    # t_start_new = datetime.datetime.now()
    # t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
    # print( t_start_str_new )


    #vars
    count_imgs = 0

    inference_results = []
    df_ii_inferences = pd.DataFrame( columns=('ii', 'multi-acc-0', 'multi-acc-1', 'multi-acc-2', 'multi-acc-3', 'multi-acc-4', 'multi-acc-5') ) #to link ii with 'multi-acc-X' to can make a merge


    #inference for each file
    for index, row in df.iterrows():
        file_name = row['filename']
        img_path = join( folder_path, file_name )
        if isfile( img_path ):

            #mac files
            if file_name.startswith('.DS_Store') or file_name.startswith('._'):
                continue

            img = load_img( img_path, target_size = (models.multi_model_img_width, models.multi_model_img_height) )

            img = img_to_array( img )
            img = img / 255
            img = np.expand_dims( img, axis=0 )

            inference = models.multi_classification_model.predict( img )


            inference_results.append((inference[0][0], inference[0][1], inference[0][2], inference[0][3], inference[0][4], inference[0][5]))

            df_ii_inferences.loc[ 0 if pd.isnull( df_ii_inferences.index.max() ) else df_ii_inferences.index.max() + 1 ] = [int(row['ii'])] + [float(inference[0][0])] + [float(inference[0][1])] + [float(inference[0][2])] + [float(inference[0][3])] + [float(inference[0][4])] + [float(inference[0][5])]

            count_imgs+=1

    df_ii_inferences['ii'] = df_ii_inferences['ii'].astype(int)

    return df_ii_inferences
    #return inference_results



def getMulticlassCholletInference(df, folder_path):

    # return array(inference_c1, inference_c2, inference_c3, inference_c4, inference_c5, inference_c6)

    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # CONFING

    # dimensions of our images.
    img_width, img_height = 190, 190

    model_weights_path = './weights/multiclass_tinyNN_x190_adam_batchnorm_0.8653testacc_ep20_bs16_20190605-1504.h5'


    # print("Starting multiclass inference...")
    # t_start_new = datetime.datetime.now()
    # t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
    # print( t_start_str_new )


    # build the network

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation=None, input_shape=(img_width, img_height, 3)))
    model.add( BatchNormalization() )
    model.add( Activation('relu') )
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add( BatchNormalization() )
    model.add( Activation('relu') )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add( BatchNormalization() )
    model.add( Activation('relu') )
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))


    #model.summary()

    model.load_weights(model_weights_path)
    #print("Model loaded")
    #print("")



    #vars
    count_imgs = 0

    inference_results = []
    df_ii_inferences = pd.DataFrame( columns=('ii', 'multi-acc-0', 'multi-acc-1', 'multi-acc-2', 'multi-acc-3', 'multi-acc-4', 'multi-acc-5') ) #to link ii with 'multi-acc-X' to can make a merge


    #inference for each file
    for index, row in df.iterrows():
        file_name = row['filename']
        img_path = join( folder_path, file_name )
        if isfile( img_path ):

            #mac files
            if file_name.startswith('.DS_Store') or file_name.startswith('._'):
                continue

            img = load_img( img_path, target_size = (img_width, img_height) )

            img = img_to_array( img )
            img = img / 255
            img = np.expand_dims( img, axis=0 )

            inference = model.predict( img )


            inference_results.append((inference[0][0], inference[0][1], inference[0][2], inference[0][3], inference[0][4], inference[0][5]))

            df_ii_inferences.loc[ 0 if pd.isnull( df_ii_inferences.index.max() ) else df_ii_inferences.index.max() + 1 ] = [int(row['ii'])] + [float(inference[0][0])] + [float(inference[0][1])] + [float(inference[0][2])] + [float(inference[0][3])] + [float(inference[0][4])] + [float(inference[0][5])]

            count_imgs+=1


    df_ii_inferences['ii'] = df_ii_inferences['ii'].astype(int)

    #release gpu memory
    clean_gpu_session()


    return df_ii_inferences
    #return inference_results


def getMulticlassArtifactClassification(df):

    artifact_list_ids = df['artifact-id'].unique()
    df_artifacts_inferences = pd.DataFrame( columns=('artifact-id', 'artifact-multi-acc-0', 'artifact-multi-acc-1', 'artifact-multi-acc-2', 'artifact-multi-acc-3', 'artifact-multi-acc-4', 'artifact-multi-acc-5', 'artifact-multi-decision') ) #to link artifact-id with artifact-multi-acc-X and decision to can make a merge


    ##for index, artifact_id in artifact_list_ids: ##nope --> TypeError: cannot unpack non-iterable numpy.int64 object

    for artifact_id in range(len(artifact_list_ids) ):  ###issue
    #for _, artifact_id in enumerate(artifact_list_ids):
        df_artifact = df[ df['artifact-id'] == artifact_list_ids[artifact_id] ]
        #df_artifact = df[ df['artifact-id'] == artifact_id ]


        list_acc_columns = np.array(['multi-acc-0','multi-acc-1','multi-acc-2','multi-acc-3','multi-acc-4','multi-acc-5'])
        artifact_inferences_means = df_artifact[ list_acc_columns ].mean()

        #print(artifact_inferences_means.max())
        # also with:  np.argmax( artifact_inferences_means.to_numpy() )
        #artifact_inferences_means.idxmax() --> bin-acc-pos (I need to convert into 0-X index)

        specie_index = np.where( list_acc_columns == artifact_inferences_means.idxmax() )[0][0]

        df_artifacts_inferences.loc[ 0 if pd.isnull( df_artifacts_inferences.index.max() ) else df_artifacts_inferences.index.max() + 1 ] = \
        np.array( [int( artifact_list_ids[artifact_id] )] + (artifact_inferences_means.to_numpy()).tolist() + [int( specie_index )] )
        #np.array( [int( artifact_id + 1 )] + (artifact_inferences_means.to_numpy()).tolist() + [int( specie_index )] )



    #convert from float to int
    df_artifacts_inferences['artifact-id'] = df_artifacts_inferences['artifact-id'].astype(int)
    df_artifacts_inferences['artifact-multi-decision'] = df_artifacts_inferences['artifact-multi-decision'].astype(int)

    return df_artifacts_inferences


def getBestArtifact(df):

    max_val_list = df[['multi-acc-0','multi-acc-1','multi-acc-2','multi-acc-3','multi-acc-4','multi-acc-5']].max() #max. per species (column)

    #max_val = df[['multi-acc-0','multi-acc-1','multi-acc-2','multi-acc-3','multi-acc-4','multi-acc-5']].values.max()
    max_val = max_val_list.max()
    max_col_name = max_val_list.idxmax() #to get the column name to export results
    max_index_species = np.argmax( max_val_list.to_numpy() )  #to get species index

    max_element = df[ df[['multi-acc-0','multi-acc-1','multi-acc-2','multi-acc-3','multi-acc-4','multi-acc-5']].eq( max_val ).any(1) ]

    #results
    df_results = pd.DataFrame()
    df_results[['filename','acc']] = max_element[['filename', max_col_name]]
    df_results.insert(1,'species', max_index_species)

    return df_results.head(1) #filename, species, acc


def getBestArtifactFromSegmentation(df):
    #also it is needed to save the multi-species acc

    df_max_segm = df.loc[df['detection-acc'].idxmax()] # get the row of max value (only one value)
    df_max_segm_artifact_acc = df_max_segm[['artifact-multi-acc-0','artifact-multi-acc-1','artifact-multi-acc-2','artifact-multi-acc-3','artifact-multi-acc-4','artifact-multi-acc-5']]

    artifact_acc_max_val = df_max_segm_artifact_acc.max()
    artifact_max_col_name = df_max_segm_artifact_acc.astype('float64').idxmax() #to get the column name to export results // no more needed
    artifact_max_index_species = np.argmax( df_max_segm_artifact_acc.to_numpy() )  #to get species index


    #results
    df_results = pd.DataFrame()
    #df_results[['filename','acc']] = df_max_segm[['filename', artifact_max_col_name]]
    #df_results.insert(1,'species', artifact_max_index_species)
    df_results.insert(0,'filename', df_max_segm[['filename']])
    df_results.insert(1,'species', artifact_max_index_species)
    df_results.insert(2,'acc', artifact_acc_max_val)
    #df_results.insert(2,'acc', df_max_segm[ artifact_max_col_name ] )

    return df_results.head(1) #filename, species, acc



##
