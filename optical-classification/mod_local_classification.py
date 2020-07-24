"""
Created on 2019-09-15
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
#imports
from __future__ import division
import math

from mod_read_crop_files import *
#from mod_tracker import *
from mod_bin_classifier import *
from mod_multiclass_classifier import *

from sort import *

import numpy as np
import pandas as pd

import os
from os import listdir
from os.path import isfile, join
import shutil #for moving files

import datetime
import sys

import argparse


import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from keras.applications import VGG16
#from keras.applications.vgg16 import preprocess_input


import tensorflow as tf



###### USE ########
# import:
#   from mod_local_classification import *
#
# Load NN classification models:
#   loadClassificationModels()
#
# Main function (examples):
#   localClassification()
#   localClassification('./crops/', True)
#   localClassification(input_path='./crops/')
#
# optional parameters
#   input_path:      to provide to provide the folder with the segments (crops) from previous fish detection stage (RetinaNet results). Default: ./crops/
#   debug:           use for debugging [False by default].



# CONFIG

output_classification_folder = "../classification/local_output/" # the folder will contain the restuls and other files for the species classification
output_classification_csv_file = "camera_classification.csv"
## CSV (initial) content: FILENAME, FRAME-NUMBER, COORDS.
## to feed the tracker as first step in the dataflow.
## Filename --> just the name without path and including extension.
## Filename ex.: IMG_20190830_120011rectangle720_850_1020_940.jpg
## CSV line ex.:
## IMG_20190830_120011rectangle720_850_1020_940.jpg,1,720,850,1020,940
output_classification_best_file = "best_classification.txt"
output_detection_best_file = "best_detection.txt"

output_classification_path_to_crops_file = "path_crops_folder.txt"



###### functions ########



def saveResults(df, folder_path, output_file):
    return df.to_csv( join( folder_path, output_file) , index=False)    #,encoding='utf-8'


def savePathToCrops(crops_folder_path, working_folder_path, file_name):

    output_file = None
    status = None

    crops_folder_path = os.path.abspath( crops_folder_path )

    try:
        #output_file = open( '%s/%s.txt' % (working_out_folder_path, species_name), 'w') ##join()
        output_file = open( '%s/%s' % (working_folder_path, file_name), 'w') ##join()
        output_file.write('%s\r\n' % ( crops_folder_path ) )
        output_file.close()

        if status is None and not status:
            status = True

    except IOError:
        status = False
    finally:
        if output_file is not None:
            output_file.close() #just in case

    return status


def cleanFilesInFolder(folder_path):

    for file_name in listdir( folder_path ):
        file_name_wPath = join( folder_path, file_name )
        if isfile( file_name_wPath ):
            os.remove( file_name_wPath )


def loadClassificationModels(mm):

    ##bin_model_img_width, bin_model_img_height = 150, 150
    ##multi_model_img_width, multi_model_img_height = 190, 190

    # CONFIG
    ##bin_model_weights_path = './weights/VGG16froz_lay256_x150_binary_fish_0.9809tstacc_20190507-1833.h5'
    ##multi_model_weights_path = './weights/multiclass_tinyNN_x190_adam_batchnorm_0.8653testacc_ep20_bs16_20190605-1504.h5'

    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.10
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

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

    #if(debug):
        # print("Starting binary inference...")
        # t_start_new = datetime.datetime.now()
        # t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
        # print( t_start_str_new )



    # BINARY MODEL

    # build the VGG16 network

    mm.binary_classification_model_vgg = VGG16(weights=None, include_top=False, input_shape = (mm.bin_model_img_width, mm.bin_model_img_height, 3)) ## global var.??
    #print('Model loaded')
    #print('')

    mm.binary_classification_model = Sequential()

    for layer in mm.binary_classification_model_vgg.layers:
        mm.binary_classification_model.add(layer)#top_model.add(Flatten(input_shape=(7, 7, 512)))

    mm.binary_classification_model_top = Sequential()
    mm.binary_classification_model_top.add(Flatten(input_shape=mm.binary_classification_model.output_shape[1:]))
    mm.binary_classification_model_top.add(Dense(256, activation='relu'))
    mm.binary_classification_model_top.add(Dropout(0.5))
    mm.binary_classification_model_top.add(Dense(2, activation='softmax'))


    mm.binary_classification_model.add(mm.binary_classification_model_top)

    mm.binary_classification_model.load_weights( mm.bin_model_weights_path )
    #print("Model loaded")
    #print("")



    # MULTICLASS MODEL

    # build the network

    mm.multi_classification_model = Sequential()

    mm.multi_classification_model.add(Conv2D(32, (3, 3), activation=None, input_shape=(mm.multi_model_img_width, mm.multi_model_img_height, 3)))
    mm.multi_classification_model.add( BatchNormalization() )
    mm.multi_classification_model.add( Activation('relu') )
    mm.multi_classification_model.add(Conv2D(32, (3, 3), activation='relu'))
    mm.multi_classification_model.add( BatchNormalization() )
    mm.multi_classification_model.add( Activation('relu') )
    mm.multi_classification_model.add(MaxPooling2D(pool_size=(2,2)))
    mm.multi_classification_model.add(Dropout(0.25))

    mm.multi_classification_model.add(Flatten())
    mm.multi_classification_model.add(Dense(128, activation='relu'))
    mm.multi_classification_model.add( BatchNormalization() )
    mm.multi_classification_model.add( Activation('relu') )
    mm.multi_classification_model.add(Dropout(0.5))
    mm.multi_classification_model.add(Dense(6, activation='softmax'))


    #model.summary()

    mm.multi_classification_model.load_weights( mm.multi_model_weights_path )
    #print("Model loaded")
    #print("")




def getTargetsId(df): #old name: getTargetsIdFromPandas

    total_time = 0.0
    total_frames = 0

    # if (debug):
    #     colours = np.random.rand(64,3) #for debugging
    #
    #     wbbox_folder_path = working_folder_path + '/frames_wbbox'
    #     if not os.path.exists( wbbox_folder_path ):
    #
    #     	print("[DEGUG] Creating dir: " + str( wbbox_folder_path ) )
    #     	print("")
    #     	os.makedirs( wbbox_folder_path )
    #
    #     plt.ion()
    #     fig = plt.figure(figsize=(15,10))

    df_ii_id = pd.DataFrame( columns=('ii', 'artifact-id') ) #to link ii with artifact-id to can make a merge


    artifact_tracker = Sort() #create instance of the SORT tracker


    for frame in range( int( df['frame'].max() ) ):
        frame += 1 #detection and frame numbers begin at 1

        df_frame = df[ df['frame'] == frame ]
        #'ii', 'filename', 'frame', 'x1', 'y1', 'x2', 'y2', 'detection-acc'

        if len( df_frame ):
            dets = df_frame[['x1', 'y1', 'x2', 'y2', 'detection-acc', 'ii']].to_numpy() #copy=True
            #dets_noindex = df_frame[['x1', 'y1', 'x2', 'y2', 'detection-acc']].to_numpy() #copy=True
            dets_noindex = dets[:,:-1]
            #dets = df_frame.iloc[:, list(range(2,7)) ].to_array()

        else:
            #dets = np.array([]) #a "empty row" needed when to target in frame
            dets_noindex = np.array([]) #a "empty row" needed when to target in frame



        total_frames += 1

        # if(debug):
        #     ax1 = fig.add_subplot(111, aspect='equal')
        #     #fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
        #
        #     folder_name = seq.split('rectangle')[0]
        #     #folder_name = seq.split('-frame')[0]
        #
        #             ######## MODIFYYY folder
        #     #fn = 'xxxxxxxxxxxxxxxxxxxx/%s/frames/%06d.jpg'%(folder_name,frame)
        #     fn = 'data/dorado/frames/%s-frame%s.png'%(folder_name, frame)
        #     im =io.imread(fn)
        #     ax1.imshow(im)
        #     #plt.title(seq+' Tracked Targets')
        #     plt.title('Frame' + str(frame) + ' << __________ >>' ) ##can be changed if needed later in the code
        #
        # if(debug):
        #     start_time = time.time()




        trackers = artifact_tracker.update( dets_noindex ) #'x1', 'y1', 'x2', 'y2', 'detection-acc', 'ii'
        #return a np array where each row contains a valid bounding box and track_id (last column-1)



        # if(debug):
        #     cycle_time = time.time() - start_time
        #     total_time += cycle_time


        match_results = np.empty((0, len(df_frame) ), float) #matrix to store all matching metric results: columns>orig_coords, rows>tracker_coords


        for d in trackers:

            match_partial_results = [] #for each tracker coord prediction (to store matching metric with each original cooord.) ## py list - must be re-nicialize each loop

            if d[5] == 0: #only coords that match with orginal targets (provided by tracker) ## d[5]: frames without original target provided
                tracker_coords = d[:-2] #x1, y1, x2, y2, ID, num_no_obj

                # d[5]: number of frames with (original) target provided. IMP!!!!
                # d[4]: ID artifact
                # d[0]:
                # d[1]:
                # d[2]:
                # d[3]:

                #calculation of the matching metric per each original coord.
                for orig_coords in dets:    #'x1', 'y1', 'x2', 'y2', 'detection-acc', 'ii'


                    #min_val = getCoordsMatchingMetric(orig_coords[:-2], tracker_coords)
                    #match_partial_results.append( min_val )

                    #to create (stacking) a matrix with all the results: columns>orig_coords, rows>tracker_coords
                    match_partial_results.append( iou(tracker_coords, orig_coords[:-2]) ) #type - python list


                if len(df_frame):  # this can go one level above

                    #matrix with all the results: columns>orig_coords, rows>tracker_coords
                    match_results = np.vstack((match_results, match_partial_results)) #type - np.array



                # if(debug):
                #     d = d.astype(np.int32) ########## VS 64
                #     #iou calculation over all targets
                #     iou_max = 0.
                #
                #     for coord_orig in dets :
                #         iou_max = max(iou_max, iou(d.astype(np.float), coord_orig) )
                #
                #     label_id_x = d[0]-40 #max(0, d[0]-40)
                #     label_id_y = d[1]-8 #max(0, d[1]-8)
                #     label_iou_x = label_id_x
                #     label_iou_y = d[3]+40 #max(0, d[3]+40)
                #     label_noobj_nframes = d[5] #max(0, d[3]+40)
                #
                #     if ( len(df_frame) != 0 ) and iou_max != 0. and label_noobj_nframes == 0:
                #         text_color = colours[d[4]%64,:]
                #     else:
                #         text_color = 'white' #white, red
                #
                #     if text_color in ('white', 'red'): #white, red
                #         ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1], linestyle = 'dotted', fill=False,lw=2,ec=text_color)) #dotted, dashed, solid
                #     else :
                #         ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=2,ec=text_color))
                #     ax1.set_adjustable('box')
                #
                #
                #     plt.text(label_id_x, label_id_y, d[4], fontsize=16, fontweight='bold', color=text_color)
                #     #d[0]-60
                #     plt.text(label_id_x, label_iou_y, '%.4f/%d'%(iou_max, label_noobj_nframes), fontsize=16, fontweight='bold', color=text_color) #iou(bb_test,bb_gt)
                #
                #     if ( len(df_frame) != 0 ):
                #         plt.title('Frame' + str(frame) + ' << OBJECT x%d>>'%dets.shape[0] )
                #     else:
                #         plt.title('Frame' + str(frame) + ' << __________ >>' )
            else:
                if len(df_frame):
                    match_partial_results.append( [-1] * len(df_frame) ) #fill with '-1' when the coord from the tracker are not associated to an original target in that frame (d[5]!=0)
                    match_results = np.vstack((match_results, match_partial_results))

            del match_partial_results



        if len(df_frame):

            match_max_results = np.argwhere( match_results == np.max(match_results, axis=0) ) #max by columns, give the positions in the matrix.
            match_max_results = match_max_results[ np.argsort( match_max_results[:,-1] ) ] #re-sort by orig_coords order (columns in match_results, rows are associated to tracker coords.) ## take las column as new matrix index order.


            for index_internal_orig_coords in range( len(df_frame) ):

                index_tracker = match_max_results[ index_internal_orig_coords ][0] #give the tracker index --> match with ID (use index to get ID in trackers[])

                artifact_id = trackers[ index_tracker ][4]

                index_orginal = df_frame.iloc[index_internal_orig_coords, np.where(df_frame.columns == 'ii')[0][0] ]
                ## (( np.where(df_frame.columns == 'ii')[0][0] --> provide the column index/position from the name))
                #also with 'df_frame.iloc[ index_internal_orig_coords ,0]'


                df_ii_id.loc[ 0 if pd.isnull( df_ii_id.index.max() ) else df_ii_id.index.max() + 1 ] = [int(index_orginal)] + [int(artifact_id)]


            del match_results




        # if(debug):
        #     fig.canvas.flush_events()
        #     plt.draw()
        #
        #     ######## MODIFYYY folder
        #     name_fig = 'data/dorado/frames_wbbox/%s-frame%s.png'%(folder_name, frame)
        #     fig.savefig( name_fig )
        #     ax1.cla()



    #print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

    artifact_tracker = None
    #del artifact_tracker
    #artifact_tracker.frame_count

    return df_ii_id



###### functions-end ########




###### main-function ########

def localClassification(models, input_path='./crops/', debug=False):
    # call:
    # localClassification(), localClassification('./crops/', True), localClassification(input_path='./crops/')
    #optional parameters
    #input_path: Parameter to provide the folder with the segments (crops) from previous fish detection stage (RetinaNet results). Default: ./crops/
    #debug:  Use for debugging [False by default].


    # paths
    crops_folder_path = input_path
    if not crops_folder_path.endswith('/'):
        crops_folder_path = crops_folder_path + '/'
    #if crops_folder_path.endswith('/'):
    #    crops_folder_path = crops_folder_path[:-1]

    working_folder_path = os.path.abspath( join( crops_folder_path, output_classification_folder) )


    # main

    print("Starting classification...")
    t_start_new = datetime.datetime.now()
    t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
    print( t_start_str_new )


    #if not os.path.exists( join( crops_folder_path, output_classification_folder ) ):
    if not os.path.exists( working_folder_path ):

    	print("Creating dir: " + str( output_classification_folder ) )
        print( str( working_folder_path ) )
    	print("")
    	os.makedirs( working_folder_path )




    df = getSegmentsAsDataframe( crops_folder_path )

    if len( df ): # not empty DataFrame

        #clean previous results
        cleanFilesInFolder( working_folder_path )

        #TRACKING ARTIFACTS - associate an ID to each artifact/crop/segment
        df_ids = getTargetsId( df )
        df_wIds = pd.merge(df, df_ids, how='outer', on='ii')

        #BINARY INFERENCE
        #df_binInference = getBinaryVGG16Inference(df_wIds, crops_folder_path)
        df_binInference = getBinaryPreloadVGG16Inference(models, df_wIds, crops_folder_path)
        df_wBinInference = pd.merge(df_wIds, df_binInference, how='outer', on='ii')


        if debug :
            print( df )
            print( df_ids )
            #print( df_wIds )
            #print( df_binInference )
            #print( df_wBinInference )
            print( '----------------')
            print( '')
            #print( df[['artifact-id','frame','filename']] )


        #MULTICLASS INFERENCE
        #df_multiInference = getMulticlassCholletInference(df_wBinInference, crops_folder_path)
        df_multiInference = getMulticlassPreloadCholletInference(models, df_wBinInference, crops_folder_path)
        df_wMultiInference = pd.merge(df_wBinInference, df_multiInference, how='outer', on='ii')


        if debug :
            print( df_multiInference )
            print( df_wMultiInference )
            print( '----------------')
            print( '')

        #BINARY ARTIFACTS MEAN AND DESICION
        df_binArtifInference = getBinaryArtifactClassification(df_wMultiInference)
        df_wBinArtifInference = pd.merge(df_wMultiInference, df_binArtifInference, how='outer', on='artifact-id')


        if debug :
            print( 'df_binArtifInference' )
            print( df_binArtifInference )
            print( 'df_wBinArtifInference' )
            print( df_wBinArtifInference )
            print( '----------------')
            print( '')


        #MULTICLASS ARTIFACTS MEAN AND DESICION
        df_multiArtifInference = getMulticlassArtifactClassification( df_wBinArtifInference )
        df_wMultiArtifInference = pd.merge(df_wBinArtifInference, df_multiArtifInference, how='outer', on='artifact-id')


        ## printing result in the console
        if debug :
            print("")
            print("")
            print("[Classification results]")
            #df_wMultiArtifInference_perTarget = df_wMultiArtifInference.copy()
            #df_wMultiArtifInference_perTarget = df_wMultiArtifInference_perTarget[['artifact-id','artifact-bin-decision','artifact-multi-decision']]
            #df_wMultiArtifInference_perTarget =  df_wMultiArtifInference_perTarget.groupby(df_wMultiArtifInference['artifact-id'])
            #df[ ( df['x1'] <= boundary_area_left ) | ( df['x2'] >= boundary_area_right ) ]
            #print( df_wMultiArtifInference.keys() )

            #print( df_wMultiArtifInference_perTarget[ ( df_wMultiArtifInference_perTarget['artifact-id'] == df_wMultiArtifInference_perTarget['artifact-id'].nunique() ) ] )
            #print( df_wMultiArtifInference_perTarget[['artifact-id','artifact-bin-decision','artifact-multi-decision']] )



        if debug :
            print( df_wMultiArtifInference[['artifact-id','artifact-bin-decision','artifact-multi-decision']] )

            print( df_wMultiArtifInference[['artifact-id','frame','filename']] )
            #df_bestDetection.head(1)['filename'].to_numpy()
            #df_specie['artifact-id'].nunique()
            print("")
            print("")
            print("Legend:")
            print("0: albacore")
            print('1: amberjack')
            print('2: atl_mackerel')
            print('3: dorado')
            print('4: med_mackerel')
            print('5: swordfish')
            print('6: others')
            print("")
            # print("")


        #GET BEST ARTIFACTS DETECTION
        #df_bestDetection = getBestArtifact(df_wMultiArtifInference)
        df_bestDetection = getBestArtifactFromSegmentation(df_wMultiArtifInference)

        #save results
        saving_result = saveResults( df_wMultiArtifInference, working_folder_path, output_classification_csv_file)
        #saving_result_best = saveResults( df_bestDetection, working_folder_path, output_classification_best_file)
        saving_result_best = saveResults( df_bestDetection, working_folder_path, output_detection_best_file)



        #save crops path to aux file (for boundary results in central unit)
        savePathToCrops(crops_folder_path, working_folder_path, output_classification_path_to_crops_file)


        #save best file results
        filename_best_result = df_bestDetection.head(1)['filename'].to_numpy()[0]

        filename_best_result_wPath = join(crops_folder_path, filename_best_result)

        if isfile( filename_best_result_wPath):
            shutil.copy( filename_best_result_wPath , working_folder_path)
        else:
            print("It was not possible to find file for best results: %s" % (filename_best_result_wPath) )
            print("")


        if saving_result is not None and saving_result_best is not None:
            print("")
            print("Unable to save all the results properly.")
            print("CONTENT:")
            print( saving_result )
            print("")
            print("BEST DETECTION:")
            print( saving_result_best )
            print("")
            #sys.exit(1)
            return 1 ## exit with errors or warnings


    else:
        print("")
        print("Unable to localize cropped files to classify.")
        print("PATH: %s" % crops_folder_path)
        print("")
        #sys.exit(1)
        return 1 ## exit with errors or warnings


    return 0 ## successful exit



#sys.exit(0)
