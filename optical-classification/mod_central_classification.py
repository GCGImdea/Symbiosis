"""
Created on 2019-12-20
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
#imports
from __future__ import division
import math

#from mod_read_crop_files import *
#from mod_tracker import *
#from mod_bin_classifier import *
from mod_multiclass_classifier import *

import os
from os import listdir
from os.path import isfile, join
import shutil #for moving files

import numpy as np
import pandas as pd
import argparse
import sys

###### USE ########
# import:
#    from mod_central_classification import *
#
# Main function (examples):
#   centralClassification()
#   centralClassification('guess')
#   centralClassification('guess', './classification/', True)
#   centralClassification(input_path='./classification/')
#
# optional parameters
#   cameras_element: to provide the camera element to be processed: top, bottom. Default: guess
#   input_path:      to provide where the local results (cameras) are. Default: ./classification/
#   debug:           use for debugging [False by default].



# config. and vars.

input_classification_folder = "./classification/" # the folder will contain the local results and other files from the species classification per each camera in subfolders
###output_classification_folder = "./classification/output/" # the folder will contain the final results and other files for the species classification
output_classification_folder = "output/" # the folder will contain the final results and other files for the species classification
input_classification_csv_file = "camera_classification.csv"
## CSV content: ......
input_classification_path_to_crops_file = "path_crops_folder.txt"

output_classification_best_file = "best_classification.txt"
output_detection_best_file = "best_detection.txt"
#output_classification_boundary_file = "boundary_targets.txt"

#recalculate values -  boundary areas
boundary_area_left = int( ( 2048 * 10.5 ) / 70.5 ) #305pixels == 10.5degrees of overlapping
boundary_area_right = 2048 - boundary_area_left

working_in_folder_path = None
working_out_folder_path = None


###### functions ########

def getTimestampFromFilename(strr):

    strr_sub = strr.split('frame')[0]

    if strr_sub.startswith('IMG_'): #just in case
        strr_sub = strr_sub[4:]

    if strr_sub.endswith('-'): #just in case
        strr_sub = strr_sub[:-1]

    if strr_sub.endswith('_'): #just in case
        strr_sub = strr_sub[:-1]

    return strr_sub


def checkCamerasElementParameter(element):   #change name
    cam_elements = {
        "top": True,
        "bottom": True,
        "guess": True,
        "all": False
    }
    return cam_elements.get(element, False)


def getIPsubnetByCamerasElement(element):
    cam_elements = {
        "top": 20,
        "bottom": 40
    }
    return cam_elements.get(element, -1)


def getSpeciesNameByNumber(element):
    cam_elements = {
        0: "albacore",
        1: 'amberjack',
        2: 'atl_mackerel',
        3: 'dorado',
        4: 'med_mackerel',
        5: 'swordfish',
        6: 'others'
    }
    return cam_elements.get(element, -1)


#check
def getCameraFolderName(element, ncam):

    subnet_name = getIPsubnetByCamerasElement(element)

    camera_folder = ''

    if subnet_name is not -1:
        #if subnet_name and (ncam > 0 and ncam < 7):
        if subnet_name and (ncam > 0 and ncam < 256):
            #./cameraXX.Y/
            camera_folder = "camera%d.%d/" % ( subnet_name, ncam )

    return camera_folder



def getCameraNamesFolderList(cameras_element):

    folder_list = []
    if checkCamerasElementParameter(cameras_element):
        for ii in range(1,7): #for the number of cameras in each level (cameras_element)
            camera_folder = getCameraFolderName( cameras_element, ii )
            if camera_folder is not '':
                folder_list.append( camera_folder )


    return folder_list


def checkCamerasElement(cameras_element):   #change name

    cameras_element_present = False

    for camera_folder in getCameraNamesFolderList(cameras_element):

        camera_folder_wpath = join( working_in_folder_path, camera_folder)

        if os.path.isdir( camera_folder_wpath ):
            if isfile( join( camera_folder_wpath, input_classification_csv_file) ):
                cameras_element_present = True
                break

    return cameras_element_present


def getPathToCropsPerCameraFromFile(camera_name):

    path_crops_folder_file = join(working_in_folder_path, camera_name, input_classification_path_to_crops_file)

    path_crops_folder = open( path_crops_folder_file ).read().replace('\n','').replace('\r','')

    return path_crops_folder


def getFilenameWithPath(strr, prefix):  #call: df['acb'].apply( getFilenameWithPath, prefix='/path/to' )

    return join(prefix, strr)



def getNumberOfDifferentSpecies(df):  #count different fishes
    #species_list_ids = df['artifact-multi-decision'].unique()

    df_speciesid_count = pd.DataFrame( columns=('species', 'count') )

    if len(df):
        #for species_index in( species_list_ids ):
        for ii, species_index in enumerate( df['artifact-multi-decision'].unique() ): #sorted() <-- without enumeration

            df_specie = df[ df['artifact-multi-decision'] == species_index ]
            number_fishes_per_specie = df_specie['artifact-id'].nunique()

            df_speciesid_count.loc[ 0 if pd.isnull( df_speciesid_count.index.max() ) else df_speciesid_count.index.max() + 1 ] = [int(species_index)] + [int(number_fishes_per_specie)]

    return df_speciesid_count #df: 'species', 'count'. ex.: 2->24, 3->11,...


def saveResults(df, folder_path, output_file):
    return df.to_csv( join( folder_path, output_file) , index=False, header=False)    #,encoding='utf-8'


def saveNumberOfFishesPerSpecies(df_list, folder_path):

    for index, row in df_list.iterrows(): ## the column 'species' in not accesible, but it is as indexes

        ##species_index = row['species']
        species_index = index
        species_name = getSpeciesNameByNumber( species_index )
        number_fishes = row['count']

        output_file = None
        status = None
        try:
            #output_file = open( '%s/%s.txt' % (working_out_folder_path, species_name), 'w') ##join()
            output_file = open( '%s/%s.txt' % (folder_path, species_name), 'w') ##join()
            output_file.write('%d\r\n' % (int(number_fishes) ) )
            output_file.close()

            if status is None and not status:
                status = True

        except IOError:
            status = False
        finally:
            if output_file is not None:
                output_file.close() #just in case


    return status


def getArtifactInBoundaryBasic(df):

    artifacts_in_boundary = df[ ( df['x1'] <= boundary_area_left ) | ( df['x2'] >= boundary_area_right ) ]

    return artifact_in_boundary.reset_index(level=0, drop=True).copy()



def getArtifactInBoundary(df):

    list_columns = np.array(['filename','x1','y1','x2','y2','artifact-multi-decision','ncam'])
    ##list_columns_final_order = np.array(['ncam','timestamp','x1','y1','x2','y2','species_name','artifact-multi-decision'])
    list_columns_final_order = np.array(['ncam','filename','x1','y1','x2','y2','species_name','artifact-multi-decision'])


    artifact_in_boundary = df[ ( df['x1'] <= boundary_area_left ) | ( df['x2'] >= boundary_area_right ) ]

    artifact_in_boundary = artifact_in_boundary[list_columns].reset_index(level=0, drop=True).copy()
    # add column with species names
    artifact_in_boundary['species_name'] = artifact_in_boundary['artifact-multi-decision'].apply( getSpeciesNameByNumber )

    # add column with timestamp from filenames
    #artifact_in_boundary['timestamp'] = artifact_in_boundary['filename'].apply( getTimestampFromFilename )


    #delete column "filename" - not needed
    #artifact_in_boundary.drop("filename", axis=1, inplace=True)


    return artifact_in_boundary[ list_columns_final_order ]




def processCamerasElement(cameras_element):

    if checkCamerasElementParameter(cameras_element):

        df = pd.DataFrame() # columns=('filename', 'frame', 'x1', 'y1', 'x2', 'y2', 'detection-acc',)
        #filename, frame, x1, y1, x2, y2, detection-acc, bin-prob-neg, bin-prob-pos, multi-prob-1, multi-prob-2, multi-prob-3, multi-prob-4, multi-prob-5, multi-prob-6, artifact-bin-prob-pos, artifact-bin-prob-neg, artifact-bin-decision, artifact-multi-prob-1, artifact-multi-prob-2, artifact-multi-prob-3, artifact-multi-prob-4, artifact-multi-prob-5, artifact-multi-prob-6, artifact-multi-decision
        df_wFishesPerSpecie = pd.DataFrame()


        folder_counter = 0
        for camera_folder in getCameraNamesFolderList(cameras_element):

            camera_folder_wpath = join( working_in_folder_path, camera_folder)

            ncam = camera_folder.split('.')[-1]
            ncam = int( ncam[:-1] ) #remove last '/'

            if os.path.isdir( camera_folder_wpath ):

                # ncam = camera_folder.split('.')[-1]
                # ncam = int( ncam[:-1] ) #remove last '/'
                if isfile( join( camera_folder_wpath, input_classification_csv_file) ):

                    df_cam = pd.read_csv( join( camera_folder_wpath, input_classification_csv_file), header='infer' )
                    #df_cam['ncam'] = np.array([ ncam ] * len(df) )
                    df_cam['ncam'] = ncam

                    ##df_cam.reset_index(level=0, drop=True, inplace=True) #not here, after concat

                    ##
                    df_wFishesPerSpecieAndCam = getNumberOfDifferentSpecies( df_cam )  #df: 'species', 'count'. ex.: 2->24, 3->11,..
                    if len(df_wFishesPerSpecieAndCam):
                        df_wFishesPerSpecieAndCam['ncam'] = ncam
                        df_wFishesPerSpecie = pd.concat([df_wFishesPerSpecie, df_wFishesPerSpecieAndCam], axis = 0)
                        df_wFishesPerSpecie.reset_index(level=0, drop=True, inplace=True)   #df: 'species', 'count'. ex.: 2->24, 3->11,..2->3, 3->....


                    if len(df):
                        df = pd.concat([df, df_cam], axis = 0)
                        df.reset_index(level=0, drop=True, inplace=True)
                    else:
                        df = df_cam.copy()

                    del df_cam

                    folder_counter += 1

                else:

                    print('CSV from camera %d not Found [%s].' % (ncam, cameras_element) )
                ##ifp-end-csv-camera-exists

            else:
                print('CSV from camera %d not Found [%s].' % (ncam, cameras_element) )
            ## if-end-isdir


        ## for-end-cameras-read-csv


        if len(df): #or  ## len(df) ## folder_counter > 0
            #NUMBER OF FISHES FROM ALL CAMERAS
            # group per species
            if len(df_wFishesPerSpecie):
                df_wFishesPerSpecieNoCam = df_wFishesPerSpecie.copy()
                df_wFishesPerSpecieNoCam.drop("ncam", axis=1, inplace=True)
                number_fishes_per_specie = df_wFishesPerSpecieNoCam.groupby(['species']).sum()[['count']] ## , as_index=False
                #problem: groupby('species') is removing 'count' column.
                #number_fishes_per_specie = df_wFishesPerSpecie.groupby('species').sum()[['count']] ## , as_index=False
                #df: 'species', 'count'. ex.: 2->24, 3->11,..2->3, 3->....



                #save one file per species with the number.
                saving_result_per_species = saveNumberOfFishesPerSpecies( number_fishes_per_specie, working_out_folder_path )
                ## df: 'species', 'count' (totals)

            else:
                print('Dataframe with number of fishes per species in empty. Nothing to save.')
                print('')

                # saveResults( pd.DataFrame() , working_out_folder_path, 'nodata_species.txt')



            #BEST DETECTION

            #function define in 'mod_multiclass_classifier.py'
            #df_bestDetection = getBestArtifact(df) #filename, acc, species
            df_bestDetection = getBestArtifactFromSegmentation(df) #filename, acc, species

            df_bestDetection['species-name'] = df_bestDetection['species'].apply( getSpeciesNameByNumber )

            list_columns_final_order = np.array(['filename','acc','species-name','species'])

            #saving_result_best = saveResults( df_bestDetection[ list_columns_final_order ], working_out_folder_path, output_classification_best_file)
            saving_result_best = saveResults( df_bestDetection[ list_columns_final_order ], working_out_folder_path, output_detection_best_file)

            #copy best file
            filename_best_result = df_bestDetection.head(1)['filename'].to_numpy()[0]

            camera_best_result = df[ df['filename'] == filename_best_result ].head(1)['ncam']
            camera_folder_best_result = getCameraFolderName( cameras_element, camera_best_result.to_numpy()[0] )


            filename_best_result_wPath = join(working_in_folder_path ,camera_folder_best_result, filename_best_result)

            if isfile( filename_best_result_wPath):
                shutil.copy( filename_best_result_wPath , working_out_folder_path)
            else:
                print("It was not possible to find file for best results: %s" % (filename_best_result_wPath) )
                print("")



            #BOUNDARY AREAS
            artifacts_in_boundary = getArtifactInBoundary( df ) #camera, timestamp, coords, species_name, species_index

            #save boundary results
            saving_boundary_result = ""  ##VS None and +=

            for ii, cam_index in enumerate( df['ncam'].unique() ):

                camera_name = getCameraFolderName(cameras_element, cam_index)
                #saving_boundary_result +=
                artifacts_in_boundary_per_cam = artifacts_in_boundary[ artifacts_in_boundary['ncam'] == cam_index ].reset_index(level=0, drop=True).copy()

                path_for_filename = getPathToCropsPerCameraFromFile(camera_name)
                artifacts_in_boundary_per_cam['filename'] = artifacts_in_boundary_per_cam['filename'].apply( getFilenameWithPath, prefix=path_for_filename )

                saveResults( artifacts_in_boundary_per_cam, working_out_folder_path , "%s_boundary.txt" % camera_name[:-1] )

                #saveResults( artifacts_in_boundary[ artifacts_in_boundary['ncam'] == cam_index ], working_out_folder_path , "%s_boundary.txt" % camera_name[:-1] )
                #FILE: cameraXX.Y_boundary.txt
                #FORMAT: camera, timestamp, coords, species, species_index (each detection one line)

            ## for-end

            statusCameraElement = True
        else:
            print('Input CSVs are empty. Nothing to process.')
            print('')

            saveResults( pd.DataFrame() , working_out_folder_path, 'nodata.txt')
            statusCameraElement = False
        ## if-end-len-df

    else: ## checkCamerasElementParameter-check-allowed-elements
        print('Cameras element unknown.')
        print("'--cameras' parameter was not properly provided.")
        print('')

        statusCameraElement = False

    return statusCameraElement


def cleanFilesInFolder(folder_path):

    for file_name in listdir( folder_path ):
        file_name_wPath = join( folder_path, file_name )
        if isfile( file_name_wPath ):
            os.remove( file_name_wPath )


###### functions-end ########



###### main-function ########

def centralClassification(cameras_element='guess', input_path='./classification/', debug=False):
    # call:
    # centralClassification(), centralClassification('guess'), centralClassification('guess', './classification/', True),  centralClassification(input_path='./classification/')
    #optional parameters
    #cameras_element: Parameter to provide the camera element to be processed: top, bottom. Default: guess
    #input_path: Parameter to provide where the local results (cameras) are. Default: ./classification/
    #debug:  Use for debugging [False by default].

    global output_classification_folder
    global working_in_folder_path
    global working_out_folder_path

    if input_path != '':
        input_classification_folder = input_path
        if not input_classification_folder.endswith('/'): #just in case
            input_classification_folder += '/'
        output_classification_folder = input_classification_folder + output_classification_folder


    # cameras subfolder name format:
    # ./classification/cameraXX.Y
    # ./classification/camera20.6
    # ./classification/camera40.2

    # paths
    working_in_folder_path = os.path.abspath( input_classification_folder )
    working_out_folder_path = os.path.abspath( output_classification_folder )


    # main

    print( "Starting classification..." )
    t_start_new = datetime.datetime.now()
    t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
    print( t_start_str_new )

    print( "Input folder: %s\n" % (working_in_folder_path) )


    #checking output folder
    if not os.path.exists( working_out_folder_path ):

    	print("Creating output folder: " + str( working_out_folder_path ) )
    	print("")
    	os.makedirs( working_out_folder_path )


    if checkCamerasElementParameter(cameras_element):

        #clean previous results
        cleanFilesInFolder( working_out_folder_path )

        if cameras_element == 'top' or cameras_element == 'bottom':
            if checkCamerasElement( cameras_element ):
                print('Processing %s camera elements' % (cameras_element) )
                if processCamerasElement( cameras_element ):
                    print('Successfull')
                    return 0 ## successful exit
                else:
                    print('Someting went wrong')
                    return 1 ## exit with errors or warnings
            else:
                print("Input data not found [%s]." % (cameras_element) )
                print("")
                return 1 ## exit with errors or warnings
        else:
            if checkCamerasElement( 'top' ):
                print('Processing %s camera elements' % ('top') )
                if processCamerasElement( 'top' ):
                    print('Successfull')
                    return 0 ## successful exit
                else:
                    print('Someting went wrong')
                    return 1 ## exit with errors or warnings

            elif checkCamerasElement( 'bottom' ):
                print('Processing %s camera elements' % ('bottom'))
                if processCamerasElement( 'bottom' ):
                    print('Successfull')
                    return 0 ## successful exit
                else:
                    print('Someting went wrong')
                    return 1 ## exit with errors or warnings

            else:
                print('There is no data to process.')
                return 1 ## exit with errors or warnings


    else:
        #print("")
        print('Cameras element unknown.')
        print("'cameras_element' parameter was not properly provided.")
        print('')

        return 1 ## exit with errors or warnings


###### main-function-end ########




##sys.exit(0)
