"""
Created on 2019-09-15
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


# config. and vars.

input_folder = "./classification/" # the folder will contain the local results and other files from the species classification per each camera in subfolders
###output_folder = "./classification/output/" # the folder will contain the final results and other files for the species classification
output_folder = "output/" # the folder will contain the final results and other files for the species classification
input_csv_file = "camera_classification.csv"
## CSV content: ......
input_path_to_crops_file = "path_crops_folder.txt"

output_best_file = "best_classification.txt"
output_boundary_file = "boundary_targets.txt"

#recalculate values -  boundary areas
boundary_area_left = int( ( 2048 * 10.5 ) / 70.5 ) #305pixels == 10.5degrees of overlapping
boundary_area_right = 4096 - boundary_area_left


#functions

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
            if isfile( join( camera_folder_wpath, input_csv_file) ):
                cameras_element_present = True
                break

    return cameras_element_present


def getPathToCropsPerCamera(camera_name):

    path_crops_folder_file = join(working_in_folder_path, camera_name, input_path_to_crops_file)

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
            output_file = open( '%s/%s.txt' % (working_out_folder_path, species_name), 'w') ##join()
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
                if isfile( join( camera_folder_wpath, input_csv_file) ):

                    df_cam = pd.read_csv( join( camera_folder_wpath, input_csv_file), header='infer' )
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

            saving_result_best = saveResults( df_bestDetection[ list_columns_final_order ], working_out_folder_path, output_best_file)

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

                path_for_filename = getPathToCropsPerCamera(camera_name)
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


###### function-end ########



# read parameters
parser = argparse.ArgumentParser(description='Classification in central Jetson')

parser.add_argument("--cameras", dest='cameras_element', help="Parameter to provide the camera element will be processed: top, bottom. Default: guess", default='guess') #optinal

parser.add_argument("--d", help="Parameter to provide where the local results (cameras) are. Default: ./classification/", default='') #optinal

parser.add_argument('--debug', dest='debug', help='Use for debuging [False by default].',action='store_true')  #optinal

args = parser.parse_args()
cameras_element = args.cameras_element
path_d = args.d
debug = args.debug

if path_d != '':
    input_folder = path_d
    if not input_folder.endswith('/'): #just in case
        input_folder += '/'
    output_folder = input_folder + output_folder


# cameras subfolder name format:
# ./classification/cameraXX.Y
# ./classification/camera20.6
# ./classification/camera40.2

# paths
working_in_folder_path = os.path.abspath( input_folder )
working_out_folder_path = os.path.abspath( output_folder )




# main

print("Starting classification...")
t_start_new = datetime.datetime.now()
t_start_str_new = "[ %s ]" % t_start_new.strftime("%Y/%m/%d - %H:%M")
print( t_start_str_new )


#checking output folder
if not os.path.exists( working_out_folder_path ):

	print("Creating dir: " + str( working_out_folder_path ) )
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
            else:
                 print('Someting went wrong')
        else:
            print("Input data not found [%s]." % (cameras_element) )
            print("")
    else:
        if checkCamerasElement( 'top' ):
            print('Processing %s camera elements' % ('top') )
            if processCamerasElement( 'top' ):
                print('Successfull')
            else:
                 print('Someting went wrong')

        elif checkCamerasElement( 'bottom' ):
            print('Processing %s camera elements' % ('bottom'))
            if processCamerasElement( 'bottom' ):
                print('Successfull')
            else:
                 print('Someting went wrong')
        else:
            print('There is no data to process.')

else:
    #print("")
    print('Cameras element unknown.')
    print("'--cameras' parameter was not properly provided.")
    print('')



sys.exit(0)
