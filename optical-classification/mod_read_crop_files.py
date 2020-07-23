"""
Created on 2019-09-15
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
from __future__ import division

import math

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import re #regular expresions

import pandas as pd

# functions
## lambda function to human readable sort (to avoid problems)
## ex: rectangle100 .. rectangle80 --> rectangle80 .. rectangle100
def natural_sort(ll):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(ll, key = alphanum_key)


def countFiles(path):
    count_imgs = 0

    for file_name in listdir( path ):
        if isfile( join( path, file_name ) ):
            count_imgs+=1

    total_files = count_imgs

    return total_files



def getSegmentsAsDataframe(crops_folder_path): #old name: listSegmetsIntoPandas

    # counter and variables
    nframe = 0  # it will be updated to 1 in the first loop
    vid_name_prev = ""

    df = pd.DataFrame( columns=('filename', 'frame', 'x1', 'y1', 'x2', 'y2', 'detection-acc') )


    if os.path.exists( crops_folder_path ):
        for file_name in natural_sort( listdir( crops_folder_path ) ):
            file_name_wpath = join( crops_folder_path, file_name )

            if isfile( file_name_wpath ):

                #mac files
                if file_name.startswith('.DS_Store'):
                    continue

                if not file_name.lower().endswith('jpg') and not file_name.lower().endswith('jpeg') \
                    and not file_name.lower().endswith('png') and not file_name.endswith('bmp') :
                    continue

                # espected format: IMG_20190513_153057rectangleX_X_X_X
                # espected format: IMG_20190513_153057yYYYYYYrectangleX_X_X_X
                ## VS
                # espected format: IMG_20190513_153057rectangleX_X_X_X_acc0.999
                # espected format: IMG_20190513_153057YYYYYYrectangleX_X_X_X_acc0.999
                ## VS
                # espected format: IMG_20190513_153057frame2rectangleX_X_X_X_acc0.999
                # espected format: IMG_20190513_153057YYYYYYframe2rectangleX_X_X_X_acc0.999

                #now format - 20191028
                #20191028_141559_909402
                #[IMG_20190807_103944.634547.PNG]

                filename_splitted = file_name.split("rectangle")
                vid_name = filename_splitted[0] #prevoiusly needed for frame number

                nframe = int(filename_splitted[0].split("frame")[-1])

                #MAL #frame_coords_info = filename_splitted[-1].split(".")[0].split("acc")[0] #X_X_X_X_acc.ext -> X_X_X_X_
                frame_coords_info = filename_splitted[-1].split("acc")[0] #[_]X_X_X_X[_]accZ.ZZ.ext -> [_]X_X_X_X[_]
                if frame_coords_info.startswith('_'): #just in case
                    frame_coords_info = frame_coords_info[1:]

                if frame_coords_info.endswith('_'): #for acc
                    frame_coords_info = frame_coords_info[1:]

                coords = frame_coords_info.split("_")
                coords = list(map(int, coords))

                acc_retina = filename_splitted[-1].split("acc")[-1] #Z.ZZ.ext
                #rfind: last occurrence
                acc_retina = float( acc_retina[:acc_retina.rfind(".")] ) #Z.ZZ


                #inserting new entry
                #FILENAME, FRAME-NUMBER, X1, Y1, X2, Y2, ACC
                df.loc[ 0 if pd.isnull( df.index.max() ) else df.index.max() + 1 ] = [file_name] + [int(nframe)] + list(coords) + [acc_retina]
                #[int(coords[0])] + [int(coords[1])] + [int(coords[2])] + [int(coords[3])]

        #for-end

        df.insert(0,'ii', df.index)

    return df
