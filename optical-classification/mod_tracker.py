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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

from sort import *

import pandas as pd


# functions

def getAreaFromCoords(coords):
    return abs(coords[2] - coords[0]) * abs(coords[3] - coords[1])


def getCenterFromCoords(coords):
    return np.array([ (float(coords[2] + coords[0]))/2, (float(coords[3] + coords[1]))/2 ])


def getCoordsMatchingMetric(orig_coords, tracker_coords):  #not in use
    #to MINIMIZE, the minimun value will be to right one.
    #Known issue: negative coordinates.
    #To impruve, square pow for areas partial result to comparable to centers partial results (or remove square root).

    # arg min |A - A'| + ||pos - pos'||
    #         (area)      (centros)
    #         z - z'      (x,y) - (x', y')

    area = getAreaFromCoords(orig_coords)      #orig_coords <--> row[:-1]
    center = getCenterFromCoords(orig_coords)  #orig_coords <--> row[:-1]

    area_d = getAreaFromCoords(tracker_coords)       #tracker_coords <--> d[:-1]
    center_d = getCenterFromCoords(tracker_coords)   #tracker_coords <--> d[:-1]

    return abs(area_d - area) + math.sqrt( math.pow( (center[0] - center_d[0]), 2) + math.pow( (center[1] - center_d[1]), 2) )


def getTargetsId(df): #old name: getTargetsIdFromPandas

    total_time = 0.0
    total_frames = 0

    # if (debug)
    #     colours = np.random.rand(64,3) #for debuging
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
        #             ######## MODIFYYY
        #     #fn = 'xxxxxxxxxxxxxxxxxxxx/%s/frames/%06d.jpg'%(folder_name,frame)
        #     fn = 'data/dorado/frames/%s-frame%s.png'%(folder_name, frame)
        #     im =io.imread(fn)
        #     ax1.imshow(im)
        #     #plt.title(seq+' Tracked Targets')
        #     plt.title('Frame' + str(frame) + ' << __________ >>' ) ##can be changed if needed later in the code
        #
        # if(debug):
        #     start_time = time.time()



        ######## MODIFYYY

        #trackers = artifact_tracker.update( (dets[:,:-1]).astype(float) ) #not working #'x1', 'y1', 'x2', 'y2', 'detection-acc', 'ii'
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
        #     ######## MODIFYYY
        #     name_fig = 'data/dorado/frames_wbbox/%s-frame%s.png'%(folder_name, frame)
        #     fig.savefig( name_fig )
        #     ax1.cla()



    #print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))


    return df_ii_id
