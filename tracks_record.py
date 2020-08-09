import sys
import cv2
import numpy as np
import glob
from scipy import spatial
import pandas as pd
import warnings
warnings.simplefilter('once', DeprecationWarning)

import utils

import datetime


# Set data directory - RY reanalisys_2017 data 
datadir = ""
# Set output files path - tracks dataframe folder
df_output_path = ""


rain_depth_min_trsh = 0.04

# Get files paths
hdf5files = [f for f in glob.glob(datadir + "ry_2016*.hdf5")]
# Sort by files names
hdf5files.sort()

# FEATURE DETECTION: Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,  # maximum number of corners to return
                       qualityLevel = 0.2,# minimal accepted quality of corners
                       minDistance = 7,   # minimal Euclidean distance between corners
                       blockSize = 21 )   # size of pixel neighborhood for covariance calculation

# FEATURE TRACKING: Parameters for Lucas Kanade (lk) Optical Flow technique
lk_params_ = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))

########################################################################################################################


# set max_id to zero
max_id = 0
old_pt = []
old_ids = []
start = None # Start must be None for first loop

# Run over daily RY files

TT = 0
for i in range(0,len(hdf5files)-1):
    run_time_start = datetime.datetime.now()
    
    frame = 1 # Set frame to 1 as first frame

    print(i, hdf5files[i])
    print("now = ",hdf5files[i][-16:-5],"next =", hdf5files[i+1][-16:-5])

    
    # Read HDF5 files and convert to a numpy array
    frame_RY = utils.read_hdf5_to_np(hdf5files[i])        # current "i" day
    frame_RY_1day = utils.read_hdf5_to_np(hdf5files[i+1]) # one day after
    
    # Scale rain deep to 8bits int
    frame_8b = utils.RYScaler(frame_RY, 
                              fixed_max_min=True)  # current "i" day
    frame_8b_1day = utils.RYScaler(frame_RY_1day[0], 
                                   fixed_max_min= True)# one day after
    
    run_time_stop = datetime.datetime.now()
    print(frame_RY.shape,frame_8b.shape)
    print('IO Time: ', run_time_stop - run_time_start)
    
    # create a pandas dataframe for each day , columns names will be frames positions
    fr_id = list(range(1, 289))
    df = pd.DataFrame(columns=fr_id)
    df.loc[0, 1] = np.nan # Start the dataframe ID 0 with entrys egual to NaN

    # Run over 5min frames (288 frames per day)
    for ii in range(0,288):

        # test if is the first frame of the day, but not the first frame of this run
        if ii == 0 and start != None:
            # Saving features on pandas dataframe for those tracks that keep going on next day
            if len(old_ids) != 0:
                # run over Old IDs to save them
                for fn in range(0, len(old_ids)):
                    rain_value = frame_RY[ii,
                                          int(old_pt[fn,0,1]),
                                          int(old_pt[fn,0,0])]
                    
                    df.loc[old_ids[fn], frame] = [old_pt[fn, 0, 0], old_pt[fn, 0, 1], 
                                                  old_err[fn, 0], rain_value]

        # find our points of interest - corners
        pts = cv2.goodFeaturesToTrack(frame_8b[ii, :, :], **feature_params)
        test_goods_points = []
        
        for point in range(0,len(pts)):
            # rain depht value
            # for some reason cv2 inverts the axes (col,row) to (row,col).
            rain_depth_YX = frame_RY[ii,int(pts[point,0,1]),int(pts[point,0,0])]
            
            if ~np.isnan(rain_depth_YX) and (rain_depth_YX >= rain_depth_min_trsh):
                test_goods_points.append(point)
        
        # Corners great than rain_depth_min_trsh
        p0 = pts[test_goods_points].copy()
        
        
        # test special cases of None result
        if p0 is None and len(old_pt) != 0:
            p0 = old_pt
        elif p0 is None and len(old_pt) == 0:
            frame = frame + 1
            continue

        # looking for new points
        if len(old_pt) != 0:

            # Calc distance between new corners and old corners to avoid redundant tracks
            dist_new_old = spatial.distance_matrix(p0[:,0,:], old_pt[:,0,:])

            # look for corners where the distance between them is greater than 7, those are new corners
            mask = np.sum(dist_new_old <= 7., axis=1) == 0
            p0 = np.concatenate([old_pt, p0[mask, :]])

            # Empty out old points list
            old_pt = []
            
        # If there are corners to track
        if len(p0) != 0: 

            # define new track ids
            dif = abs(len(p0) - len(old_ids))

            # aggregates the "old" and new indexes
            if len(old_ids) != 0:
                ids = np.concatenate([old_ids, np.arange(max_id + 1,max_id + dif + 1)])

            else:
                ids = np.arange(max_id+1,max_id + len(p0) + 1)


            # track corners: Lukas-Kanade optical flow algorithm
            if ii != 287:
                new_corners, st, err = utils.track(frame_8b[ii, :, :], frame_8b[ii + 1, :, :],
                                                   p0, lk_params_, fb_threshold=1)

                start = 0
            else: # last frame, next day frame
                new_corners, st, err = utils.track(frame_8b[ii, :, :], frame_8b_1day,
                                                   p0, lk_params_, fb_threshold=1)

            success = st.ravel() == 1

            # Tracked features
            ids_s = ids[success]
            p0_s = p0[success]
            new_corners_s = new_corners[success]
            err_s = err[success]

            # Saving features in pandas dataframe
            
            if len(ids_s) != 0:
                for fn in range(0,len(ids_s)):                
                    # write the first entry a new of corner
                    if not ids_s[fn] in df.index:
                        # rain depth
                        rain_value = frame_RY[ii,
                                              int(p0_s[fn,0,1]),
                                              int(p0_s[fn,0,0])]
                        
                        df.loc[ids_s[fn], frame] = [p0_s[fn, 0, 0],
                                                    p0_s[fn, 0, 1],
                                                    np.nan,
                                                    rain_value]

                    # write the track
                    if ii != 287:
                        # rain depth
                        rain_value = frame_RY[ii+1, 
                                              int(new_corners_s[fn,0,1]),
                                              int(new_corners_s[fn,0,0])]
                        
                        df.loc[ids_s[fn],frame+1] = [new_corners_s[fn, 0, 0],
                                                     new_corners_s[fn, 0, 1],
                                                     err_s[fn, 0], rain_value]

            # new corners will be old in the next loop
            old_ids = ids_s
            old_pt = new_corners_s
            old_err = err_s

            # carrying the max ID during the days
            if np.max(df.index.values.tolist()) != 0:
                max_id = np.max(df.index.values.tolist())
            #print(max_id)

        # next frame
        frame = frame + 1
    run_time_stop2 = datetime.datetime.now()
    print('Day run time: ', run_time_stop2 - run_time_start)

    df_name = df_output_path + hdf5files[i][-16:-4] + ".plk"
    df.to_pickle(df_name)

