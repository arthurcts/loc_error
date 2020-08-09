import cv2
import numpy as np
import numexpr as ne
from scipy import spatial
from utils import dist_points, kdtree_

def lin_extrp_lookingback(track,tstep_looking_back):
    """
    Simple linear advection from t0 to tn
    :param track: array like of tracks [x,y] points
    :time_step:
    :tstep_looking_back : Integer - time steps looking back
    :return:
    """
    ###
    #   observation ~  Forecast
    #    (t-1)--> (t0) ~~> (t+1) ~~> (t+2) ~~> ...
    #               ~
    ###
    
    # Minimum track length to retrieve a prediction
    min_track_len = tstep_looking_back + 1
    
    if min_track_len > len(track):
        return None,None
        
        
    # create an array of N times Forecast time t0,where N is the maximum lead 
    # time possible
    t0 = np.tile(track[tstep_looking_back], (track.shape[0] - tstep_looking_back - 1, 1))
    
    # Array of lead time number
    lead_time = np.c_[1:track.shape[0] - tstep_looking_back, 
                      1:track.shape[0] - tstep_looking_back]

    # Compute the vector between t0 and t-n (tstep_looking_back)
    dij = track[tstep_looking_back] - track[0]
    # delta vector to extrapolate
    delta_v = dij / tstep_looking_back
    # Repeated deltas array
    deltas = np.tile(delta_v, (track.shape[0] - tstep_looking_back -1, 1))
    
    # Extrapolation points
    frc_points = t0 + deltas * lead_time
    
    
    frc_dist = dist_points(frc_points, track[tstep_looking_back + 1:, :]) # from t+1 to end

    return frc_points,frc_dist


def linear_extrapolation(track, time_step):
    """
    Simple linear advection from t-1 to t to n
    :param track:
    :return:
    """
    t0 = np.tile(track[time_step + 1], (track.shape[0] - 2 - time_step, 1))
    ###
    #   observation ¦  Forecast
    #    (t-1)--> (t0) ~~> (t+1) ~~> (t+2) ~~> ...
    #               ¦
    ###

    # array of lead times
    lead_time = np.c_[1:track.shape[0] - 1 - time_step, 1:track.shape[0] - 1 - time_step]

    # Compute the delta between t0 and t-1
    d = track[time_step + 1] - track[time_step]
    # Repeated deltas array
    deltas = np.tile(d, (track.shape[0] - 2 - time_step, 1))

    frc_points = t0 + deltas * lead_time

    frc_dist = dist_points(frc_points, track[time_step + 2:, :]) # from t+1 to end

    return frc_points,frc_dist

def dense_linear_extrapolation(track, flow, time_step, kdtree, distance_threshold = 8.):

    dd, ii = kdtree.query(track[time_step+1],k=1) # nearest neighbor from t0 point

    # The distance to the nearest point must be less than 2x the diagonal of the cube.
    if dd > 3.46:
        return None, None

    # Finds the delta value at the nearest point from t0
    dxy = flow[kdtree.data[ii][1],kdtree.data[ii][0],:]

    if np.isnan(dxy[0]):
        return None, None # frc_points, frc_dist

    if dist_points(track[time_step+1],track[time_step+1] + dxy) >= distance_threshold:
        return None, None

    # t0 array
    t0 = np.tile(track[time_step+1], (track.shape[0] - 2 - time_step, 1))
    
    # Make an array of lead times
    lead_time = np.c_[1:track.shape[0] - 1 - time_step, 1:track.shape[0] - 1 - time_step]

    deltas = np.tile(dxy, (track.shape[0] - 2 - time_step, 1))
    
    frc_points = t0 + deltas * lead_time

    frc_dist = dist_points(frc_points, track[time_step + 2 :, :]) # from t+1 to end

    return frc_points, frc_dist


def dense_rotation(track, flow, time_step, kdtree, distance_threshold = 8.):

    last_point = track[time_step + 1]  # First point t0

    dd, ii = kdtree.query(last_point,k=1) # nearest neighbors from t0

    # The distance to the nearest point must be less than 2x the diagonal of the cube.
    if dd > 3.46:
        return None, None

    frc_points = []

    for i in range(0,track.shape[0] - 2 - time_step): # from t+1 to t+n
        # Delta at this point
        try:
            dxy = flow[kdtree.data[ii][1], kdtree.data[ii][0], :]
        except:
            print("last point",last_point)
            print(dd,ii)
            print(kdtree.data[ii][1], kdtree.data[ii][0])
            return None, None

        if np.isnan(dxy[0]):
            return None, None  # frc_points, frc_dist

        next_point = last_point + dxy

        if dist_points(next_point,last_point) >= distance_threshold:
            return None, None

        frc_points.append(next_point)

        dd, ii = kdtree.query(next_point, k=1)  # nearest neighbor; distance , index

        # The distance to the nearest point must be less than 2x the diagonal of the cube.
        if dd > 3.46:
            return None, None

        last_point = next_point
    # Track points
    frc_points = np.array(frc_points)
    # distance btw track and frc points
    frc_dist = dist_points(frc_points, track[time_step + 2:, :])  # from t+1 to end

    return frc_points, frc_dist

def dense_linear_extrapolation_npoints(track, flow, kdtree,npoints = 9, flow_threshold=0.05,distance_threshold = 8.):

    dd, ii = kdtree.query(track[1],k=npoints) # nearest neighbor from forecast time point

    dd = dd[dd < 3.46] # 2 times cube diagonal
    ii = ii[dd < 3.46]

    dxy_ave = np.nanmean(flow[kdtree.data[ii][:,0],kdtree.data[ii][:,1]], axis = 0)

    if np.isnan(dxy_ave[0]):
        return None, None  # frc_points, frc_dist

    #velocity = np.sqrt(dxy_ave[..., 0] ** 2 + dxy_ave[..., 1] ** 2)

    if (np.abs(dxy_ave)[0] <= flow_threshold) and (np.abs(dxy_ave)[1] <= flow_threshold): # Set minimum threshold
        return None,None # frc_points, frc_dist


    if dist_points(track[1],track[1] + dxy_ave) >= distance_threshold:
        return None, None

    forecast_time = np.tile(track[1], (track.shape[0] - 2, 1))
    lead_time = np.c_[1:track.shape[0] - 1, 1:track.shape[0] - 1]
    deltas = np.tile(dxy_ave, ((track.shape[0] - 2, 1)))
    frc_points = forecast_time + deltas * lead_time

    frc_dist = dist_points(frc_points, track[2:, :]) # from t+1 to end

    return frc_points, frc_dist

def dense_rotation_npoints(track, flow, kdtree, npoints = 9, flow_threshold=0.05,distance_threshold = 8.):

    last_point = track[1]  # Forecast point
    dd, ii = kdtree.query(track[1],k=npoints) # nearest neighbor from forecast time point
    dd = dd[dd < 3.46] # 2 times cube diagonal
    ii = ii[dd < 3.46]

    frc_points = []
    for i in range(0,track.shape[0] - 2): # from t+1 to t+n

        # Delta D in this point
        dxy = np.nanmean(flow[kdtree.data[ii][:,0],kdtree.data[ii][:,1]], axis = 0)

        # Set minimum threshold
        if (i == 0) and (np.abs(dxy)[0] <= flow_threshold and np.abs(dxy)[1] <= flow_threshold):
            return None, None

        next_point = last_point + dxy

        if np.isnan(next_point[0]):
            return None, None  # frc_points, frc_dist

        if dist_points(next_point,last_point) >= distance_threshold:
            return None, None

        frc_points.append(next_point)

        dd, ii = kdtree.query(track[1], k=npoints)  # nearest neighbor from forecast time point
        dd = dd[dd < 3.46]  # 2 times cube diagonal
        ii = ii[dd < 3.46]
        last_point = next_point

    frc_points = np.array(frc_points)
    frc_dist = dist_points(frc_points, track[2:, :])  # from t+1 to end

    return frc_points , frc_dist
