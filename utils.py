import cv2
import numpy as np
import numexpr as ne
import pandas as pd
from scipy import spatial
import h5py
import matplotlib.pyplot as plt

def calc_dense_flow(prvs_img,next_img,farneback_param,swap_axes=False):

    delta = cv2.calcOpticalFlowFarneback(prvs_img, next_img, None,
                                         **farneback_param)

    # Actual velocity field
    velocity = ne.evaluate("sqrt(xv ** 2 + yv ** 2)",
                           {'xv': delta[..., 0], 'yv': delta[..., 1]})
    
    delta[ne.evaluate("velocity < 0.05")] = np.nan
    delta[ne.evaluate("velocity > 12.")] = np.nan
    
    if swap_axes:
        delta = np.swapaxes(delta, 0, 1)
    
    return delta

def calc_DIS_flow(prvs_img,next_img,swap_axes=False):
    #inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM) # cv2 3.4
    inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) # cv2 4.1
    delta = inst.calc(prvs_img, next_img, None)

    # Actual velocity field
    velocity = ne.evaluate("sqrt(xv ** 2 + yv ** 2)",
                           {'xv': delta[..., 0], 'yv': delta[..., 1]})
    
    delta[ne.evaluate("velocity < 0.05")] = np.nan
    delta[ne.evaluate("velocity > 12.")] = np.nan

    if swap_axes:
        # Swap axe 0 for 1
        delta = np.swapaxes(delta, 0, 1)
    return delta

def two_days_DIS_flow(frame_8b,frame_8b_1day):

    dense_field_day1 = np.empty([frame_8b.shape[0],frame_8b.shape[1],frame_8b.shape[2],2])
    dense_field_day1.fill(np.nan)

    # Saving 8bits converted data

    for frame in range(0,frame_8b.shape[0]):

        if frame == frame_8b.shape[0]-1: # its take last frame from current day and the first frame from the next day
            dense_field_day1[frame, ...] = calc_DIS_flow(frame_8b[frame, :, :], frame_8b_1day[0, :, :], swap_axes=False)

        else:
            dense_field_day1[frame, ...] = calc_DIS_flow(frame_8b[frame, :, :], frame_8b[frame+1, :, :], swap_axes=False)

    dense_field_day2 = np.empty([frame_8b.shape[0],frame_8b.shape[1],frame_8b.shape[2],2])
    dense_field_day2.fill(np.nan)

    for frame in range(0,frame_8b_1day.shape[0]-1):
        dense_field_day1[frame,...] = calc_DIS_flow(frame_8b_1day[frame, :, :], frame_8b_1day[frame+1, :, :], swap_axes=False)

    return dense_field_day1,dense_field_day2


def read_hdf5_to_np(hdf5_file_path):
    """
    read HDF5 files
    :param hdf5_file_path: string of hdf5 file path
    :return: numpy array
    """
    f = h5py.File(hdf5_file_path, 'r')

    dataset = np.array(f.get('data'))
    return dataset

def depth2intensity(depth, interval=300):
    """
    Function for convertion rainfall depth (in mm) to
    rainfall intensity (mm/h)
    Args:
        depth: float
        float or array of float
        rainfall depth (mm)
        interval : number
        time interval (in sec) which is correspondend to depth values
    Returns:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
    """
    return ne.evaluate("depth * 3600 / interval")

def intensity2depth(intensity, interval=300):
    """
    Function for convertion rainfall intensity (mm/h) to
    rainfall depth (in mm)
    Args:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
        interval : number
        time interval (in sec) which is correspondend to depth values
    Returns:
        depth: float
        float or array of float
        rainfall depth (mm)
    """
    return ne.evaluate("intensity * interval / 3600")

def RYScaler(X_mm, fixed_max_min = True):
    '''
    .. from rainymotion.utils.RYScaler function

    Scale RY data from mm (in float64) to brightness (in uint8).
    Args:
        X (numpy.ndarray): RY radar image
        float: c1, dBZ minimum value, scaling coefficient
        float: c2, dBZ maximum value, scaling coefficient
    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255 for corresponding input rainfall intensity

    '''
    def mmh2rfl(r, a=256., b=1.42):
        '''
        .. based on wradlib.zr.r2z function
        .. r --> z
        '''
        return ne.evaluate("a * r ** b")

    def rfl2dbz(z):
        '''
        .. based on wradlib.trafo.decibel function
        .. z --> d
        '''
        #return 10. * np.log10(z)
        return ne.evaluate("10. * log10(z) ")

    # mm to mm/h
    X_mmh = depth2intensity(X_mm)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[ne.evaluate("X_rfl == 0")] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[ne.evaluate("X_dbz < 0")] = 0
    
    if not fixed_max_min:
        # MinMaxScaling
        dbz_min = X_dbz.min()
        dbz_max = X_dbz.max()
        
        if dbz_min >= 0:
            c1 = dbz_min
        else:
            c1 = 0.
            
        if dbz_max <= 54.:
            c2 = dbz_max
        else:
            c2 = 54.
    
    else: # Fixed values for max and min
        c1 = 0.
        c2 = 54.
        
    return (ne.evaluate("(X_dbz - c1) / (c2 - c1) * 255")).astype(np.uint8)


def kdtree_(xd,yd):
    """

    :param xd: X dimension
    :param yd: Y dimension
    :return: Kdtree object
    """
    x, y = np.mgrid[0:xd, 0:yd]
    tree = spatial.KDTree(np.c_[x.ravel(), y.ravel()])
    return tree


def format_array(a):
    a_rows = a.shape[0]
    a_cols = len(a[0])
    new_ar = np.empty(shape=(a_rows, a_cols), dtype=(float))

    for row in range(0, a_rows):
        new_ar[row, :] = np.array(a[row])
    return new_ar


def dist_points(a, b):
    """
    Euclidean distance between two points or arrays.

    :param a: array like
    :param b: array like
    :return: dist
    """
    d_ab = a-b

    if len(d_ab.shape) > 1:
        dist = np.linalg.norm(d_ab, axis=1)
    else:
        dist = np.linalg.norm(d_ab)

    return dist

def plot_flow_points(flow,point,img = None):
    ii, jj = np.mgrid[0:1100, 0:900]
    print(ii.shape,jj.shape)
    f, ax = plt.subplots(figsize=(12, 12))
    if type(img) is np.ndarray:

        ax.imshow(img, origin="lower")
    ax.quiver(jj, ii, flow[jj, ii, 0], flow[jj, ii, 1], scale_units='xy', scale=1, color="black")

    ax.plot(point[:,0],point[:,1],linestyle='-', marker='o', color='blue')
    ax.grid()
    # plt.xlim(225, 250)
    # plt.ylim(925, 960)

    return ax

def dbz_lister(track,frames, kdtree, dbz_data, dbz_data_next_day, fixed_frame = False):

    dd, ii = kdtree.query(track, k=1)  # nearest neighbors from forecast time point

    frame_zero_position = np.where(frames == 0)[0]

    dbz_values = []

    # Fixed time frame to get the dbz values
    if fixed_frame:
        for f, ftime in enumerate(frames[2:]): # Start from 3th time step, i.e. 1st lead time
            dbz_values.append(dbz_data[frames[1],
                                       kdtree.data[ii][f][1], 
                                       kdtree.data[ii][f][0]])

    # If the goes over a day
    elif (len(frame_zero_position) > 1):
        for f, ftime in enumerate(frames):
            if f < frame_zero_position[1]:
                dbz_values.append(dbz_data[ftime, kdtree.data[ii][f][1], kdtree.data[ii][f][0]])
            else:
                dbz_values.append(dbz_data_next_day[ftime, kdtree.data[ii][f][1], kdtree.data[ii][f][0]])

    # If the goes over a day
    elif (len(frame_zero_position) != 0 ) and (frame_zero_position != 0):
        for f, ftime in enumerate(frames):
            if f < frame_zero_position:
                dbz_values.append(dbz_data[ftime, kdtree.data[ii][f][1], kdtree.data[ii][f][0]])
            else:
                dbz_values.append(dbz_data_next_day[ftime, kdtree.data[ii][f][1], kdtree.data[ii][f][0]])
        
    else:
        for f, ftime in enumerate(frames):
            dbz_values.append(dbz_data[ftime, kdtree.data[ii][f][1], kdtree.data[ii][f][0]])

    return dbz_values

def calc_track_properties(track):
    dist_bt_pts = dist_points(track[:-1, :], track[1:, :])
    dist_total = np.sum(dist_bt_pts)
    dist_ideal = dist_points(track[0], track[-1])

    # sinuosity_index
    si = dist_total/dist_ideal

    # average velocity for 5min time step
    vel = (dist_total * 60.) / ((len(track)-1) * 5.)

    return np.round(dist_total,decimals=3), np.round(si,decimals=3),np.round(vel,decimals=3)

def track(im0, im1, p0, lk_params_, fb_threshold=-1):
    """
    Main tracking method using sparse optical flow (LK)
    im0: previous image in gray scale
    im1: next image
    lk_params: Lukas Kanade params dict
    fb_threshold: minimum acceptable backtracking distance
    """
    if p0 is None or not len(p0):
        return np.array([])

    # Forward flow
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **lk_params_)

    if fb_threshold > 0:
        # Backward flow
        p0r, st0, err0 = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **lk_params_)

        p0r[st0 == 0] = np.nan

        # Set only good tracks
        fb_good = np.fabs(np.linalg.norm(p0r[:,0,:] - p0[:,0,:], axis=1)) < fb_threshold


        p1[~fb_good] = np.nan
        st1[~fb_good] = 0
        err1[~fb_good] = np.nan

    return p1, st1, err1
