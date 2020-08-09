import numpy as np
import pandas as pd
import glob
import time
import sys
import teste_utils as utils
import teste_models_timesteps as models

ss = time.time()

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

###############################################################################

# Set output directory
output_path = "/home/costa/radardata/results/track_forecasts/daily_minmax/"
# Set recorded tracks data directory
plk_dir = "/home/costa/radardata/tracked_corners/dmm_trshd/"
# Set data hdf5 directory
hdf5_dir = "/home/costa/radardata/Germany_mosaic/reanalisys_2017/"

# maximum distance between consecutive points (8km in 5min ~ 96km/h)
distance_threshold = 8
# Minimum aceptable mean value for a track rain depth 
rain_depth_min_threshold = 0.04
# minimum advection value in dense flow
min_flow_threshold = 0.05
# maximum advection value in dense flow
max_flow_threshold = 12
# OF farneback parameters
farneback_param = dict(pyr_scale=0.5,
                       levels=3,
                       winsize=15,
                       iterations=3,
                       poly_n=5,
                       poly_sigma=1.1,
                       flags=0)

# Set input RY data dimensions
row_dim, col_dim = 1100, 900
########################################################################################################################
# As OpenCV returns flipped (col,row), here we'll also flipp to col,row
tree = utils.kdtree_(col_dim, row_dim) # calc kdtree for find the nearest point

data_month_name = "ry_2016" + str(sys.argv[1])
data_month_name_next = "ry_2016" + str(sys.argv[2]) 

#data_month_name = "ry_201608"

# Get tracks file paths
plkfiles = [f for f in glob.glob(plk_dir + data_month_name + "*.plk")]
plkfiles_next = [f for f in glob.glob(plk_dir + data_month_name_next + "*.plk")]
plkfiles_next.sort()

plkfiles.append(plkfiles_next[0]) # add day 01 from next month
plkfiles.sort()

# Get hdf5 file paths
hdf5files = [f for f in glob.glob(hdf5_dir + data_month_name + "*.hdf5")]
hdf5files_next = [f for f in glob.glob(hdf5_dir + data_month_name_next + "*.hdf5")]
hdf5files_next.sort()

hdf5files.append(hdf5files_next[0]) # add day 01 from next month
hdf5files.sort()

t0_old = None

result_dist = []
result_error = []

# Run over daily files
for i in range(0,len(plkfiles)-1):
#for i in range(16,18):
    
    path_plk = plkfiles[i]

    p_name = path_plk.split("/")[-1].split(".")[0]
    h_name = hdf5files[i].split("/")[-1].split(".")[0]

    frame_mm_nextday = None
    frame_8b_nextday = None

    if p_name != h_name:
        print("Files doesn't match:")
        print(path_plk)
        print(hdf5files[i])
        exit()

    print(path_plk)

    s = time.time()

    # read track file
    df = pd.read_pickle(path_plk)
    df_next = pd.read_pickle(plkfiles[i + 1])

    # read hdf5 file and convert to array
    frame_mm = utils.read_hdf5_to_np(hdf5files[i])
    
    # Scale to 8bits
    frame_8b = utils.RYScaler(frame_mm, fixed_max_min = True)
    
    #del frame_dbz, frame_dbz_1day
    e = time.time()
    print("hdf5 and transfomation time :",e-s,"s")
    

    # Create a daily dictionary for results
    out_dic = {}

    print(len(df.index[df.index != 0]))
    
    # Loop over tracks IDs
    for idx in df.index[df.index != 0]:

        trk_matrix = None

        # test whether the track goes on the next day
        if idx not in df_next.index:
            # track values from dataframe
            trk0 = df.loc[idx].values[~pd.isnull(df.loc[idx].values)]

            # format tracks points to array
            trk_matrix = utils.format_array(trk0)
            
            # frames where the points are
            lt_matrix = np.where(df.loc[idx].notnull() == 1)[0]

        else:
            # Get track values [x,y,error] from dataframe
            trk0 = df.loc[idx].values[~pd.isnull(df.loc[idx].values)]
            trk1 = df_next.loc[idx].values[~pd.isnull(df_next.loc[idx].values)]  # Next day

            # Get frames values
            lt0 = np.where(df.loc[idx].notnull() == 1)[0]
            lt1 = np.where(df_next.loc[idx].notnull() == 1)[0]  # Next day

            # format tracks points to array
            trk_m = np.concatenate([trk0, trk1])
            trk_matrix = utils.format_array(trk_m)

            # frames values
            lt_matrix = np.concatenate([lt0, lt1])

            # Read the "next day" HDF5 file
            if frame_mm_nextday is None:
                frame_mm_nextday = utils.read_hdf5_to_np(hdf5files[i + 1])

            # Transform the "next day" HDF5 file to 8bits
            if frame_8b_nextday is None:
                frame_8b_nextday = utils.RYScaler(frame_mm_nextday[0:len(lt1)], 
                                                  fixed_max_min = True)
                #frame_8b_nextday = utils.mm_to_8bits_ne(frame_dbz_nextday[0:len(lt1)])

            elif len(lt1) > frame_8b_nextday.shape[0]:
                frame_8b_nextday = utils.RYScaler(frame_mm_nextday[0:len(lt1)], 
                                                  fixed_max_min = True)
                #frame_8b_nextday = utils.mm_to_8bits_ne(frame_dbz_nextday[0:len(lt1)])
        
        
        # Test whether the track goes beyond two times steps
        if (trk_matrix.shape[0] > 2) and np.isnan(trk_matrix[0, 2]):  # Tracks must start with nan as OF error
            
            rain_depth_val = utils.dbz_lister(np.round(trk_matrix,decimals=3)[:,0:2],
                                              lt_matrix,tree,frame_mm,
                                              frame_mm_nextday)

            
            ave_rain_depth = np.nanmean(rain_depth_val)
                
            # As the transformation uses daily maximum and minimum threshold, 
            # tracks that are fluctuations around zero and will be ignored.
            if np.isnan(rain_depth_val).any():
                continue
            if ave_rain_depth < rain_depth_min_threshold:
                continue
    
            # Create a neasted dic to store the results
            out_dic[idx] = {}
            out_dic[idx]['fname'] = p_name
            out_dic[idx]['frames'] = lt_matrix
            out_dic[idx]['track'] = np.round(trk_matrix,decimals=3)[:,0:2]
            out_dic[idx]['of_error'] = np.round(trk_matrix, decimals=3)[:, 2]
            
            out_dic[idx]['track_rain_val'] = rain_depth_val

            #s = time.time()
            track_dist, track_si, track_vel = utils.calc_track_properties(np.round(trk_matrix,decimals=3)[:,0:2])
            out_dic[idx]['track_dist'] = track_dist
            out_dic[idx]['track_si'] = track_si
            out_dic[idx]['track_vel'] = track_vel


            ###############################################
            #### Start lead time dependence forecast ######
            ###############################################
            next_day = False
            # loop up to point tn-2 , which there is still one prediction point
            #for lead_time in range(0,len(lt_matrix)-2):
            for lead_time in range(0,1):
                
                t_minus_1 = lt_matrix[lead_time]     # t-1
                t_0 = lt_matrix[lead_time + 1]       # t0 forecast time

                if t_0 > t_minus_1:
                    delta = utils.calc_DIS_flow(frame_8b[t_minus_1, :, :], frame_8b[t_0, :, :],
                                                swap_axes=False)
                #elif t_0 < t_minus_1:
                elif t_0 == 0: # Next day frame must be equal to zero
                    next_day = True
                    delta = utils.calc_DIS_flow(frame_8b[t_minus_1, :, :], frame_8b_nextday[t_0, :, :],
                                                swap_axes=False)
                elif next_day is True:
                    delta = utils.calc_DIS_flow(frame_8b_nextday[t_minus_1, :, :], frame_8b_nextday[t_0, :, :],
                                                swap_axes=False)
                else:
                    print(idx)
                    print("algo deu errado-->",t_minus_1, t_0)
                    break

                

                
                # linear extrapolation from LK Of points
                lin_points, lin_dist = models.linear_extrapolation(trk_matrix[:,:2],
                                                                   time_step=lead_time)

                #lin_dbz = utils.dbz_lister(np.round(lin_points, decimals=3),lt_matrix,tree,frame_dbz,frame_dbz_1day,
                #                           fixed_frame=True)

                # Save results
                lin_lk_dict_name = 'lin_lk_' + str(lead_time)
                lin_lk_dist_dict_name = 'lin_lk_dist_' + str(lead_time)
                out_dic[idx][lin_lk_dict_name] = np.round(lin_points, decimals=3)
                out_dic[idx][lin_lk_dist_dict_name] = np.round(lin_dist, decimals=3)
                #out_dic[idx]['lin_lk_dbz'] = lin_dbz
                
                
                ########################################################################################################
                # linear extrapolation from LK Of points looking back
                ########################################################################################################
                lin_points_lb, lin_dist_lb = models.lin_extrp_lookingback(trk_matrix[:,:2], tstep_looking_back=4)

                #lin_dbz = utils.dbz_lister(np.round(lin_points, decimals=3),lt_matrix,tree,frame_dbz,frame_dbz_1day,
                #                           fixed_frame=True)

                # Save results
                lin_lk_dict_name = 'lin_lookbk_' + str(lead_time)
                lin_lk_dist_dict_name = 'lin_lookbk_dist_' + str(lead_time)

                if type(lin_points_lb) is np.ndarray:
                    out_dic[idx][lin_lk_dict_name] = np.round(lin_points_lb, decimals=3)
                    out_dic[idx][lin_lk_dist_dict_name] = np.round(lin_dist_lb, decimals=3)


                else:
                    out_dic[idx][lin_lk_dict_name] = lin_points_lb
                    out_dic[idx][lin_lk_dist_dict_name] = lin_dist_lb

                lin_lk_dict_name = 'lin_lookbk_' + str(lead_time)
                lin_lk_dist_dict_name = 'lin_lookbk_dist_' + str(lead_time)


                ########################################################################################################                
                # linear extrapolation from Dense OF fields
                ########################################################################################################
                dense_lin_points, dense_lin_dist = models.dense_linear_extrapolation(track=trk_matrix[:,:2],
                                                                                     flow=delta, time_step=lead_time,
                                                                                     kdtree=tree)
                
                # Save results
                lin_dense_dict_name = 'lin_dense_' + str(lead_time)
                lin_dense_dist_dict_name = 'lin_dense_dist_' + str(lead_time)

                if type(dense_lin_points) is np.ndarray:
                    out_dic[idx][lin_dense_dict_name] = np.round(dense_lin_points, decimals=3)
                    out_dic[idx][lin_dense_dist_dict_name] = np.round(dense_lin_dist, decimals=3)

                    #lin_dense_dbz = utils.dbz_lister(np.round(dense_lin_points, decimals=3), lt_matrix, tree,
                    #                                 frame_dbz, frame_dbz_nextday, fixed_frame=True)

                    #out_dic[idx]['lin_dense_dbz'] = lin_dense_dbz
                else:
                    out_dic[idx][lin_dense_dict_name] = dense_lin_points
                    out_dic[idx][lin_dense_dist_dict_name] = dense_lin_dist
                    #out_dic[idx]['lin_dense_dbz'] = None


                ########################################################################################################
                # Dense rotation advection
                ########################################################################################################
                dense_rot_points, dense_rot_dist = models.dense_rotation(track=trk_matrix[:,:2], flow=delta,
                                                                         time_step=lead_time, kdtree=tree)
                

                # Save results
                rot_dense_dict_name = 'rot_dense_' + str(lead_time)
                rot_dense_dist_dict_name = 'rot_dense_dist_' + str(lead_time)

                if type(dense_rot_points) is np.ndarray:
                    out_dic[idx][rot_dense_dict_name] = np.round(dense_rot_points, decimals=3)
                    out_dic[idx][rot_dense_dist_dict_name] = np.round(dense_rot_dist, decimals=3)

                    # dense_rot_dbz = utils.dbz_lister(np.round(dense_rot_points, decimals=3), lt_matrix, tree,
                    #                                 frame_dbz, frame_dbz_nextday, fixed_frame=True)

                    # out_dic[idx]['rot_dense_dbz'] = dense_rot_dbz
                else:
                    out_dic[idx][rot_dense_dict_name] = dense_rot_points
                    out_dic[idx][rot_dense_dist_dict_name] = dense_rot_dist
                    # out_dic[idx]['rot_dense_dbz'] = None
                #e=time.time()
                #print("dense rot",e-s)
                
    output_name = output_path + p_name + "_filtered_dmm.npy"
    np.save(output_name, out_dic)
    

ee = time.time()
print("run time:", ee - ss)
exit()
