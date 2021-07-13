from __future__ import print_function

import sys
import os

import scipy
new_paths = [
    u'../arrangement/',
    u'../Map-Alignment-2D',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import json
import time
import numpy as np
import skimage.transform
from matplotlib import pyplot as plt

# map alignment package
import map_alignment.map_alignment as mapali
import map_alignment.mapali_plotting as maplt

# nonrigid optimization of alignment package
import optimize_alignment.optimize_alignment as optali
import optimize_alignment.plotting as optplt

def retrieve(relative_path_to_file):
    '''
    Returns requested json file as dictionary
    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dirname, relative_path_to_file)
    with open(file) as f:
        data = json.load(f)
    return data

def store(data, relative_path_to_file):
    '''
    Converts dictionary into JSON format and stores it in file
    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dirname, relative_path_to_file)
    with open(file, "w") as f:
        json.dump(data, f)

def get_maps():
    maps = retrieve('api/storage/maps/maps.json')
    src = maps[0]
    dst = maps[1]

    return src, dst

def initalise_alignment_matrix(str):
    tf_matrix = np.empty((3, 3))
    arr = str.splitlines()

    for i in range(len(arr)):
        x = arr[i].strip().split()

        for j in range(len(x)):
            tf_matrix[i][j] = x[j]

    return tf_matrix

def get_alignment_matrix():
    m = retrieve('api/storage/alignment/output/matrix.json')["matrix"]
    tf_matrix = np.array(m)
    return tf_matrix

def get_parameters():
    params = retrieve('api/storage/optimization/input/parameters.json')
    min_dist = params["min_dist"]
    max_corners = params["max_corners"]
    return min_dist, max_corners

def inital_alignment():
    visualize = False
    return

################################################################################

if __name__ == '__main__':
    '''
    Parameters and Options:
    -----------------------
    --img_src 'the-address-name-to-sensor-map'
    --img_dst 'the-address-name-to-layout-map'
    -visualize
    -multiprocessing
    Examples:
    ---------
    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png' --hyp_sel_metric 'fitness' -visualize -multiprocessing
    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png' --hyp_sel_metric 'matchscore' -visualize -save_to_file -multiprocessing
    '''

    ########################### INITIALIZATION from CLI ############################
    args = sys.argv

    # fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]
    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break

    # setting defaults values for parameter
    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False
    multiprocessing = True if 'multiprocessing' in options else False

    ########################## FIRST STAGE - MAP ALIGNMENT #########################

    ############################ Alignment Configurations ##########################

    ########## lock and load
    lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
                  'binary_threshold_2': [100, 255], # with cv2 - for trait detection
                  'traits_from_file': False, # else provide yaml file name
                  'trait_detection_source': 'binary_inverted',
                  'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
                  'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
                  'orthogonal_orientations': True} # for dominant orientation detection

    ################################# Map Alignment ################################

    ########## image loading, SKIZ, distance transform and trait detection
    src_results, src_lnl_t = mapali._lock_n_load(img_src, lnl_config)
    dst_results, dst_lnl_t = mapali._lock_n_load(img_dst, lnl_config)

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'initial_alignment/build/data.txt')
    mat = ''

    with open(filename) as f: # load initial alignment matrix
        line = f.readline()
        while line:
            mat += line
            line = f.readline()

    np.set_printoptions(suppress=True)
    tform = initalise_alignment_matrix(mat)
    tform_align = skimage.transform._geometric.ProjectiveTransform(matrix=tform)

    # src, dst = get_maps()
    # src_results, src_lnl_t = mapali._lock_n_load(src["path_to_file"], lnl_config)
    # dst_results, dst_lnl_t = mapali._lock_n_load(dst["path_to_file"], lnl_config)

    # np.set_printoptions(suppress=True)
    # tform = get_alignment_matrix()
    # tform_align = skimage.transform._geometric.ProjectiveTransform(matrix=tform)

    ################### SECOND STAGE - OPTIMIZATION OF ALIGNMENT ###################
    opt_config = {
        # dst
        'fitness_sigma': 50,
        'gradient_ksize': 3,
        'correlation_sigma': 800,

        # src - good feature to track
        'edge_refine_dilate_itr': 5, #3
        'max_corner': 500,
        'quality_level': .01,
        'min_distance': 25,

        # optimization - loop
        'opt_rate': 10**1,          # optimization rate
        'max_itr': 10000,           # maximum number of iterations
        'tol_fit': .9999, #.99        # break if (fitness > tol_fit)
    }
    opt_config['tol_mot'] = 0.001 # * opt_config['opt_rate'] # break if (max_motion < tol_mot)

    # min_dist, max_corners = get_parameters()
    # opt_config['min_distance'] = min_dist
    # opt_config['max_corner'] = max_corners

    ############# POINT SAMPLING occupied cells (of the source image) ##############
    opt_tic = time.time()
    X_original = optali.get_corner_sample( src_results['image'],
                                           edge_refine_dilate_itr=opt_config['edge_refine_dilate_itr'],
                                           maxCorners=opt_config['max_corner'],
                                           qualityLevel=opt_config['quality_level'],
                                           minDistance=opt_config['min_distance'])
    X_aligned = tform_align._apply_mat( X_original, tform_align.params )

    ############# construction of MOTION FIELD (of the destination map) ############
    fit_map, grd_map = optali.get_fitness_gradient( dst_results['image'],
                                                    fitness_sigma=opt_config['fitness_sigma'],
                                                    grd_k_size=opt_config['gradient_ksize'],
                                                    normalize=True)

    # TODO: store gradient map

    ################# data point correlation (for averaging motion) ################
    X_correlation = optali.data_point_correlation(X_aligned, correlation_sigma=opt_config['correlation_sigma'], normalize=True)

    ################################# Optimization #################################
    X_optimized, optimization_log = optali.optimize_alignment( X0=X_aligned, X_correlation=X_correlation,
                                                               gradient_map=grd_map, fitness_map=fit_map,
                                                               config=opt_config,
                                                               verbose=True)
    print ('total optimization time: {:.5f}'.format( time.time() - opt_tic ) )

    tform_opt = skimage.transform.PiecewiseAffineTransform()
    tform_opt.estimate(X_aligned, X_optimized)

    # TODO: store tf matrix

    '''
    Note on how to generate X_aligned and X_optimized from tforms:
    print ( np.allclose(X_aligned , tform_align(X_original))  )
    print ( np.allclose(X_aligned , tform_align._apply_mat(X_original, tform_align.params))  )
    print ( np.allclose(X_optimized , tform_opt(X_aligned)) )
    '''

    if visualize or save_to_file:
        ###################### warpings of the source image ######################
        # warp source image according to alignment tform_align
        src_img_aligned = skimage.transform.warp(src_results['image'], tform_align.inverse, #_inv_matrix,
                                                 output_shape=(dst_results['image'].shape),
                                                 preserve_range=True, mode='constant', cval=127)

        # warp aligned source image according to optimization
        src_img_optimized = skimage.transform.warp(src_img_aligned, tform_opt.inverse,
                                                   output_shape=(dst_results['image'].shape),
                                                   preserve_range=True, mode='constant', cval=127)


        ########## save/plotting alignment, motion of points and optimized alignment
        optplt.plot_alignment_motion_optimized(dst_results['image'],
                                               src_img_aligned, src_img_optimized, grd_map,
                                               X_aligned, X_optimized, save_to_file)

    # TODO: calculate final matrix
    # TODO: store final matrix