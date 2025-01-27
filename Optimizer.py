from __future__ import print_function

import sys
import os

import scipy
from scipy import ndimage as ndi
from scipy.spatial.qhull import Delaunay

new_paths = [
    u'../arrangement/',
    u'../Map-Alignment-2D'
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import json
import time
import numpy as np
import skimage.transform
import skimage._shared.utils as sksh
from matplotlib import pyplot as plt

# map alignment package
import map_alignment.map_alignment as mapali
import map_alignment.mapali_plotting as maplt

# nonrigid optimization of alignment package
import optimize_alignment.optimize_alignment as optali
import optimize_alignment.plotting as optplt

################################## STORAGE #####################################

def retrieve(relative_path_to_file):
    '''
    Returns requested json file as dictionary
    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.split(dirname)[0]
    file = os.path.join(parent, relative_path_to_file)
    with open(file) as f:
        data = json.load(f)
    return data

def store(data, relative_path_to_file, encodingType=None):
    '''
    Converts dictionary into JSON format and stores it in file
    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.split(dirname)[0]
    file = os.path.join(parent, relative_path_to_file)
    with open(file, "w") as f:
        json.dump(data, f, default=encodingType)

########################### RETRIEVE INPUT VALUES #############################

def get_maps():
    maps = retrieve('api/storage/maps/maps.json')
    src = maps[0]
    dst = maps[1]

    return src, dst

def get_alignment_matrix():
    m = retrieve('api/storage/alignment/output/matrix.json')["matrix"]
    tf_matrix = np.array(m)
    return tf_matrix

def get_parameters():
    min_dist = retrieve('api/storage/optimization/input/min_dist.json')
    max_corners = retrieve('api/storage/optimization/input/max_corners.json')
    return min_dist['min_dist'], max_corners['max_corners']

########################## INITIAL ALIGNMENT ################################

# def initialise_alignment_matrix(str):
#     tf_matrix = np.empty((3, 3))
#     arr = str.splitlines()

#     for i in range(len(arr)):
#         x = arr[i].strip().split()

#         for j in range(len(x)):
#             tf_matrix[i][j] = x[j]

#     return tf_matrix

# def initialise():
#     args = sys.argv

#     # fetching options from input arguments
#     # options are marked with single dash
#     options = []
#     for arg in args[1:]:
#         if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
#             options += [arg[1:]]
#     # fetching parameters from input arguments
#     # parameters are marked with double dash,
#     # the value of a parameter is the next argument
#     listiterator = args[1:].__iter__()
#     while 1:
#         try:
#             item = next( listiterator )
#             if item[:2] == '--':
#                 exec(item[2:] + ' = next( listiterator )')

#         except:
#             break

#     details = locals()
#     return details['img_src'], details['img_dst']


def initial_alignment():

    ############################ Alignment Configurations ##########################

    lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
                  'binary_threshold_2': [100, 255], # with cv2 - for trait detection
                  'traits_from_file': False, # else provide yaml file name
                  'trait_detection_source': 'binary_inverted',
                  'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
                  'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
                  'orthogonal_orientations': True} # for dominant orientation detection

    ################################# Map Alignment ################################

    ########## image loading, SKIZ, distance transform and trait detection
    # src_results, src_lnl_t = mapali._lock_n_load(img_src, lnl_config)
    # dst_results, dst_lnl_t = mapali._lock_n_load(img_dst, lnl_config)

    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, 'initial_alignment/build/data.txt')
    # mat = ''

    # with open(filename) as f: # load initial alignment matrix
    #     line = f.readline()
    #     while line:
    #         mat += line
    #         line = f.readline()

    # np.set_printoptions(suppress=True)
    # tform = initialise_alignment_matrix(mat)
    # tform_align = skimage.transform._geometric.ProjectiveTransform(matrix=tform)

    src, dst = get_maps()
    src_results, src_lnl_t = mapali._lock_n_load(src["path_to_file"], lnl_config)
    dst_results, dst_lnl_t = mapali._lock_n_load(dst["path_to_file"], lnl_config)

    np.set_printoptions(suppress=True)
    tform = get_alignment_matrix()
    tform_align = skimage.transform._geometric.ProjectiveTransform(matrix=tform)

    return src_results, dst_results, tform_align

############################## ENCODERS #################################

def encode_complex(z):
    '''
    Encoder from https://realpython.com/python-json/
    '''
    if isinstance(z, complex):
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

def encode_tf_obj(obj):
    if isinstance(obj, Delaunay):
        return obj.points.tolist()
    elif isinstance(obj, skimage.transform.AffineTransform):
        v = (obj.__dict__)['params']
        return {'params' : v.tolist()}
    else:
        type_name = obj.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

########################## OPTIMIZATION STEP ################################

def optimize(src_results, dst_results, tform_align):

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

    min_dist, max_corners = get_parameters()
    opt_config['min_distance'] = min_dist
    opt_config['max_corner'] = max_corners

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

    ################# data point correlation (for averaging motion) ################
    X_correlation = optali.data_point_correlation(X_aligned, correlation_sigma=opt_config['correlation_sigma'], normalize=True)

    ################################# Optimization #################################
    X_optimized, optimization_log = optali.optimize_alignment( X0=X_aligned, X_correlation=X_correlation,
                                                               gradient_map=grd_map, fitness_map=fit_map,
                                                               config=opt_config,
                                                               verbose=True)
    print ('total optimization time: {:.5f}'.format( time.time() - opt_tic ) )

    tform_opt = skimage.transform._geometric.PiecewiseAffineTransform()
    tform_opt.estimate(X_aligned, X_optimized)

    return X_aligned, X_optimized, grd_map, tform_opt

############################### VISUALIZING #####################################

def visualize(src_results, dst_results, tform_align, tform_opt, X_aligned, X_optimized, grd_map):

    '''
    Note on how to generate X_aligned and X_optimized from tforms:
    print ( np.allclose(X_aligned , tform_align(X_original))  )
    print ( np.allclose(X_aligned , tform_align._apply_mat(X_original, tform_align.params))  )
    print ( np.allclose(X_optimized , tform_opt(X_aligned)) )
    '''

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
                                            X_aligned, X_optimized)

########################### TRANSFORM COORDINATE ##############################

def get_transformed(req):
    '''
    Assume coords is a numpy array in the form [x, y].
    The input coords is in the frame of src.
    The output coords is in the frame of dst.
    '''
    coords = np.array([req['x'], req['y']])
    data = retrieve('api/storage/maps/maps.json')
    src = data[0]
    dst = data[1]

    translated = np.array([coords[0] + src["translation"]["x"], coords[1] + src["translation"]["y"], coords[2]])
    tform_align, tform_opt = main_without_store_and_visualize()
    aligned = tform_align._apply_mat(translated, tform_align.params)
    optimized = tform_opt.inverse(aligned)
    transformed = np.array([optimized[0] + dst["translation"]["x"],optimized[1] + dst["translation"]["y"], coords[2]])
    return transformed

##################################### MAIN #####################################

def main_without_store_and_visualize():
    src_results, dst_results, tform_align = initial_alignment()
    X_aligned, X_optimized, grd_map, tform_opt = optimize(src_results, dst_results, tform_align)
    return tform_align, tform_opt

def main_without_store():
    src_results, dst_results, tform_align = initial_alignment()
    X_aligned, X_optimized, grd_map, tform_opt = optimize(src_results, dst_results, tform_align)
    visualize(src_results, dst_results, tform_align, tform_opt, X_aligned, X_optimized, grd_map)

def main_without_visualize():
    src_results, dst_results, tform_align = initial_alignment()
    X_aligned, X_optimized, grd_map, tform_opt = optimize(src_results, dst_results, tform_align)
    store(grd_map.tolist(), 'api/storage/optimization/output/gmap.json', encodingType=encode_complex) # Potential problem: gmap.json: tokenization, wrapping and folding have been turned off for this large file in order to reduce memory usage and avoid freezing or crashing.
    store(tform_opt.__dict__, 'api/storage/optimization/output/tform_object.json', encodingType=encode_tf_obj)

def main():
    # img_src, img_dst = initialise()
    src_results, dst_results, tform_align = initial_alignment()
    X_aligned, X_optimized, grd_map, tform_opt = optimize(src_results, dst_results, tform_align)
    store(grd_map.tolist(), 'api/storage/optimization/output/gmap.json', encodingType=encode_complex) # Potential problem: gmap.json: tokenization, wrapping and folding have been turned off for this large file in order to reduce memory usage and avoid freezing or crashing.
    store(tform_opt.__dict__, 'api/storage/optimization/output/tform_object.json', encodingType=encode_tf_obj)
    visualize(src_results, dst_results, tform_align, tform_opt, X_aligned, X_optimized, grd_map)

################################################################################

if __name__ == '__main__':
    main()