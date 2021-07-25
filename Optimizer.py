from __future__ import print_function

import sys
import os

import scipy
from scipy import ndimage as ndi
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

def get_maps():
    maps = retrieve('api/storage/maps/maps.json')
    src = maps[0]
    dst = maps[1]

    return src, dst

# def initialise_alignment_matrix(str):
#     tf_matrix = np.empty((3, 3))
#     arr = str.splitlines()

#     for i in range(len(arr)):
#         x = arr[i].strip().split()

#         for j in range(len(x)):
#             tf_matrix[i][j] = x[j]

#     return tf_matrix

def get_alignment_matrix():
    m = retrieve('api/storage/alignment/output/matrix.json')["matrix"]
    tf_matrix = np.array(m)
    return tf_matrix

def get_parameters():
    min_dist = retrieve('api/storage/optimization/input/min_dist.json')
    max_corners = retrieve('api/storage/optimization/input/max_corners.json')
    return min_dist['min_dist'], max_corners['max_corners']

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

def encode_complex(z):
    '''
    Encoder from https://realpython.com/python-json/
    '''
    if isinstance(z, complex):
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

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

    # store gradient map
    store(grd_map.tolist(), 'api/storage/optimization/output/gmap.json', encodingType=encode_complex)

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

    # TODO: store tf matrix

    return X_aligned, X_optimized, grd_map, tform_opt

def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            ("Unknown mode: '{}', or cannot translate mode. The "
             "mode should be one of 'constant', 'edge', 'symmetric', "
             "'reflect', or 'wrap'. See the documentation of numpy.pad for"
             "more info.").format(mode))
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    mode = grid_modes.get(mode, mode)
    return mode

def mapper(image, inverse_map, output_shape, mode, cval):
    order = 0 if image.dtype == bool else 1

    if order > 0:
        image = sksh.convert_to_float(image, True)
        if image.dtype == np.float16:
            image = image.astype(np.float32)

    input_shape = np.array(image.shape)
    output_shape = sksh.safe_as_int(output_shape)

    warped = None

    def coord_map(*args):
        np.set_printoptions(threshold=sys.maxsize)
        return inverse_map(*args)

    if len(input_shape) == 3 and len(output_shape) == 2:

        output_shape = (output_shape[0], output_shape[1], input_shape[2])

    coords = skimage.transform.warp_coords(coord_map, output_shape)

    prefilter = order > 1
    ndi_mode = _to_ndimage_mode(mode)

    warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                 mode=ndi_mode, order=order, cval=cval)

    skimage.transform._warps._clip_warp_output(image, warped, order, mode, cval, True)

    return warped

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
    # src_img_optimized = skimage.transform.warp(src_img_aligned, tform_opt.inverse,
    #                                             output_shape=(dst_results['image'].shape),
    #                                             preserve_range=True, mode='constant', cval=127)
    src_img_optimized = mapper(src_img_aligned, tform_opt.inverse, (dst_results['image']).shape, 'constant', 127)
    ########## save/plotting alignment, motion of points and optimized alignment
    optplt.plot_alignment_motion_optimized(dst_results['image'],
                                            src_img_aligned, src_img_optimized, grd_map,
                                            X_aligned, X_optimized)

    # TODO: calculate final matrix
    # TODO: store final matrix

def main():
    # img_src, img_dst = initialise()
    src_results, dst_results, tform_align = initial_alignment()
    X_aligned, X_optimized, grd_map, tform_opt = optimize(src_results, dst_results, tform_align)
    visualize(src_results, dst_results, tform_align, tform_opt, X_aligned, X_optimized, grd_map)

################################################################################

if __name__ == '__main__':
    '''
    Parameters and Options:
    -----------------------
    --img_src 'the-address-name-to-sensor-map'
    --img_dst 'the-address-name-to-layout-map'

    Examples:
    ---------
    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png'
    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png'
    '''

    main()
