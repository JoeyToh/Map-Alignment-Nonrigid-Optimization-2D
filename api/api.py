from flask import Flask, json, jsonify, make_response
from flask.globals import request
import os

app = Flask(__name__)
app.config["DEBUG"] = True

################################### DATA ######################################

# maps = [
#     {
# 	    'map-number' : 1,
#       'path-to-file': '/home/hopermf/Desktop/intern_joey/Map-Alignment-Nonrigid-Optimization-2D/map_sample/magni_chart_5cm.jpeg',
#       '_id' : ObjectId(\"5f1012ace22b8136334824e2\"),
#       'fleet_name' : \"fleet_a\",
# 	    'floor_name' : \"L1\",
#         'translation' : {
#             'x' : 39.85,
#             'y' : -39.85,
#             'z' : 0.0
#         },
# 	    'rotation' : 0.0,
# 	    'resolution' : 0.05,
# 	    'axis' : {
#             'x' : 1.0,
#             'y' : 1.0,
#             'z' : 1.0
# 	    },
#         'createdAt' : \"ISODate(2021-02-25T04:26:23.148Z)\",
# 	    'updatedAt' : \"ISODate(2021-03-12T08:19:40.830Z)\"
#     },
#     {
#       'map-number' : 2,
#       'path_to_file' : "/home/hopermf/Desktop/intern_joey/Map-Alignment-Nonrigid-Optimization-2D/map_sample/mir_5cm.jpeg",
# 		'_id' : ObjectId(\"6c178ace22b8136334824e2\"),
# 		'fleet_name' : \"fleet_b\",
# 		'floor_name' : \"L1\",
# 		'translation' : {
#     			'x' : 34.85,
#     			'y' : -20.85,
#     			'z' : 0.0
# 		},
# 		'rotation' : 0.0,
# 		'resolution' : 0.05,
# 		'axis' : {
#     			'x' : 1.0,
#     			'y' : 1.0,
#     			'z' : 1.0
# 		},
# 		'createdAt' : \"ISODate(2021-02-25T04:26:23.148Z)\",
# 		'updatedAt' : \"ISODate(2021-03-12T08:19:40.830Z)\"
# 	}
# ]

# points = [
#     {
#         "map-number" : 1,
#         "points" : ["(1, 2)", "(3, 4)", "(5, 6)"]
#     },
#     {
#         "map-number" : 2,
#         "points" : ["(7, 8)", "(2, 7)", "(9, 2.1)"]
#     }
# ]
################################# STORAGE #####################################

def store(data, relative_path_to_file):
    '''
    Converts dictionary into JSON format and stores it in file
    '''
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, relative_path_to_file)

    with open(file, "w") as f:
        json.dump(data, f)

def retrieve(relative_path_to_file):
    '''
    Returns requested json file as dictionary
    '''
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, relative_path_to_file)
    with open(file) as f:
        data = json.load(f)
    return data

################################## ERROR ######################################

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': error}), 404)

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': error}), 400)

############################### RESPONSE ######################################

def response(data, code=200):
    return jsonify(data), code

################################# MAPS ########################################

def get_maps():
    maps = retrieve('storage/maps/maps.json')
    return response(maps)

def store_maps(maps):
    '''
    Input example:
    [
        {
	        'map-number' : 1,
            'path_to_file' : "/home/hopermf/Desktop/intern_joey/Map-Alignment-Nonrigid-Optimization-2D/map_sample/magni_chart_5cm.jpeg",
            '_id' : ObjectId("5f1012ace22b8136334824e2"),
            'fleet_name' : "fleet_a",
	        'floor_name' : "L1",
            'translation' : {
               'x' : 39.85,
               'y' : -39.85,
               'z' : 0.0
           },
	        'rotation' : 0.0,
	        'resolution' : 0.05,
	        'axis' : {
               'x' : 1.0,
               'y' : 1.0,
               'z' : 1.0
	        },
           'createdAt' : "ISODate(2021-02-25T04:26:23.148Z)",
	        'updatedAt' : "ISODate(2021-03-12T08:19:40.830Z)"
       },
       {
            'map-number' : 2,
            'path_to_file' : "/home/hopermf/Desktop/intern_joey/Map-Alignment-Nonrigid-Optimization-2D/map_sample/mir_5cm.jpeg",
	    	'_id' : ObjectId("6c178ace22b8136334824e2"),
	    	'fleet_name' : "fleet_b",
	    	'floor_name' : "L1",
	    	'translation' : {
    			'x' : 34.85,
       			'y' : -20.85,
       			'z' : 0.0
	    	},
	    	'rotation' : 0.0,
	    	'resolution' : 0.05,
	    	'axis' : {
    			'x' : 1.0,
       			'y' : 1.0,
       			'z' : 1.0
	    	},
	    	'createdAt' : "ISODate(2021-02-25T04:26:23.148Z)",
	    	'updatedAt' : "ISODate(2021-03-12T08:19:40.830Z)"
	    }
    ]

    '''
    store(maps, 'storage/maps/maps.json')
    updated_maps = retrieve('storage/maps/maps.json')
    return response(updated_maps, 201)

@app.route('/api/v1/data/maps', methods=['POST', 'GET'])
def update_maps():
    if request.method == 'POST':
        return store_maps(request.json)
    else:
        return get_maps()

################################# POINTS ######################################

def get_points():
    points = retrieve('storage/alignment/input/points/points.json')
    return response(points)

def delete_point(point):
    '''
    Input example:
    {
        "map-number" : 1,
        "point" : "(1, 2)"
    }
    '''
    current_pts = retrieve('storage/alignment/input/points/points.json')
    map_no = point["map-number"]
    point_to_delete = point["point"]
    new_pts = []

    for i in current_pts:
        if i["map-number"] == map_no:
            pts = i["points"]
            idx = pts.index(point_to_delete)
            del pts[idx]

            if map_no == 1:
                other_pts = current_pts[1]["points"]
                del other_pts[idx]

                map_1 = {
                    "map-number" : 1,
                    "points" : pts
                }
                map_2 = {
                    "map-number" : 2,
                    "points" : other_pts
                }

                new_pts.extend([map_1, map_2])

            else:
                other_pts = current_pts[0]["points"]
                del other_pts[idx]

                map_1 = {
                    "map-number" : 1,
                    "points" : other_pts
                }
                map_2 = {
                    "map-number" : 2,
                    "points" : pts
                }

                new_pts.extend([map_1, map_2])

            break

    store(new_pts, 'storage/alignment/input/points/points.json')
    return response(new_pts)

def add_points(points):
    '''
    Input example:
    [
        {
            "map-number" : 1,
            "points" : ["(0.1, 0.22)", "(3.45, 1.34)", "(2.5, 64)"]
        },
        {
            "map-number" : 2,
            "points" : ["(5.7, 28)", "(0.2, 0.7)", "(0.9, 21)"]
     }
    ]
    '''
    current_pts = retrieve('storage/alignment/input/points/points.json')
    map_1_old = current_pts[0]
    map_2_old = current_pts[1]
    map_1_to_add = points[0]
    map_2_to_add = points[1]

    assert(map_1_old["map-number"] == map_1_to_add["map-number"]) # ensure that points are added to the correct map
    assert(map_2_old["map-number"] == map_2_to_add["map-number"]) # ensure that points are added to the correct map
    assert(len(map_1_to_add["points"]) == len(map_2_to_add["points"])) # ensure that same number of points are added for each map

    points_new = []
    map_1_new = {
        "map-number" : 1,
        "points" : map_1_old["points"].extend(map_1_to_add["points"])
    }
    map_2_new = {
        "map-number" : 2,
        "points" : map_2_old["points"].extend(map_2_to_add["points"])
    }
    points_new.extend(map_1_new, map_2_new)
    store(points_new, 'storage/alignment/input/points/points.json')

    assert(len(map_1_new["points"]) == len(map_2_new["points"])) # ensure that the total number of points are the same for each map

    return response(points_new)

@app.route('/api/v1/data/alignment/input/points', methods=['PATCH', 'DELETE', 'GET'])
def update_points():
    if request.method == 'PATCH':
        return add_points(request.json)
    elif request.method == 'DELETE':
        return delete_point(request.json)
    else:
        return get_points()

############################ ALIGNMENT PARAMS #################################

def get_alignment_parameters():
    params = retrieve('storage/alignment/input/parameters.json')
    return response(params)

def update_alignment_parameters(params):
    '''
    Input example:
    {
        "param-1": 50,
        "param-2": 0.2
    }
    '''
    store(params, 'storage/alignment/input/parameters.json')
    updated_params = retrieve('storage/alignment/input/parameters.json')
    return response(updated_params)

@app.route('/api/v1/data/alignment/input/parameters', methods=['PUT', 'GET'])
def alignment_parameters():
    if request.method == 'PUT':
        update_alignment_parameters()
    else:
        get_alignment_parameters()

############################# ALIGNMENT MATRIX ################################

@app.route('/api/v1/data/alignment/output/matrix', methods=['GET'])
def get_alignment_matrix():
    # Call Homography.cc
    matrix = retrieve('storage/alignment/output/matrix.json')
    return response(matrix)

########################## OPTIMIZATION PARAMS ################################

def get_optimization_parameters():
    params = retrieve('storage/optimization/input/parameters.json')
    return response(params)

def update_optimization_parameters(params):
    '''
    Input example:
    {
        "param-1": 50,
        "param-2": 0.2
    }
    '''
    store(params, 'storage/optimization/input/parameters.json')
    updated_params = retrieve('storage/optimization/input/parameters.json')
    return response(updated_params)

@app.route('/api/v1/data/optimization/input/parameters', methods=['PUT', 'GET'])
def optimization_parameters():
    if request.method == 'PUT':
        update_optimization_parameters()
    else:
        get_optimization_parameters()

########################### OPTIMIZATION MATRIX ###############################

@app.route('/api/v1/data/optimization/output/matrix', methods=['GET'])
def get_optimization_matrix():
    # Call Optimizer.py
    matrix = retrieve('storage/optimization/output/matrix.json')
    return response(matrix)

###############################################################################

app.run()
