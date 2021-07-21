from flask import Flask, json, jsonify, make_response, abort
from flask.globals import request
import os

from flask.signals import message_flashed

app = Flask(__name__)
app.config["DEBUG"] = True


################################### STORAGE ###################################

def store(data, relative_path_to_file):
    '''
    Converts dictionary into JSON format and stores it in file
    '''
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, relative_path_to_file)

    try:
        with open(file, "r+") as f:
            json.dump(data, f)
            f.close()
        return create_msg(1, 201, "File successfully stored.")
    except FileNotFoundError:
        return create_msg(0, 400, "File not found.")

def retrieve(relative_path_to_file):
    '''
    Returns requested json file as dictionary
    '''
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, relative_path_to_file)
    try:
        with open(file) as f:
            data = json.load(f)
            f.close()
        return create_msg(1, 200, data)
    except ValueError:
        return create_msg(0, 400, "File is empty.")
    except FileNotFoundError:
        return create_msg(0, 400, "File not found.")

#################################### ERROR ####################################

@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

################################## RESPONSE ###################################

def response(data, code=200):
    return jsonify(data), code

def create_msg(status, code, msg):
    return {"status": status, "code": code, "msg": msg}

#################################### MAPS #####################################

def get_maps():
    return retrieve('storage/maps/maps.json')

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

    attributes = ("map-number", "path_to_file", "_id", "fleet_name", "floor_name", "translation", "rotation", "resolution", "axis", "createdAt", "updatedAt")
    if all(key in maps[0] for key in attributes) and all(key in maps[1] for key in attributes):
        return store(maps, 'storage/maps/maps.json')
    else:
        return create_msg(0, 400, "Missing map details.")

@app.route('/api/v1/data/maps', methods=['POST', 'GET'])
def update_maps():
    if request.method == 'POST':
        x = store_maps(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])

    else:
        x = get_maps()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])

################################### POINTS ####################################

def get_points():
    return retrieve('storage/alignment/input/points.json')

def delete(point):
    '''
    Input example:
    {
        "map-number" : 1,
        "point" : "(1, 2)"
    }
    '''
    attributes = ("map-number", "point")
    if not all(key in point for key in attributes):
        return create_msg(0, 400, "Missing map ID or point.")

    else:
        data = retrieve('storage/alignment/input/points.json')
        if data["status"] == 0:
            return data

        else:
            current_pts = data["msg"]
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

            return store(new_pts, 'storage/alignment/input/points.json')

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
    attributes = ("map-number", "points")
    if not ( all(key in points[0] for key in attributes) and all(key in points[1] for key in attributes) ):
        return create_msg(0, 400, "Missing map ID or points.")

    data = retrieve('storage/alignment/input/points.json')

    if data["status"] == 0 and data["msg"] == "File not found.":
        return data

    else:
        map_1_to_add = points[0]
        map_2_to_add = points[1]

        if not len(map_1_to_add["points"]) == len(map_2_to_add["points"]):

            return create_msg(0, 400, "Differing number of corresponding points from each map.")

        elif len(map_1_to_add["points"]) <= 0 or len(map_2_to_add["points"]) <= 0:

            return create_msg(0, 400, "Number of input points cannot be less than 1.")

        points_new = []

        if data["status"] == 1:
            current_pts = data["msg"]
            map_1 = current_pts[0]
            map_2 = current_pts[1]

            if not ( (map_1["map-number"] == map_1_to_add["map-number"] or map_2["map-number"] == map_1_to_add["map-number"])
                and (map_1["map-number"] == map_2_to_add["map-number"] or map_2["map-number"] == map_2_to_add["map-number"]) ):

                return create_msg(0, 400, "Mismatching maps. Maps that points are selected from should have same IDs as maps stored in database.")

            else:
                map_1["points"].extend(map_1_to_add["points"])
                map_2["points"].extend(map_2_to_add["points"])
                points_new.extend([map_1, map_2])
        else:
            points_new = points

        store(points_new, 'storage/alignment/input/points.json')

        if not len(points_new[0]["points"]) == len(points_new[1]["points"]):
            return create_msg(0, 400, "Differing number of corresponding points from each map.")
        else:
            return create_msg(1, 201, "Points successfully updated.")

@app.route('/api/v1/data/alignment/input/points', methods=['POST', 'GET'])
def update_points():
    if request.method == 'POST':
        x = add_points(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])
    else:
        x = get_points()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])

@app.route('/api/v1/data/alignment/input/points/delete', methods=['POST'])
def delete_points():
    x = delete(request.json)

    if x["status"] == 0:
            abort(x["code"], description=x["msg"])
    else:
            return response(x["msg"])

############################## ALIGNMENT PARAMS ###############################

def get_ave_sym_dist():
    return retrieve('storage/alignment/input/ave_sym_dist.json')

def update_ave_sym_dist(value):
    '''
    Input example:
    {
        "ave_sym_dist": 0.1,
    }
    '''
    attributes = ("ave_sym_dist")
    if not all(key in value for key in attributes):
        return store(value, 'storage/alignment/input/ave_sym_dist.json')
    else:
        return create_msg(0, 400, "Missing parameter value.")

@app.route('/api/v1/data/alignment/input/parameters/ave_sym_dist', methods=['POST', 'GET'])
def ave_sym_dist():
    if request.method == 'POST':
        x = update_ave_sym_dist(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])
    else:
        x = get_ave_sym_dist()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])


def get_max_num_iter():
    return retrieve('storage/alignment/input/max_num_iter.json')

def update_max_num_iter(value):
    '''
    Input example:
    {
        "max_num_iter": 50,
    }
    '''
    attributes = ("max_num_iter")
    if not all(key in value for key in attributes):
        return store(value, 'storage/alignment/input/max_num_iter.json')
    else:
        return create_msg(0, 400, "Missing parameter value.")

@app.route('/api/v1/data/alignment/input/parameters/max_num_iterations', methods=['POST', 'GET'])
def max_num_iterations():
    if request.method == 'POST':
        x = update_max_num_iter(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])
    else:
        x = get_max_num_iter()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])

############################## ALIGNMENT MATRIX ###############################

@app.route('/api/v1/data/alignment/output/matrix', methods=['GET'])
def get_alignment_matrix():

    # Call Homography.cc

    x = retrieve('storage/alignment/output/matrix.json')

    if x["status"] == 0:
            abort(x["code"], description=x["msg"])
    else:
        return response(x["msg"], code=x["code"])

############################# OPTIMIZATION PARAMS #############################

def get_min_dist():
    return retrieve('storage/optimization/input/min_dist.json')

def update_min_dist(value):
    '''
    Input example:
    {
        "min_dist": 0.1,
    }
    '''
    attributes = ("min_dist")
    if not all(key in value for key in attributes):
        return store(value, 'storage/optimization/input/min_dist.json')
    else:
        return create_msg(0, 400, "Missing parameter value.")

@app.route('/api/v1/data/optimization/input/parameters/min_dist', methods=['POST', 'GET'])
def min_dist():
    if request.method == 'POST':
        x = update_min_dist(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])
    else:
        x = get_min_dist()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])

def get_max_corners():
    return retrieve('storage/optimization/input/max_corners.json')

def update_max_corners(value):
    '''
    Input example:
    {
        "min_dist": 0.1,
    }
    '''
    attributes = ("max_corners")
    if not all(key in value for key in attributes):
        return store(value, 'storage/optimization/input/max_corners.json')
    else:
        return create_msg(0, 400, "Missing parameter value.")

@app.route('/api/v1/data/optimization/input/parameters/max_corners', methods=['POST', 'GET'])
def max_corners():
    if request.method == 'POST':
        x = update_max_corners(request.json)

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"], code=x["code"])
    else:
        x = get_max_corners()

        if x["status"] == 0:
            abort(x["code"], description=x["msg"])
        else:
            return response(x["msg"])

############################# OPTIMIZATION MATRIX #############################

@app.route('/api/v1/data/optimization/output/matrix', methods=['GET'])
def get_optimization_matrix():
    x = retrieve('storage/optimization/output/matrix.json')

    if x["status"] == 0:
            abort(x["code"], description=x["msg"])
    else:
            return response(x["msg"], code=x["code"])

################################# GRADIENT MAP ################################

@app.route('/api/v1/data/optimization/output/gradient', methods=['GET'])
def get_gradient_map():

    # Call Optimizer.py
    x = retrieve('storage/optimization/output/gmap.json')

    if x["status"] == 0:
            abort(x["code"], description=x["msg"])
    else:
            return response(x["msg"], code=x["code"])

################################# FINAL MATRIX ################################

@app.route('/api/v1/data/optimization/output/final', methods=['GET'])
def get_final_matrix():
    x = retrieve('storage/optimization/output/finalmatrix.json')

    if x["status"] == 0:
            abort(x["code"], description=x["msg"])
    else:
            return response(x["msg"], code=x["code"])

###############################################################################

app.run()