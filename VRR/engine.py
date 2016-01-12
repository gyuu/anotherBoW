from flask import (
	Flask, request, json,
	render_template, Response,
	)
from werkzeug import secure_filename
from image_paths import get_image_subpaths
from random import sample
import os, shutil

# from search import search_api
from time import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath("static/upload")
app.config['IMAGE_FOLDER'] = os.path.abspath("static/images")

answer_folder = os.path.abspath("static/answer")
app.config['ANSWER_FILE'] = os.path.join(answer_folder, "answer.txt")

image_subpaths = get_image_subpaths(app.config['IMAGE_FOLDER'])

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload-mat', methods=['POST'])
def process_upload_mat():
	f = request.files['image']
	if f:
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		# do the image query, and return a list of image path.
		answer_file = app.config['ANSWER_FILE']
		while not os.path.exists(answer_file):
			print "Waiting for answer..."

		with open(answer_file) as f:
			images = [l.strip() for l in f.readlines()]
		os.remove(answer_file)

		return Response(json.dumps(images), mimetype='application/json')

# @app.route('/upload', methods=['POST'])
# def process_upload_py():
# 	f = request.files['image']
# 	if f:
# 		filename = secure_filename(f.filename)
# 		img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# 		f.save(os.path.join(img_path))

# 		start = time()
# 		images = search_api(img_path)
# 		end = time()

# 		results = {
# 			"images":images,
# 			"time":round(end - start, 2),
# 		}

# 		return Response(json.dumps(results), mimetype='application/json')

# @app.route('/bypath', methods=['POST'])
# def process_query_by_path():
# 	path = request.form['path']
		
# 	start = time()
# 	images = search_api(path)
# 	end = time()

# 	results = {
# 		"images":images,
# 		"time":round(end - start, 2),
# 	}

# 	return Response(json.dumps(results), mimetype='application/json')

@app.route('/bypath-mat', methods=['POST'])
def process_query_by_path():
	filename = request.form['path']
	shutil.copy(filename, os.path.join(app.config['UPLOAD_FOLDER'], filename.split('/')[-1]))

	# do the image query, and return a list of image path.
	answer_file = app.config['ANSWER_FILE']
	while not os.path.exists(answer_file):
		print "Waiting for answer..."

	with open(answer_file) as f:
		images = [l.strip() for l in f.readlines()]
	os.remove(answer_file)

	return Response(json.dumps(images), mimetype='application/json')


@app.route('/random_image', methods=['GET'])
def get_random_images():
	images = sample(image_subpaths, 20);
	# print json.dumps(images)
	return Response(json.dumps(images), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)

