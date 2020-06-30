# ------------------------------------------------------------------------------
# Simple Flask Rest API

# curl -F 'file=@CNN.java' http://127.0.0.1:5000/summarizer | curl http://127.0.0.1:5000/ -o output.java
# curl -F 'file=@CNN.java' http://127.0.0.1:5000/summarizer


# ------------------------------------------------------------------------------

import os
from flask import Flask, render_template, request, send_file
import neural_code_sum

app = Flask(__name__)
# app.config.from_object(os.environ['APP_SETTINGS'])
# print(os.environ['APP_SETTINGS'])


# ------------------------------------------------------------------------------
# Test Rest API
# ------------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'java'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods = ['GET','POST'])
def hello():
	if(request.method == 'POST'):
		if('file' not in request.files):
			return 'NO FILE'
		file = request.files['file']
		if(file.filename == ''):
			print('NO FILES')
			return redirect(request.url)
		if(file and allowed_file(file.filename)):
			uploadedFile = file.filename
			file.save(os.path.join(UPLOAD_FOLDER, file.filename))
			if(uploadedFile != ''):
				neural_code_sum.starter(uploadedFile)
			return render_template('index.html', message='success')
	return render_template('index.html', message='NOT UPLOADED (ONLY .JAVA FILES ALLOWED)')


@app.route('/summarizer', methods = ['POST'])
def documentation():
	if(request.method == 'POST'):
		if('file' not in request.files):
			return 'NO FILE'
		file = request.files['file']
		if(file.filename == ''):
			print('NO FILES')
			return redirect(request.url)
		if(file and allowed_file(file.filename)):
			uploadedFile = file.filename
			file.save(os.path.join(UPLOAD_FOLDER, file.filename))
			if(uploadedFile != ''):
				neural_code_sum.starter(uploadedFile)
				doc = os.path.dirname(os.path.realpath(__file__))+'/output.java'
				return send_file(doc,as_attachment=True,cache_timeout=0)

# ------------------------------------------------------------------------------
# Download Flask Server
# ------------------------------------------------------------------------------

@app.route('/download', methods = ['GET'])
def download_file():
	global uploadedFile
	doc = os.path.dirname(os.path.realpath(__file__))+'/output.java'
	return send_file(doc,as_attachment=True,cache_timeout=0)


@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


if __name__ == '__main__':
    app.run()
