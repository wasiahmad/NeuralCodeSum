# ------------------------------------------------------------------------------
# Simple Flask Rest API
# ------------------------------------------------------------------------------

import os
from flask import Flask
from flask import request

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
print(os.environ['APP_SETTINGS'])


# ------------------------------------------------------------------------------
# Test Rest API
# ------------------------------------------------------------------------------

@app.route('/')
def hello():
    return "Hello World!"

# ------------------------------------------------------------------------------
# Shutdown Flask Server
# ------------------------------------------------------------------------------


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
