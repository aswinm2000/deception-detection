from flask import Flask, request, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import requests
from predict import *

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)

class HelloWorld(Resource):
    def post(self):
        data = request.get_json()
        path = data["path"]
        try:
            prediction = predict(path)
            if(prediction==1):
                return "Truth"
            else:
                return "Lie"
        except:
            return "Failure"


api.add_resource(HelloWorld,'/api/[redict]')

if __name__ == '__main__':
    app.run(debug=False,port = 5002)
