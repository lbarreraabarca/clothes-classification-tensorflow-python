
from flask import Flask
from com.data.factory.controllers.Controller import controller
from com.data.factory.controllers.ProductClassificator import productClassificator
from com.data.factory.utils.logger import logging
from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.science.adapters.TensorFlowOperator import TensorFlowOperator

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    PORT = 8080
    app = Flask(__name__)
    app.register_blueprint(controller, url_prefix='/api/v1')
    app.register_blueprint(productClassificator, url_prefix='/api/v1')
    app.run(host='127.0.0.1', port=PORT, debug=True)
