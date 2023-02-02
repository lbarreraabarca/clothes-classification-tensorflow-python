from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.science.adapters.TensorFlowOperator import TensorFlowOperator

class ProductClasificatorService():
    def run():
        neuralNetwork: NeuralNetwork = TensorFlowOperator
        model = neuralNetwork.loadModel('keras/model')