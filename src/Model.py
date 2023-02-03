from com.data.factory.utils.logger import logging
from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.science.adapters.TensorFlowOperator import TensorFlowOperator

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    LOG.info("Starting deep learning process.")
    classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    neuralNetwork = TensorFlowOperator()
    neuralNetwork.preprocessingData()
    model = neuralNetwork.trainModel(10)
    neuralNetwork.saveModel(model, 'keras/model')
    LOG.info("Loading model")
    model = neuralNetwork.loadModel('keras/model')

    model.evaluate(neuralNetwork.testSet, neuralNetwork.testLabel, verbose=2)
