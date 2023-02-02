from com.data.factory.utils.logger import logging
from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.science.adapters.TensorFlowOperator import TensorFlowOperator

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    LOG.info("Starting deep learning process.")
    classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    neuralNetwork: NeuralNetwork = TensorFlowOperator()
    neuralNetwork.preprocessingData()
    model = neuralNetwork.trainModel(10)
    model.save('keras/model')
