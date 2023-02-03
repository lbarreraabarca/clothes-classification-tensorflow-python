import uuid
import numpy as np
from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.science.adapters.TensorFlowOperator import TensorFlowOperator
from com.data.factory.adapters.ImageFileOperator import ImageFileOperator
from com.data.factory.utils.logger import logging

LOG = logging.getLogger(__name__)

class ProductClasificatorService():
    def run(self):
        classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        LOG.info('Downloading image')
        image = ImageFileOperator()
        url = 'https://http2.mlstatic.com/D_NQ_NP_2X_660180-MLC48760902612_012022-F.webp'
        imagePath = f'tmp/{str(uuid.uuid4())}'
        image.download(url, imagePath)

        LOG.info('Loading model')
        neuralNetwork = TensorFlowOperator()
        neuralNetwork.preprocessingData()
        model = neuralNetwork.loadModel('keras/model')

        LOG.info('Vectorizing image.')
        imageVector = neuralNetwork.vectorizeImage(imagePath)
        LOG.info('Predicting image')
        predictions = model.predict(imageVector)
        label = np.argmax(predictions[0])

        response = {
            "response": {
                "label": str(label),
                "classNames": str(classNames[label])
            }
        }
        return (response, 200)
