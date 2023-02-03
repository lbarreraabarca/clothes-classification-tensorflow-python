from com.data.science.ports.NeuralNetwork import NeuralNetwork
from com.data.factory.utils.logger import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt


LOG = logging.getLogger(__name__)

class TensorFlowOperator(NeuralNetwork):

    def __init__(self):
        self._trainSet = None
        self._trainLabel = None
        self._testSet = None
        self._testLabel = None

    def preprocessingData(self):
        LOG.info('Downloading images from keras.')
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        
        
        LOG.info('Preprocessing images.')
        train_images.shape
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        print(train_images[0].shape)

        self._trainSet = train_images
        self._trainLabel = train_labels
        self._testSet = test_images
        self._testLabel = test_labels

    def trainModel(self, epochs: int):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.trainSet, self.trainLabel, epochs=epochs)
        return model

    def loadModel(self, modelPath: str):
        if modelPath is None or modelPath == '':
            raise ValueError('Model path cannot be None or empty.')
        return tf.keras.models.load_model(modelPath)

    def saveModel(self, model, modelPath: str):
        if modelPath is None or modelPath == '':
            raise ValueError('Model path cannot be None or empty.')
        return tf.keras.models.save_model(model, modelPath)

    def vectorizeImage(self, imagePath: str):
        if imagePath is None or imagePath == '':
            raise ValueError('Image path cannot be None or empty.')
        try:
            image = keras.utils.load_img(imagePath)
            image = tf.image.rgb_to_grayscale(image)
            inputArray = tf.keras.utils.img_to_array(image) / 255.0
            
            plt.figure(figsize=(10,10))
            plt.subplot(5,5,1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(inputArray, cmap=plt.cm.binary)
            #plt.show()
            inputArray = tf.image.resize(np.array([inputArray]), [28,28])[0,...,0].numpy()
            return (np.expand_dims(inputArray,0))
            #print(inputArray.shape)
            #return inputArray
        except Exception as e:
            raise Exception(f'Error when it was vectorizing the image {imagePath}.')

    @property
    def trainSet(self):
        return self._trainSet

    @property
    def trainLabel(self):
        return self._trainLabel

    @property
    def testSet(self):
        return self._testSet

    @property
    def testLabel(self):
        return self._testLabel
