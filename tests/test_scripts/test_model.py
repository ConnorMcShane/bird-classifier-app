import os
import sys
import tensorflow_hub as hub
import tensorflow as tf
import cv2

from bird_classifier.logger import Logger
from bird_classifier.model import ModelWrapper
from bird_classifier.metrics import Metrics
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


class TestClass:
    """Test class for ModelWrapper."""

    def test_model_initialisation(self):

        # initialise the logger
        self.logger = Logger(Config)

        # initialise the metrics
        self.model = ModelWrapper(Config, self.logger)

        assert len(self.logger.logger.handlers) == 2
        assert isinstance(self.model.model, hub.keras_layer.KerasLayer)


    def test_model_call(self):

        # initialise the logger
        self.logger = Logger(Config)

        # initialise the metrics
        self.model = ModelWrapper(Config, self.logger)

        # initialise the metrics
        self.metrics = Metrics(Config, self.logger)

        # load the example image
        image = cv2.imread(os.path.join(Config.root_dir, Config.example_image_file))
        image = cv2.resize(image, Config.img_size)
        image = image[:, :, ::-1]
        image = image / 255.0
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        # get the model output
        output = self.model(image)
        top_n_classes, top_n_values = self.metrics.get_top_n_result(output.numpy(), n=1)

        # check that the top n classes are correct
        assert top_n_classes[0][0] == "Erithacus rubecula"
        assert round(float(top_n_values[0][0]), 4) == float(0.8474)
