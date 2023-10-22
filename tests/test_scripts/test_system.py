"""System tests for the bird_classifier module."""
import os
import sys

from bird_classifier.bird_classifier import BirdClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


url_dict = {
    0:'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    1:'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    2:'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    3:'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    4:'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
}


class TestClass:
    """Test class for BirdClassifier."""

    def test_bird_classifier_initialisation(self):

        # initialise the bird classifier
        self.classifier = BirdClassifier()


    def test_bird_classifier_main(self):

        # initialise the bird classifier
        self.classifier = BirdClassifier()

        # run the main function
        self.classifier.classify(url_dict)
