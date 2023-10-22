import os
import sys
import numpy as np

from bird_classifier.utils import Utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


class TestClass:
    """Test class for utils."""

    def test_utils_initialisation(self):
        
        # initialise the utils
        self.utils = Utils(Config)


    def test_load_url(self):

        # initialise the utils
        self.utils = Utils(Config)

        # test that the url is loaded correctly
        response = self.utils.load_url("https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg")
        assert isinstance(response, bytes)


    def test_load_url_fail(self):

        # initialise the utils
        self.utils = Utils(Config)

        # test that the url is loaded correctly
        response = self.utils.load_url("https://thisisnotevenarealurl.com/this_is_fake/no_image_here.jpg")
        assert response is None
