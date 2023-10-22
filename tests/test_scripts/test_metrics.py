import os
import sys
import numpy as np

from bird_classifier.metrics import Metrics
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


class TestClass:
    """Test class for metrics."""

    def test_metrics_initialisation(self):

        # initialise the metrics
        self.metrics = Metrics(Config)

        assert self.metrics.top_n == Config.top_n
        assert len(self.metrics.classes_dict) == 965

    def test_ind_to_class(self):
        
        # initialise the metrics
        self.metrics = Metrics(Config)

        # test that the correct class is returned
        assert self.metrics.ind_to_class(0) == "Haemorhous cassinii"
        assert self.metrics.ind_to_class(963) == "Ardenna gravis"

    
    def test_get_top_n_results(self):

        # initialise the metrics
        self.metrics = Metrics(Config)

        # initialise the outputs
        outputs = np.load(os.path.join(Config.root_dir, Config.example_output_file))

        # get the top n results
        top_n_classes, top_n_values = self.metrics.get_top_n_result(outputs)

        # check that the top n classes are correct
        assert len(top_n_classes[0]) == Config.top_n
        assert top_n_classes[0][0] == "Phalacrocorax varius varius"
        assert top_n_classes[1][0] == "Galerida cristata"
        assert top_n_classes[2][0] == "Eumomota superciliosa"

        # check that the top n values are correct
        assert len(top_n_values[0]) == Config.top_n
        assert round(float(top_n_values[0][0]), 4) == float(0.834)
        assert round(float(top_n_values[1][0]), 4) == float(0.8316)
        assert round(float(top_n_values[2][0]), 4) == float(0.42)
