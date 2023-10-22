"""Main module for the bird classifier."""
import time

from bird_classifier.logger import Logger
from bird_classifier.data_loader import DataLoader
from bird_classifier.model import ModelWrapper
from bird_classifier.metrics import Metrics
from bird_classifier.config import Config


class BirdClassifier:
    """Main class for the bird classifier."""

    def __init__(self, logger=None):
        """Initialise the bird classifier."""

        # Initialise the logger
        start_time = time.time()
        self.logger = Logger(Config, logger)
        self.model = ModelWrapper(Config, self.logger)
        self.metrics = Metrics(Config, self.logger)
        self.logger("INFO", f"Bird classifier initialised in {time.time() - start_time:.2f} seconds")


    def classify(self, url_dict):
        """Classify the birds in the images at the URLs in the url_dict.

        Args:
            url_dict (dict): Dictionary of image URLs
        """

        start_time = time.time()
        # Convert the keys to ints if they are strings
        if isinstance(next(iter(url_dict.keys())), str):
            url_dict = {int(k): v for k, v in url_dict.items()}

        # Initialise the data loader
        self.data_loader = DataLoader(Config, url_dict, self.logger)
        output_dict = {}

        # Iterate over the data loader
        for batch_idx, (sample_ids, images, load_status) in enumerate(self.data_loader):
            
            # Infer the model outputs for this batch
            model_output = self.model(images)

            # Get the top n classes and values
            top_n_classes, top_n_values = self.metrics.get_top_n_result(model_output.numpy())

            # Prepare the outputs for logging
            batch_outputs = zip(sample_ids.numpy(), zip(top_n_classes, top_n_values, load_status.numpy()))
            batch_output_dict = {
                s_id: self.metrics.return_dict(output, self.data_loader.img_url_dict[s_id]) 
                for s_id, output in batch_outputs
            }

            # log the model outputs for this batch
            self.logger.log_model_outputs(batch_output_dict, batch_idx)
            output_dict.update(batch_output_dict)
        
        # timer
        total_time = time.time() - start_time
        num_samples = len(output_dict)

        # generate metrics
        metrics = self.metrics.performance(total_time, num_samples, batch_idx, output_dict)

        # log the metrics
        self.logger.log_metrics(metrics, self.metrics.global_step)
        self.logger.finish(total_time, num_samples)
        self.metrics.global_step += 1

        return output_dict
