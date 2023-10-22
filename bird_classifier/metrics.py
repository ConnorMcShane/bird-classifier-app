"""This module contains the Metrics class, which is used to track metrics during inference."""
import numpy as np

from bird_classifier.utils import Utils


class Metrics:
    """Class for tracking metrics during inference."""

    def __init__(self, config, logger=None):
        """Initialise the metrics class

        Args:
            config (Config): Configuration object
        """

        self.logger = logger
        self.utils = Utils(config)
        self.classes_dict = self.get_classes(config.classes_url)
        self.top_n = config.top_n
        self.global_step = 0

        self.inference_requests = 0
        self.images_loaded = 0
        self.loading_failures = 0
        self.samples_inferenced = 0


    def get_classes(self, classes_url):
        """Get the labels from the url
        
        Args:
            classes_url (str): URL of the classes file
        
        Returns:
            classes (dict): Dictionary of classes
        """

        # Get the classes
        response = self.utils.load_url(classes_url, self.logger)
        if response is None:
            raise ConnectionError(f'Failed to connect to {classes_url}')

        # Read the classes
        classes = response.decode('utf-8')
        classes = classes.split('\n')
        classes.pop(0)  # remove header (id, name)
        classes = {int(row.split(',')[0]): row.split(',')[1] for row in classes if row != ''}

        return classes


    def ind_to_class(self, ind):
        """Convert the index to a class
        
        Args:
            ind (int): Index of the class
            
        Returns:
            class (str): Class name
        """

        return self.classes_dict[ind]


    def get_top_n_result(self, outputs, n=None):
        """Get the top n results from the model outputs
        
        Args:
            outputs (np.array): Model outputs shape: (batch_size, num_classes)
            n (int, optional): Number of results to return. Defaults to 3.
        
        Returns:
            top_n_classes (list): List of top n classes
            top_n_probabilities (list): List of top n probabilities
        """

        # Set the default n
        if n is None:
            n = self.top_n
        
        # Get the top n results
        top_n_classes = np.argsort(outputs, axis=-1)[:, -n:][:, ::-1]
        top_n_values = np.take_along_axis(outputs, top_n_classes, axis=-1)

        # map the indices to the class names
        top_n_classes = np.vectorize(self.ind_to_class)(top_n_classes)

        return top_n_classes, top_n_values


    def return_dict(self, sample_output, url):
        """Return a dictionary of the results for a single sample.

        Args:
            sample_output (tuple): Tuple of the results

        Returns:
            dict: Dictionary of the results
        """
        top_n_classes, top_n_values, loaded = sample_output
        n = len(top_n_classes)
        output_dict = {
            "url": url,
            "loaded": loaded,
        }

        for position, (class_name, confidence) in enumerate(zip(top_n_classes, top_n_values)):
            if loaded:
                output_dict[f"match_{str(position + 1).zfill(3)}"] = {
                    "class_name": class_name,
                    "confidence": confidence,
                }
            else:
                output_dict[f"match_{str(position + 1).zfill(3)}"] = {
                    "class_name": "Image failed to load",
                    "confidence": 0.0,
                }

        return output_dict
    

    def performance(self, total_time, num_samples, num_batches, output_dict):
        
        for key, value in output_dict.items():
            self.inference_requests += 1
            if value["loaded"]:
                self.images_loaded += 1
                self.samples_inferenced += 1
            else:
                self.loading_failures += 1

        metrics = dict(
            inference_requests = self.inference_requests,
            images_loaded = self.images_loaded,
            loading_failures = self.loading_failures,
            samples_inferenced = self.samples_inferenced,
            batch_time = total_time / (num_batches + 1),
            average_sample_time = total_time / num_samples,
        )

        return metrics