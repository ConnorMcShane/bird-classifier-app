"""Model wrapper for TensorFlow Hub models."""
import tensorflow_hub as hub
import logging
import json


class ModelWrapper:
    """Model wrapper for TensorFlow Hub models."""

    def __init__(self, config, logger=None):
        """Initialise the model wrapper

        Args:
            config (Config): Configuration object
        """

        self.logger = logger
        self.config = config
        self.model_url = self.get_model_url()
        self.model = self.load_model(self.model_url)

    
    def get_model_url(self):
        """Get the model URL

        Returns:
            str: Model URL
        """

        # read model registry json file
        with open(self.config.model_registry, "r") as f:
            model_registry = json.load(f)

        # get model URL from model registry
        models = model_registry["models"]
        model_versions = models[self.config.model_name]["versions"]
        model_url = model_versions[self.config.model_version]["url"]

        return model_url


    def load_model(self, model_url):
        """Load the model with tensorflow_hub
        tensorflow_hub adds logging handlers to the root logger.
        this can mess up logging in the main script.
        this function clears all handlers from the root logger that are added by tensorflow_hub.
        
        Args:
            model_url (str): URL of the model
            
        Returns:
            model: TensorFlow Hub model
        """
        # get list of logging handlers before loading the model with tensorflow_hub
        original_handlers = logging.root.handlers[:]

        # load the model with tensorflow_hub
        model = hub.KerasLayer(model_url)

        # get list of logging handlers after loading the model with tensorflow_hub
        current_handlers = logging.root.handlers[:]

        # identify new handlers added by tensorflow_hub
        new_handlers = [h for h in current_handlers if h not in original_handlers]

        # clear all handlers from the root logger that are added by tensorflow_hub
        for handler in new_handlers:
            # identify handlers added by tensorflow_hub
            logging.root.removeHandler(handler)

        return model


    def __call__(self, input):
        """Call the model

        Args:
            input (Any): Input to the model

        Returns:
            Any: Output of the model
        """

        return self.model.call(input)
