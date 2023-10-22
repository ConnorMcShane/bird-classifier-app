"""Data loader for the bird classifier"""
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from bird_classifier.utils import Utils


class DataLoader:
    """Data loader for the bird classifier"""

    def __init__(self, config, url_dict, logger=None) -> None:
        """Initialise the data loader

        Args:
            config (Config): Configuration object
        """

        self.logger = logger
        self.config = config
        self.utils = Utils(config)
        self.img_url_dict = url_dict
        self.dataloader = self.get_dataloader()


    def __len__(self):
        """Get the length of the dataloader
        
        Returns:
            len (int): Length of the dataloader
        """

        return len(self.img_url_dict)


    def blank_image(self):
        """Generate a blank image"""
        return tf.zeros((*self.config.img_size, 3), dtype=np.float32)


    def preprocess_image(self, image):
        """Preprocess the image
        
        Args:
            image (np.array): Image to preprocess

        Returns:
            image (Tensor): Preprocessed image
        """

        # Read and preprocess the image
        if isinstance(image, bytes):
            image = np.asarray(bytearray(image), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        elif not isinstance(image, np.ndarray):
            self.logger("ERROR", f"Image is not a valid type: {type(image)}")
            return None
        image = cv2.resize(image, self.config.img_size)
        if self.config.img_to_rgb:
            image = image[:, :, ::-1]  # Convert to RGB
        if self.config.img_devide_by_255:
            image = image / 255.0

        return image


    def get_image(self, image_url):
        """Download the image from the url
        
        Args:
            image_url (str): URL of the image
            
        Returns:
            image (Tensor(n.array)): Preprocessed image
            load_status (Tensor(int)): Status of the image load
        """

        # Download the image
        image = self.utils.load_url(image_url, self.logger)
        if image is None:
            return self.blank_image(), tf.constant(False, dtype=tf.bool)

        # Read and preprocess the image
        image = self.preprocess_image(image)
        if image is None:
            return self.blank_image(), tf.constant(False, dtype=tf.bool)

        # Generate tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        return image, tf.constant(True, dtype=tf.bool)


    def get_dataloader(self):
        """Get the dataloader
        
        Returns:
            dataloader (iter): Dataloader
        """

        dataloader = tf.data.Dataset.from_generator(lambda: list(self.img_url_dict.keys()), output_types=tf.int32)
        sample_loader = lambda index: tf.py_function(self.load_sample, [index], (tf.int32, tf.float32, tf.bool))
        dataloader = dataloader.map(sample_loader, num_parallel_calls=self.config.batch_size)
        dataloader = dataloader.batch(self.config.batch_size)
        dataloader = dataloader.prefetch(buffer_size=tf.data.AUTOTUNE)

        return iter(dataloader)


    def load_sample(self, sample_idx):
        """
        Load and preprocess an image from a URL

        Args:
            sample_idx (Tesnor): URL of the image

        Returns:
            image (Tensor): Preprocessed image
            load_status (int): Status of the image load
        """

        image_url = self.img_url_dict[sample_idx.numpy()]
        image, load_status = self.get_image(image_url)
        idx = tf.constant(sample_idx.numpy(), dtype=tf.int32)

        return (idx, image, load_status)


    def __iter__(self):
        return self.dataloader


    def __next__(self):
        return next(self.dataloader)
