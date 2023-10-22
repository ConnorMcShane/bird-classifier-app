"""Utils methods for bird classifier."""

import requests
from requests.exceptions import ConnectionError


class Utils:
    """Utils class for bird classifier."""

    def __init__(self, config):
        """Initialise the utils class."""

        self.config = config


    @staticmethod
    def load_url(url, logger=None):
        """Load image from url.

        Args:
            url: Url to load image from.

        Returns:
            Image content if successful, else None.
        """

        try:
            headers = {'User-Agent': 'BirdClassifier/1.0'}
            response = requests.get(url, timeout=2, headers=headers)
        except ConnectionError:
            if logger is not None:
                logger("WARNING", f'Failed to connect to {url}')

            return None

        if response.status_code != 200:
            if logger is not None:
                logger("WARNING", f'Image download failed with status code {response.status_code}')
                logger("WARNING", f'Image url: {url}')
            return None
        
        return response.content
