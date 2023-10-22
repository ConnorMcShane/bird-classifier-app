import os
import sys
import logging

from bird_classifier.logger import Logger, LOGGING_LEVELS
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


class TestClass:
    """Test class for the logger."""

    def test_logger_initialisation(self):

        # delete the log file if it exists
        log_file_path = os.path.join(Config.root_dir, Config.log_file)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        # initialise the logger
        self.logger = Logger(Config)

        # check that the logger has been initialised
        assert len(self.logger.logger.handlers) == 2
        for handler in self.logger.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                assert handler.level == LOGGING_LEVELS[Config.log_level_file]
            elif isinstance(handler, logging.StreamHandler):
                assert handler.level == LOGGING_LEVELS[Config.log_level_console]

        # read the log file
        with open(log_file_path, "r") as f:
            # read whole file
            log_file = f.readlines()
        
        # check that the log file is not empty
        assert "Logger initialised" in log_file[0]
        assert "Configuration logged" in log_file[-1]


    def test_call(self):
        # delete the log file if it exists
        log_file_path = os.path.join(Config.root_dir, Config.log_file)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        # initialise the logger
        self.logger = Logger(Config)

        # log a message
        self.logger("INFO", "Test message")

        # read the log file
        with open(log_file_path, "r") as f:
            # read whole file
            log_file = f.readlines()

        # that the last message in the log file is the test message
        assert "Test message" in log_file[-1]


    def test_log_config(self):
        # delete the log file if it exists
        log_file_path = os.path.join(Config.root_dir, Config.log_file)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        # initialise the logger
        self.logger = Logger(Config)

        # log a config
        self.logger.log_config()

        # read the log file
        with open(log_file_path, "r") as f:
            # read whole file
            log_file = f.readlines()

        # that the last message in the log file is Configuration logged
        assert "Configuration logged" in log_file[-1]
