"""Logger class"""
import logging
import os


LOGGING_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
}


class Logger:
    """Logger class"""

    def __init__(self, config, logger=None):
        """Initialise the logger

        Args:
            config (Config): Configuration object
            logger (Logger, optional): Logger object. Defaults to None.
        """

        # set the configuration
        self.config = config

        # Create a logger
        if logger is None:
            self.logger = logging.getLogger("my_logger")
            self.logger.handlers.clear()
            logging.getLogger(__package__).propagate = False
        
            # Set the logger's level to DEBUG
            self.logger.setLevel(logging.DEBUG)

            # Create a file handler with a log level of DEBUG
            file_handler = logging.FileHandler(os.path.join(config.root_dir, config.log_file))
            file_handler.setLevel(config.log_level_file)  # Set the log level for file logging

            # Create a console handler with a log level of WARNING
            console_handler = logging.StreamHandler()
            console_handler.setLevel(config.log_level_console)  # Set the log level for console logging

            # Create a formatter with the desired log message format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Set the formatter for the handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        else:
            self.logger = logger

        # Log that the logger has been initialised
        self.logger.debug("Logger initialised")

        # Log the configuration
        self.log_config()

        # Initialise wandb logging
        self.init_wandb_logging()


    def __call__(self, level, message):
        """Log an info message

        Args:
            level (str): Logging level
            message (str): Message to log
        """
        self.logger.log(LOGGING_LEVELS[level], message)


    def log_config(self):
        """Log the configuration

        Args:
            config (Config): Configuration object
        """
        self.logger.debug("Logging configuration")
        for key, value in vars(self.config).items():
            if key.startswith('__'):
                continue
            self.logger.debug(f"    {key}: {value}")
        self.logger.debug("Configuration logged")


    def log_model_outputs(self, output_dict, batch_idx):
        """Log the model outputs

        Args:
            outputs (dict): Dictionary of model outputs
        """
        self.logger.debug(f"Logging model outputs for batch {batch_idx}")
        for key, value in output_dict.items():

            if self.config.wandb:

                self.stream_table.log({
                        "sample_id":int(key),
                        "url":value["url"],
                        "class_1":value["match_001"]["class_name"],
                        "confidence_1":float(value["match_001"]["confidence"]),
                        "load_status":bool(value["loaded"]),
                    })
            
            self.logger.info(f"   Sample: {key} - URL: {value['url']}")
            for sample_key, sample_value in value.items():
                if sample_key == "url" or sample_key == "sample_id":
                    continue
                self.logger.info(f"      {sample_key}: {sample_value}")

        self.logger.debug("Model outputs logged")


    def init_wandb_logging(self):
        """Initialise wandb logging"""

        if self.config.wandb:
            import wandb
            import weave
            weave.use_frontend_devmode()
            from weave.monitoring import StreamTable
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                entity=self.config.wandb_entity,
                tags=self.config.wandb_tags,
            )
            self.logger.debug("Wandb logging initialised")

            self.stream_table = StreamTable(f"{self.config.wandb_entity}/{self.config.wandb_project}/logged_predictions")
    

    def log_metrics(self, metrics_dict, step):
        """Log the metrics to wandb

        Args:
            metrics_dict (dict): Dictionary of metrics
            step (int): Global step
        """

        if self.config.wandb:
            wandb.log(metrics_dict, step=step)

    
    def finish(self, total_time, num_samples):
        """Finish the logger

        Args:
            total_time (float): Total time taken
            num_samples (int): Number of samples
        """

        # Log the total time taken
        self.logger.info(f"Total time taken: {total_time:.2f} seconds")
        self.logger.info(f"Number of samples: {num_samples}")
        self.logger.info(f"Time taken per sample: {total_time/num_samples:.2f} seconds")

        # Finish wandb logging
        if self.config.wandb:
            self.wandb_run.save()
