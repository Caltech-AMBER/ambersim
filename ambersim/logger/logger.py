import jax
import wandb
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """Base logger interface that defines common methods for logging metrics and parameters.

    Attributes:
        log_dir (str): Directory where logs are stored. If None, default locations are used.
    """

    def __init__(self, log_dir=None):
        """Initializes the BaseLogger with a specified log directory.

        Args:
            log_dir (str): Directory to store the logs. If None, uses default log directory.
        """
        self.log_dir = log_dir

    def log_metric(self, key, value, step=None):
        """Logs a metric value.

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step number at which the metric is logged.
        """
        raise NotImplementedError

    def log_params(self, params):
        """Logs parameters.

        Args:
            params (dict): A dictionary containing parameter names and their values.
        """
        raise NotImplementedError

    def log_progress(self, step, state_info):
        """Logs the state of a process using the log_metric method.

        Args:
            state_info (dict): A dictionary containing state information.
            step (int, optional): The step number at which the state is logged.
        """
        for key, value in state_info.items():
            if isinstance(value, jax.Array):
                value = float(value)  # we need floats for logging
            self.log_metric(key, value, step)


class TensorBoardLogger(BaseLogger):
    """Logger that implements logging functionality using TensorBoard.

    Inherits from BaseLogger and implements its methods for TensorBoard specific logging.
    """

    def __init__(self, log_dir=None):
        """Initializes the TensorBoardLogger with a specified log directory.

        Args:
            log_dir (str): Directory to store TensorBoard logs. If None, uses default log directory.
        """
        super().__init__(log_dir)
        self.writer = SummaryWriter(log_dir)

    def log_metric(self, key, value, step=None):
        """Logs a metric to TensorBoard.

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step number at which the metric is logged.
        """
        self.writer.add_scalar(key, value, step)

    def log_params(self, params):
        """Logs parameters to TensorBoard.

        Args:
            params (dict): A dictionary of parameters to log.
        """
        self.writer.add_hparams(params)


class WandbLogger(BaseLogger):
    """Logger that implements logging functionality using Weights & Biases (wandb).

    Inherits from BaseLogger and implements its methods for wandb specific logging.
    """

    def __init__(self, log_dir=None, project_name=None, config_dict=None):
        """Initializes the WandbLogger with a specified log directory and project name.

        Args:
            log_dir (str): Directory to store local wandb logs. If None, uses default wandb directory.
            project_name (str): Name of the wandb project. If None, a default project is used.
        """
        super().__init__(log_dir)
        wandb.init(dir=log_dir, project=project_name, config=config_dict)

    def log_metric(self, key, value, step=None):
        """Logs a metric to wandb.

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step number at which the metric is logged.
        """
        wandb.log({key: value})

    def log_params(self, params):
        """Logs parameters to wandb.

        Args:
            params (dict): A dictionary of parameters to log.
        """
        wandb.config.update(params)


class LoggerFactory:
    """Factory class to create logger instances based on specified logger type.

    Supports creation of different types of loggers like TensorBoardLogger and WandbLogger.
    """

    @staticmethod
    def get_logger(logger_type, log_dir=None):
        """Creates and returns a logger instance based on the specified logger type.

        Args:
            logger_type (str): The type of logger to create ('tensorboard' or 'wandb').
            log_dir (str, optional): Directory to store the logs. Specific to the logger type.

        Returns:
            BaseLogger: An instance of the requested logger type.

        Raises:
            ValueError: If an unsupported logger type is specified.
        """
        if logger_type == "tensorboard":
            return TensorBoardLogger(log_dir)
        elif logger_type == "wandb":
            return WandbLogger(log_dir)
        else:
            raise ValueError("Unsupported logger type")
