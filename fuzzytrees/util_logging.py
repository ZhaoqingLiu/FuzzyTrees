"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import logging
import logging.config
import os

import yaml


# =============================================================================
# Global configuration
# =============================================================================
def setup_logging(default_path='logging_config.yaml', default_level=logging.INFO):
    """
    Configure the logging module.

    Parameters
    ----------
    default_path : str
        File path to the in YAML document for configuring logging.

    default_level : {logging.CRITICAL, logging.FATAL, logging.ERROR,
                     logging.WARNING, logging.WARN, logging.INFO,
                     logging.DEBUG, logging.NOTSET}

    Warnings
    --------
    In PyYAML version 5.1+, the use of PyYAML's `yaml.load` function without
    specifying the `Loader=...` parameter, has been deprecated [6]_.

    References
    ----------
    .. [6] https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation

    Examples
    --------
    How to enable global configuration for logging?

    Call this function at the beginning of the program's main function, and then
    use `logging` to get a logger wherever you need to log.

    >>>filepath = "fuzzytrees/logging_config.yaml"
    >>>setup_logging(filepath)

    How to log?

    The root logger `root` in the log configuration file 'logging_config.yaml'
    is applicable to development debugging, and all logger within `loggers`,
    e.g., `main.core`, are applicable to production. In development, set the
    `level` of the handler `console` to `DEBUG`, and in production set it back
    to `INFO`. Customise new loggers as needed.

    For development dubugging:
    >>>logging.debug("This is a debugging message.")

    For production:
    >>>logger = logging.getLogger("main.core")
    >>>logger.debug("This is a debugging message.")
    >>>logger.info("This is a info message.")
    >>>logger.error("This is a error message.")
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



