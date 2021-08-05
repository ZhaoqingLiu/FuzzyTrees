"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import datetime
import errno
import logging
import logging.config
import os
from logging import FileHandler

import yaml


# =============================================================================
# Global configuration
# =============================================================================
def setup_logging(default_path='logging_config.yaml', default_level=logging.INFO):
    """
    Configure the logging module.

    How to enable global configuration for logging?

    Call this function at the beginning of the program's main function, and then
    use `logging` to get a logger wherever you need to log.

    How to log?

    In development, set the `level` of all handlers, e.g., `console`, `file`,
    `error`, etc., in the log configuration file to `DEBUG` for debugging
    purposes.

    Production systems, set the `level` of each handler in the log
    configuration file back to the levels appropriate for production systems,
    e.g., `console` to `INFO`, `file` to `DEBUG`, and `error` to `ERROR`.

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
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



