"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 30/7/21 12:03 am
@desc  : 
@ref   : 
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
    - In development: Set the `level` of all handlers, e.g., `console`, `file`,
      `error`, etc., in the log configuration file to `DEBUG` for debugging
      purposes.
    - Production systems: Set the `level` of each handler in the log
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
    specifying the `Loader=...` parameter, has been deprecated [1]_.

    References
    ----------
    .. [1] https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


# =============================================================================
# Custom handlers
# =============================================================================
class MidnightRotatingFileHandler(FileHandler):
    """
    This class is used to solve the log loss problem caused by the multi-process
    mode. That is, when multiple processes write logs to the same log file, the
    archive logs rolled back by the previous worker may be overwritten by the
    rollback operation of the later worker, resulting in the loss of logs.

    How to automatically shard rollback log at midnight?
    1. The name of a log file ends with a date. Write logs of the current day to
       the log file that ends with the date of the current day.
    2. Every midnight, atomically create a new log file ending with the new date.
    So how to create files atomically?
    Open the log file using OS. O_CREAT|OS. O_EXCL mode. If the log file already
    exists, opening will fail.
    """
    def __init__(self, filename):
        self._filename = filename
        self._rotate_at = self._next_rotate_datetime()
        super().__init__(filename, mode='a')

    @staticmethod
    def _next_rotate_datetime():
        # rotate at midnight
        now = datetime.datetime.now()
        return now.replace(hour=0, minute=0, second=0) + datetime.timedelta(days=1)

    def _open(self):
        now = datetime.datetime.now()
        log_today = "%s.%s" % (self._filename, now.strftime('%Y-%m-%d'))
        try:
            # create the log file atomically
            fd = os.open(log_today, os.O_CREAT | os.O_EXCL)
            # if coming here, the log file was created successfully
            os.close(fd)
        except OSError as e:
            if e.errno != errno.EEXIST:
                # should not happen
                raise
        self.baseFilename = log_today
        return super()._open()

    def emit(self, record):
        now = datetime.datetime.now()
        if now > self._rotate_at:
            # time to rotate
            self._rotate_at = self._next_rotate_datetime()
            self.close()
        super().emit(record)


if __name__ == '__main__':
    # Usage example.
    setup_logging()
    # Start the main program, e.g.,
    # exp1()
