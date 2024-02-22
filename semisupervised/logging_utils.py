import datetime
import logging
from functools import wraps

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

TIME_FORMAT = '%Y_%m_%d-%H_%M_%S'

def setup_logger(name, level=logging.INFO):
    handler = logging.FileHandler(build_log_file_name(name))
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)

    return logger

def build_log_file_name(name):
    return f'logs/{name}_{datetime.datetime.now().strftime(TIME_FORMAT)}.log'


def LogIndividualCallToFile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger(func.__name__)
        logger.log(logging.INFO, "Started")
        result = func(*args, **kwargs)
        logger.log(logging.INFO, "Finished")
        return result
    return wrapper