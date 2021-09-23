'''Repository of common utils '''

import logging
from time import time
from functools import wraps


#####################################
# Logging function
#####################################

def logging_creation(logger_name):
    """
    create logger object and initialize console handlers and log formats
    
    Parameters
    ----------
    logger_name : string
       The logger file name with its absolute path


    Returns
    -------
    logger object: logger
    """
    logger = logging.getLogger(logger_name.split("\\")[-1])
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_name + '.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def timer(func, logger):
    """This decorator prints the execution time for the decorated function.
    
    Parameters
    ----------
    func : callable python function 
           A python function to be wrapped in timer decorator
    
    logger : logger object
             The logger file name with its absolute path


    Returns
    -------
    wrapper: python wrapper function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper
