import os
import time
import logging
import functools
import traceback
from datetime import datetime
from typing import Any, Callable, List, Dict, Optional, Type, Union, Tuple
from loguru import logger

# Create logs directory
os.makedirs("logs", exist_ok=True)

class RetryExhausted(Exception):
    """Exception raised when all retry attempts are exhausted"""
    pass

def retry(max_tries=3, delay=1, backoff=2, exceptions=(Exception,), logger=None):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_tries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (e.g. value of 2 will double the delay each retry)
        exceptions: Tuple of exceptions to catch and retry on
        logger: Logger instance to use for logging retries
    
    Returns:
        The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay
            last_exception = None
            
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    mtries -= 1
                    if mtries == 0:
                        break
                        
                    msg = f"Retrying '{func.__name__}' in {mdelay:.2f}s: {str(e)}"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                        
                    time.sleep(mdelay)
                    mdelay *= backoff
            
            # If we get here, all retries have been exhausted
            error_msg = f"All {max_tries} retry attempts exhausted for '{func.__name__}'"
            if logger:
                logger.error(error_msg)
                logger.error(f"Last exception: {last_exception}")
                logger.error(traceback.format_exc())
            raise RetryExhausted(error_msg) from last_exception
            
        return wrapper
    return decorator

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console output
    
    Args:
        name: The logger name
        log_file: The log file path (default: logs/{name}_{date}.log)
        level: The logging level
        
    Returns:
        The configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up default log file if not specified
    if log_file is None:
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/{name}_{date_str}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def graceful_degradation(fallback_result, log_error=True, logger=None):
    """
    Decorator for graceful degradation of functionality
    
    Args:
        fallback_result: The result to return if the function fails
        log_error: Whether to log the error
        logger: The logger to use for logging
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    error_msg = f"Function '{func.__name__}' failed, using fallback: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                    else:
                        print(error_msg)
                return fallback_result
        return wrapper
    return decorator

class SafeDict(dict):
    """
    Dictionary that returns a default value for missing keys instead of raising KeyError
    """
    def __init__(self, *args, default_value=None, **kwargs):
        self.default_value = default_value
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_value

def log_execution_time(logger=None):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger instance to use for logging
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            log_msg = f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
                
            return result
        return wrapper
    return decorator