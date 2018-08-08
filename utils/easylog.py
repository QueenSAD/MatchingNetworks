import os
import time
import logging
import logging.handlers

def get_log():
    log_file = time.strftime("./log/%Y_%m_%d_%H_%M_%S.log",
                                    time.localtime())
    dir_path = os.path.dirname(log_file)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception as e:
        pass
    handler = logging.handlers.RotatingFileHandler(log_file)
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger_instance = logging.getLogger('logs')
    logger_instance.addHandler(handler)
    logger_instance.setLevel(logging.DEBUG)
    return logger_instance
        
