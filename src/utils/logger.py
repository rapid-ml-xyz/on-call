import logging


def get_logger(name="oncall"):
    """
    Returns a standard Python logger.
    You can configure log level, formatting, or route logs to external systems.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
