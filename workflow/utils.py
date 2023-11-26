import logging


def _build_logger():
    formatter = logging.Formatter("[EmbedGEM] %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger("embedgem")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


logger = _build_logger()
