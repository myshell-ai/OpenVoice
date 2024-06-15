import logging

from ctranslate2 import _ext

_PYTHON_TO_CT2_LEVEL = {
    logging.CRITICAL: _ext.LogLevel.Critical,
    logging.ERROR: _ext.LogLevel.Error,
    logging.WARNING: _ext.LogLevel.Warning,
    logging.INFO: _ext.LogLevel.Info,
    logging.DEBUG: _ext.LogLevel.Debug,
    logging.NOTSET: _ext.LogLevel.Trace,
}

_CT2_TO_PYTHON_LEVEL = {v: k for k, v in _PYTHON_TO_CT2_LEVEL.items()}


def set_log_level(level: int):
    """Sets the CTranslate2 logging level from a Python logging level.

    Arguments:
      level: A Python logging level.

    Example:

        >>> import logging
        >>> ctranslate2.set_log_level(logging.INFO)

    Note:
       The argument is a Python logging level for convenience, but this function
       controls the C++ logs of the library.
    """
    ct2_level = _PYTHON_TO_CT2_LEVEL.get(level)
    if ct2_level is None:
        raise ValueError("Level %d is not a valid logging level" % level)
    _ext.set_log_level(ct2_level)


def get_log_level() -> int:
    """Returns the current logging level.

    Returns:
      A Python logging level.
    """
    ct2_level = _ext.get_log_level()
    return _CT2_TO_PYTHON_LEVEL[ct2_level]
