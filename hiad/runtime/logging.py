from __future__ import annotations

import logging
import os
from pathlib import Path


def create_logger(
    name: str,
    log_file: str | os.PathLike[str],
    print_console: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """创建文件日志器，并避免同名日志器重复挂载 handler。

    Args:
        name (str): Python 日志器名称；同名日志器的旧 handler 会先关闭并移除。
        log_file (str | os.PathLike[str]): 日志文件路径，父目录会自动创建。
        print_console (bool): 是否同时向标准错误流输出同格式日志。
        level (int): ``logging`` 级别，默认 ``logging.INFO``。

    Returns:
        logging.Logger: 禁止向根日志器传播且已配置 handler 的日志器。

    Raises:
        OSError: 日志目录或文件无法创建、打开。
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d]%(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if print_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
