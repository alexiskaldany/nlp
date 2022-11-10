from loguru import logger
import sys
from code.classical import Classical
from pathlib import Path

logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="DEBUG",
    backtrace=True,
    colorize=True,
)

INPUT_FILE = Path("")