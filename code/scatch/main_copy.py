from loguru import logger
import sys
from classical import Classical
from pathlib import Path

logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="DEBUG",
    backtrace=True,
    colorize=True,
)

INPUT_FILE = Path("data/genesis_1.txt")
COMPARE_FILE = Path("data/genesis_2.txt")

genesis = Classical(INPUT_FILE.read_text(),spaCy_model="en_core_web_lg")

print(genesis.spacy_compare(COMPARE_FILE.read_text()))