import logging
import sys

from config import settings

# Set logging level
logging.basicConfig(level=logging.INFO)

# Create common logger for the app
logger = logging.getLogger(__name__)

# Configure log output location
console_output = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(console_output)

if settings.LLMSMITH.DEBUG:
    logging.getLogger("llmsmith").addHandler(console_output)
    logging.getLogger("llmsmith").setLevel(logging.DEBUG)