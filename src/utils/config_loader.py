import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(path="src/config/config.yaml"):
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(e, exc_info=True)