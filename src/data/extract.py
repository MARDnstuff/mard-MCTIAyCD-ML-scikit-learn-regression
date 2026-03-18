import pandas as pd
import logging

logger = logging.getLogger(__name__)

def extract_data(path: str) -> pd.DataFrame:
    """
    Extract data from raw data path
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(e, exc_info=True)