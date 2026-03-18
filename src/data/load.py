import logging
import os

logger = logging.getLogger(__name__)

def load_data(df, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Archivo guardado en: {path}")
    except Exception as e:
        logger.error(e, exc_info=True)