from src.config.logging import setUpLogging
from src.utils.config_loader import load_config
from src.data.extract import extract_data
import logging
# from src.data.transform import transform_data
# from src.data.load import load_data

logger = logging.getLogger(__name__)

def main():
    setUpLogging()
    config = load_config()
    logger.info("Configuration has been loaded")

    # Extract
    df = extract_data(config["data"]["raw_path"])
    logger.info(f"Datos originales: {df.shape}")

    # # Transform
    # df_clean = transform_data(df)
    # print("Datos limpios:", df_clean.shape)

    # # Load
    # load_data(df_clean, config["data"]["processed_path"])

if __name__ == "__main__":
    main()