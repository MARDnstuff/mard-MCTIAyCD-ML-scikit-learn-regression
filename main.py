from src.config.logging import setUpLogging
from src.utils.config_loader import load_config
from src.data.extract import extract_data
from src.data.transform import transform_data
from src.data.load import load_data
from src.models.train import train
import logging


logger = logging.getLogger(__name__)

def main():
    setUpLogging()
    config = load_config()
    logger.info("Configuration has been loaded")

    # Extract
    df = extract_data(config["data"]["raw_path"])
    logger.info(f"Datos originales: {df.shape}")

    # Transform → ahora retorna dict { nombre: DataFrame }
    datasets = transform_data(df, config)
    logger.info(f"Bancos generados: {list(datasets.keys())}")

    # Load + Train → por cada banco de datos
    for name, df_clean in datasets.items():
        logger.info(f"--- Procesando banco: {name} ({df_clean.shape[0]} registros) ---")

        out_path = config["data"]["processed_path"].replace(".csv", f"_{name}.csv")
        load_data(df_clean, out_path)

        train(df_clean)


if __name__ == "__main__":
    main()