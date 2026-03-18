import pandas as pd
import logging

logger = logging.getLogger(__name__)

def transform_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        df = df.copy()

        # -------------------------
        # 1. FILTRAR REGIONES
        # -------------------------
        regiones = config["data"]["regions"]
        df = df[df["REGION"].isin(regiones)]

        # -------------------------
        # 2. SELECCIONAR COLUMNAS
        # -------------------------
        columns = config["data"]["columns"]
        df = df[columns]

        # -------------------------
        # 3. RENOMBRAR TARGET
        # -------------------------
        target_col = config["data"]["target_column"]
        df = df.rename(columns={target_col: "target"})

        # -------------------------
        # 4. LIMPIEZA
        # -------------------------

        # eliminar duplicados
        df = df.drop_duplicates()

        # eliminar registros sin target (etiqueta)
        df = df.dropna(subset=["target"])

        # convertir a numérico (por si vienen como string)
        numeric_cols = df.columns.drop(["ENT", "REGION", "EDAD_V"])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------
        # 6. FEATURE ENGINEERING
        # -------------------------

        # mapear periodicidad → factor mensual
        period_map = {
            1: 4,   # semanal
            2: 2,   # quincenal
            3: 1,   # mensual
            4: 1/12,  # anual
        }

        # Se asume que P5_19 no reportado será tratado como 0 (no ingresos)
        df["P5_19"] = df["P5_19"].fillna(0)

        # Se asume com standar el periodo mensual
        df["P5_19A"] = df["P5_19A"].fillna(1)

        # Mapeamos dado el factor de periocidad
        df["factor"] = df["P5_19A"].map(period_map)

        # crear salario mensual
        df["salario_mensual"] = df["P5_19"] * df["factor"]

        logger.debug(f"Null values?: \n{df.isnull().sum()}")
        return df
    except Exception as e:
        logger.error(e, exc_info=True)