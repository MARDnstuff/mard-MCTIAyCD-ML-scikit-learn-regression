import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _clean_and_engineer(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Limpieza y feature engineering común para cualquier subconjunto de datos.
    Recibe un DataFrame ya filtrado (por región o por entidad).
    """
    df = df.copy()

    # -------------------------
    # 1. SELECCIONAR COLUMNAS
    # -------------------------
    columns = config["data"]["columns"]
    df = df[columns]

    # -------------------------
    # 2. RENOMBRAR TARGET
    # -------------------------
    target_col = config["data"]["target_column"]
    df = df.rename(columns={target_col: "target"})

    # -------------------------
    # 3. LIMPIEZA
    # -------------------------
    df = df.drop_duplicates()
    df = df.dropna(subset=["target"])

    # EDAD_V
    df["EDAD_V_unknown"] = df["EDAD_V"] == 98
    df["EDAD_V"] = df["EDAD_V"].replace({98: np.nan, 97: 97})

    # P5_19
    df["P5_19_unknown"] = df["P5_19"] == 99999
    df["P5_19"] = df["P5_19"].replace({99999: np.nan})

    # P5_21
    df["P5_21_unknown"] = df["P5_21"] == 99999
    df["P5_21"] = df["P5_21"].replace({99999: np.nan})
    df["P5_21_top"] = df["P5_21"] == 98000

    # P6_12
    df["P6_12_no_response"] = df["P6_12"] == 99999888
    df["P6_12_unknown"] = df["P6_12"] == 99999999
    df["P6_12"] = df["P6_12"].replace({99999888: np.nan, 99999999: np.nan})
    df["P6_12_top"] = df["P6_12"] == 98000000

    # Target (P6_13)
    df["target_no_response"] = df["target"] == 999888
    df["target_unknown"] = df["target"] == 999999
    df["target"] = df["target"].replace({999888: np.nan, 999999: np.nan})
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].clip(upper=980000)
    df["target_log"] = np.log1p(df["target"])

    # Convertir a numérico
    numeric_cols = df.columns.drop(["ENT", "REGION"])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 4. FEATURE ENGINEERING
    # -------------------------
    period_map = {1: 4, 2: 2, 3: 1, 4: 1/12}

    df["P5_19A_unknown"] = df["P5_19A"].isna()
    df["factor"] = df["P5_19A"].map(period_map)
    df["salario_mensual"] = df["P5_19"] * df["factor"]
    df["salario_mensual_unknown"] = df["salario_mensual"].isna()

    # Imputación por mediana
    for col in ["P5_21", "P6_12", "EDAD_V", "salario_mensual"]:
        df[col] = df[col].fillna(df[col].median())

    df["ratio_gasto_ingreso"] = df["P5_21"] / (df["salario_mensual"] + 1)
    df["ahorro"] = df["salario_mensual"] - df["P5_21"]

    df = df.drop(columns=["P5_19", "P5_19A", "factor"])

    logger.debug(f"Null values?\n{df.isnull().sum()}")
    return df


def transform_data(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """
    Genera múltiples bancos de datos para experimentación:
        - Un dataset por cada región definida en config (ej. region_1, region_2)
        - Un dataset por cada entidad federativa (ENT) dentro de esas regiones
        - Un dataset combinado con todas las regiones ("region_todas")

    Retorna un diccionario:  { "nombre_banco": DataFrame, ... }
    """
    try:
        datasets = {}
        regiones = config["data"]["regions"]

        # Filtrar solo las regiones de interés
        df_regiones = df[df["REGION"].isin(regiones)].copy()

        # ------------------------------------------
        # A) Dataset combinado de todas las regiones
        # ------------------------------------------
        df_todas = _clean_and_engineer(df_regiones, config)
        datasets["region_todas"] = df_todas
        logger.info(f"[region_todas] → {df_todas.shape[0]} registros")

        # ------------------------------------------
        # B) Un dataset por cada región individual
        # ------------------------------------------
        for region in regiones:
            df_region = df[df["REGION"] == region].copy()
            if df_region.empty:
                logger.warning(f"[region_{region}] Sin datos, se omite.")
                continue
            df_clean = _clean_and_engineer(df_region, config)
            key = f"region_{region}"
            datasets[key] = df_clean
            logger.info(f"[{key}] → {df_clean.shape[0]} registros")

        # ------------------------------------------
        # C) Un dataset por cada entidad federativa
        #    (dentro de las regiones filtradas)
        # ------------------------------------------
        # Obtenemos las ENTs ANTES de limpiar para no perderlas por columnas
        entidades = df_regiones["ENT"].dropna().unique()

        for ent in sorted(entidades):
            df_ent = df_regiones[df_regiones["ENT"] == ent].copy()
            if df_ent.empty:
                logger.warning(f"[ent_{int(ent):02d}] Sin datos, se omite.")
                continue
            df_clean = _clean_and_engineer(df_ent, config)
            if df_clean.shape[0] < 10:
                logger.warning(
                    f"[ent_{int(ent):02d}] Solo {df_clean.shape[0]} registros, "
                    "podría ser insuficiente para entrenamiento."
                )
            key = f"ent_{int(ent):02d}"
            datasets[key] = df_clean
            logger.info(f"[{key}] → {df_clean.shape[0]} registros")

        logger.info(
            f"Bancos de datos generados: {list(datasets.keys())}"
        )
        return datasets

    except Exception as e:
        logger.error(e, exc_info=True)
        return {}