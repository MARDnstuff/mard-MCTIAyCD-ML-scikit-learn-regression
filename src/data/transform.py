import pandas as pd
import numpy as np
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

        # -------------------------
        #  EDAD_V
        # -------------------------
        df["EDAD_V_unknown"] = df["EDAD_V"] == 98
        df["EDAD_V"] = df["EDAD_V"].replace({
            98: np.nan,
            97: 97  # puedes dejarlo así o capearlo
        })

        # -------------------------
        #  P5_19
        # -------------------------
        df["P5_19_unknown"] = df["P5_19"] == 99999
        df["P5_19"] = df["P5_19"].replace({
            99999: np.nan
        })

        # -------------------------
        #  P5_21
        # -------------------------
        df["P5_21_unknown"] = df["P5_21"] == 99999
        df["P5_21"] = df["P5_21"].replace({
            99999: np.nan
        })
        df["P5_21_top"] = df["P5_21"] == 98000

        # -------------------------
        #  P6_12
        # -------------------------
        df["P6_12_no_response"] = df["P6_12"] == 99999888
        df["P6_12_unknown"] = df["P6_12"] == 99999999
        df["P6_12"] = df["P6_12"].replace({
            99999888: np.nan,
            99999999: np.nan
        })
        df["P6_12_top"] = df["P6_12"] == 98000000

        # -------------------------
        #  P6_13 - Target
        # -------------------------
        df["target_no_response"] = df["target"] == 999888
        df["target_unknown"] = df["target"] == 999999
        df["target"] = df["target"].replace({
            999888: np.nan,
            999999: np.nan
        })
        # Eliminamos valores nulos
        df = df.dropna(subset=["target"])

        df["target"] = df["target"].clip(upper=980000)
        # Este será el que se utilice para el entrenamiento
        df["target_log"] = np.log1p(df["target"])


        # convertir a numérico (por si vienen como string)
        numeric_cols = df.columns.drop(["ENT", "REGION"])
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

        df["P5_19A_unknown"] = df["P5_19A"].isna()


        # Mapeamos dado el factor de periocidad
        df["factor"] = df["P5_19A"].map(period_map)

        # crear salario mensual
        df["salario_mensual"] = df["P5_19"] * df["factor"]
        df["salario_mensual_unknown"] = df["salario_mensual"].isna()
        

        # Imputación por mediana
        # rellenamos los NaN con la mediana de la columna en cuestión
        # cuando no tengo todo, uso un valor típico de la distribución
        df["P5_21"] = df["P5_21"].fillna(df["P5_21"].median())
        df["P6_12"] = df["P6_12"].fillna(df["P6_12"].median())
        df["EDAD_V"] = df["EDAD_V"].fillna(df["EDAD_V"].median())
        df["salario_mensual"] = df["salario_mensual"].fillna(df["salario_mensual"].median())

        # Porcentaje del ingreso que gasta la persona
        df["ratio_gasto_ingreso"] = df["P5_21"] / (df["salario_mensual"] + 1)

        # Sobrante (ganancia) del individuo
        df["ahorro"] = df["salario_mensual"] - df["P5_21"]

        # Drop the columnas que no se van a utilizadar
        df = df.drop(columns=["P5_19", "P5_19A", "factor"])

        logger.debug(f"Null values?: \n{df.isnull().sum()}")
        return df
    except Exception as e:
        logger.error(e, exc_info=True)