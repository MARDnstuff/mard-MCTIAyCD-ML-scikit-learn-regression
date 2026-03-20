from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score    
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def train(df: pd.DataFrame) -> None:
    try:
        # Filtramos las columnas más prometedoras
        df = df.copy()
        X = df.drop(columns=["target", "target_log"])
        Y = df["target_log"]

        features = [
            "EDAD_V",
            "P5_21",
            "P6_12",
            "EDAD_V_unknown",
            "P5_19_unknown",
            "P5_21_unknown",
            "P5_21_top",
            "P6_12_no_response",
            "P6_12_unknown",
            "P6_12_top",
            "P5_19A_unknown",
            "salario_mensual",
            "salario_mensual_unknown",
            "ratio_gasto_ingreso",
            "ahorro"
        ]


        X = df[features]

        # Dividmos 80 train y 20 test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        # Modelos en cuestión
        models = {
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            
            "SVR": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR())
            ])
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        results = {}

        logger.info("=== CROSS VALIDATION ===")
        for name, model in models.items():
            scores = cross_val_score(
                model,
                X_train,
                Y_train,
                cv=kf,
                scoring="neg_mean_squared_error"
            )

            rmse = np.sqrt(-scores.mean())
            results[name] = rmse

            logger.info(f"{name}: RMSE = {rmse:.2f}")

            best_model_name = min(results, key=results.get)
        best_model = models[best_model_name]

        logger.info(f"Best model: {best_model_name}")

        # -------------------------
        # 7. EVALUACIÓN FINAL
        # -------------------------
        best_model.fit(X_train, Y_train)
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(Y_test, y_pred)
        rmse = mean_squared_error(Y_test, y_pred) ** 0.5
        r2 = r2_score(Y_test, y_pred)

        logger.info("=== TEST METRICS ===")
        logger.info(f"MAE:  {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"R2:   {r2:.4f}")

    except Exception as e:
        logger.error(e, exc_info=True)