import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class LinearRegressionModel:
    """
    Wrapper de LinearRegression de sklearn.
    Normaliza features y target internamente con StandardScaler.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        self.target_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_prep = self.feature_pipeline.fit_transform(X_train)
        y_prep = self.target_pipeline.fit_transform(y_train)
        self.model.fit(X_prep, y_prep)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Devuelve predicciones en escala original (desnormalizadas)."""
        X_prep = self.feature_pipeline.transform(X)
        y_pred_prep = self.model.predict(X_prep)
        return self.target_pipeline.inverse_transform(y_pred_prep)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)
        return {
            "mae":    mean_absolute_error(y_test, y_pred),
            "rmse":   root_mean_squared_error(y_test, y_pred),
            "y_pred": y_pred,
        }
