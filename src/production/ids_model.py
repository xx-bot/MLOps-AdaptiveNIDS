# src/ids_model.py
import json

import mlflow.pyfunc
import pandas as pd
import xgboost as xgb

# class to load model and make predictions
class IDSModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Called once when the model is loaded; `context.artifacts` maps names to local paths.
        self.model = xgb.XGBClassifier()
        self.model.load_model(context.artifacts["xgb_model"])

        with open(context.artifacts["encoding_maps"]) as f:
            maps = json.load(f)
        self.freq_maps = maps["freq_maps"]          # {"proto": {...}, "state": {...}, "service": {...}}
        self.feature_names = maps["feature_names"]  # exact column order used in training

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, mapping in self.freq_maps.items():
            out[f"{col}_freq"] = out[col].map(mapping).fillna(0)  # unseen category -> 0
        out = out.drop(columns=list(self.freq_maps))
        return out[self.feature_names]  # enforce training column order

    def predict(self, context, model_input: pd.DataFrame, params=None):
        X = self._encode(model_input)
        return self.model.predict(X)