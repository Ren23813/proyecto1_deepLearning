"""
No reentrena nada: solo carga artefactos ya guardados y predice.

Uso:
    python implementacion.py --input pipeline_test.csv --artifacts artifacts --output predictions.csv

Salida:
    Un CSV con exactamente dos columnas: Id, Prediction
"""

import argparse
import os

import joblib
import pandas as pd
import torch


from training import MLP, Preprocessor  

ID_COL = "Id"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts(artifacts_dir: str):
    model_path = os.path.join(artifacts_dir, "final_model.pt")
    prep_path = os.path.join(artifacts_dir, "preprocessor.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontro '{model_path}'. Corre training.py primero o verifica la carpeta.")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"No se encontro '{prep_path}'. Corre training.py primero o verifica la carpeta.")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        activation=checkpoint["activation"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocessor: Preprocessor = joblib.load(prep_path)

    print(f"Modelo cargado desde '{model_path}' "
          f"(arquitectura: {checkpoint['hidden_dims']}, activacion: {checkpoint['activation']})")
    print(f"Preprocesador cargado desde '{prep_path}'")

    return model, preprocessor


def predict(model: MLP, preprocessor: Preprocessor, df: pd.DataFrame) -> pd.Series:
    # El dataset de prueba no tiene SalePrice, asi que transform() solo
    # devuelve X (y_log será "None"), usando exactamente las mismas medianas,
    # mapeos ordinales, columnas dummy y escalador ajustados en training.py
    X, _ = preprocessor.transform(df)

    with torch.no_grad():
        y_log_pred = model(torch.tensor(X).to(DEVICE)).cpu().numpy()

    y_pred = Preprocessor.inverse_target(y_log_pred)
    return pd.Series(y_pred, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones con el modelo final entrenado")
    parser.add_argument("--input", type=str, required=True,
                         help="CSV de prueba (mismas columnas que train.csv, sin SalePrice)")
    parser.add_argument("--artifacts", type=str, default="artifacts",
                         help="Carpeta con final_model.pt y preprocessor.joblib")
    parser.add_argument("--output", type=str, default="predictions.csv",
                         help="Ruta del CSV de salida (columnas: Id, Prediction)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Dataset de prueba cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    if ID_COL not in df.columns:
        raise ValueError(f"El CSV de entrada debe tener una columna '{ID_COL}'.")

    model, preprocessor = load_artifacts(args.artifacts)

    predictions = predict(model, preprocessor, df)

    output_df = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "Prediction": predictions.values,
    })
    output_df.to_csv(args.output, index=False)

    print(f"\n Predicciones guardadas en '{args.output}'")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
