"""
Estrategia de particion de datos
---------------------------------
- Test (10%): se separa una sola vez al inicio y nunca se usa durante el
  grid search. Solo se evalua una vez, al final, con el modelo ganador.
- Train+Val (90% restante): grid search mediante K-Fold Cross Validation con K=9. 
  Con K=9, cada fold usa 1/9 (~10% del total) como validacion y 8/9 (~80% del total) como entrenamiento. 
  
  Se decidió esa distribución porque el dataset es realmente pequeño. 

Preprocesamiento (ver EDA.py para el análisis que lo justifica)
------------------------------------------------------------------
- Variables categoricas "de ausencia" (PoolQC, Alley, Fence, FireplaceQu,
  Garage*, Bsmt*, MiscFeature) = NaN se traduce como 'None' (la casa físicamente no
  tiene esa caracteristica (no tiene piscina, valla, etc.)), no como dato faltante real.
- Variables ordinales con escalas de calidad (ExterQual, KitchenQual,
  etc.) = ordinal encoding manual que preserva el orden.
- Variables categoricas nominales restantes -> one-hot encoding.
- LotFrontage = imputacion por mediana agrupada por Neighborhood.
- GarageYrBlt = imputacion con YearBuilt
- Variable objetivo -> log1p(SalePrice) para reducir el sesgo (curtosis) de
  la distribución; se revierte con expm1 antes de calcular el RMSE final
  en la escala original.
- Todas las features numéricas resultantes se escalan con StandardScaler
  (ajustado solo con datos de train) para que el MLP no sesge el
  aprendizaje hacia variables con magnitudes más grandes.

El pipeline completo de preprocesamiento se guarda (joblib) junto con el
modelo entrenado para que implementacion.py pueda aplicar exactamente las
mismas transformaciones al dataset de prueba, sin reajustar nada.

Uso:
    python training.py --input train.csv --outdir artifacts     #entranamiento completo, realmente pesado (dura como 1h con GPU)
    python training.py --input train.csv --quick        # grid pequeño, solo para probar que todo corre
"""

import argparse
import itertools
import json
import os
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TARGET = "SalePrice"
ID_COL = "Id"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Definicion de columnas (basado reflexión post EDA)
# ---------------------------------------------------------------------------

# Columnas categoricas donde NaN significa "la casa físicamente no posee esa característica"
ABSENCE_CAT_COLS = [
    "PoolQC", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "MiscFeature", "MasVnrType",
]

# Escalas ordinales (el valor más alto = mejor / más completo)
QUALITY_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS = {
    "ExterQual": QUALITY_SCALE,
    "ExterCond": QUALITY_SCALE,
    "BsmtQual": QUALITY_SCALE,
    "BsmtCond": QUALITY_SCALE,
    "HeatingQC": QUALITY_SCALE,
    "KitchenQual": QUALITY_SCALE,
    "FireplaceQu": QUALITY_SCALE,
    "GarageQual": QUALITY_SCALE,
    "GarageCond": QUALITY_SCALE,
    "PoolQC": QUALITY_SCALE,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "LandSlope": {"Gtl": 0, "Mod": 1, "Sev": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Street": {"Grvl": 0, "Pave": 1},
}

# Numéricas que en realidad son categóricas (sin orden fijo)
PSEUDO_CATEGORICAL_NOMINAL = ["MSSubClass", "MoSold"]

# Categóricas nominales (sin orden, van a one-hot porque no hay una mejor que otra, solo son nombres)
NOMINAL_COLS = [
    "MSZoning", "LotConfig", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "Foundation", "Heating", "Electrical", "GarageType",
    "MiscFeature", "SaleType", "SaleCondition", "Alley", "LandContour",
    "MasVnrType",
] + PSEUDO_CATEGORICAL_NOMINAL

# Numericas cuyo NaN significa "0" real (ej. no tiene sótano)
ZERO_FILL_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]



# ---------------------------------------------------------------------------
# Pipeline de preprocesamiento reproducible
# ---------------------------------------------------------------------------
class Preprocessor:
    """Ajusta SOLO con datos de train; transform() se reusa para
    validation/test y para el dataset de prueba final en implementacion.py.
    Se serializa con joblib junto con el modelo entrenado.
    """

    def __init__(self):
        self.neighborhood_lotfrontage_median_ = {}
        self.global_lotfrontage_median_ = None
        self.electrical_mode_ = None
        self.dummy_columns_ = None
        self.feature_columns_ = None
        self.scaler_ = StandardScaler()

    def _base_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])

        for col in ABSENCE_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        if "Electrical" in df.columns:
            df["Electrical"] = df["Electrical"].fillna(self.electrical_mode_)

        if "GarageYrBlt" in df.columns:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

        for col in ZERO_FILL_NUMERIC:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "LotFrontage" in df.columns:
            def fill_lf(row):
                if pd.notna(row["LotFrontage"]):
                    return row["LotFrontage"]
                return self.neighborhood_lotfrontage_median_.get(
                    row.get("Neighborhood"), self.global_lotfrontage_median_
                )
            df["LotFrontage"] = df.apply(fill_lf, axis=1)

        for col in PSEUDO_CATEGORICAL_NOMINAL:
            if col in df.columns:
                df[col] = df[col].astype(str)

        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(float)

        # Chequeo defensivo: cualquier columna de texto que no haya sido
        # mapeada (ordinal) ni vaya a NOMINAL_COLS (one-hot) es un bug de
        # configuracion (columna nueva/olvidada)
        remaining_object_cols = [
            c for c in df.columns[df.dtypes == object]
            if c not in NOMINAL_COLS and c != TARGET
        ]
        if remaining_object_cols:
            raise ValueError(
                f"Columnas de texto sin mapear. Agregar a ORDINAL_MAPS o a "
                f"NOMINAL_COLS): {remaining_object_cols}"
            )

        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.electrical_mode_ = df["Electrical"].mode(dropna=True).iloc[0] if "Electrical" in df.columns else None
        self.global_lotfrontage_median_ = df["LotFrontage"].median() if "LotFrontage" in df.columns else 0
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self.neighborhood_lotfrontage_median_ = (
                df.groupby("Neighborhood")["LotFrontage"].median().to_dict()
            )

        clean = self._base_clean(df)

        nominal_present = [c for c in NOMINAL_COLS if c in clean.columns]
        dummies = pd.get_dummies(clean[nominal_present], prefix=nominal_present)
        self.dummy_columns_ = dummies.columns.tolist()

        numeric_part = clean.drop(columns=nominal_present + ([TARGET] if TARGET in clean.columns else []))
        X = pd.concat([numeric_part, dummies], axis=1)
        self.feature_columns_ = X.columns.tolist()

        self.scaler_.fit(X.values)
        return self

    def transform(self, df: pd.DataFrame):
        clean = self._base_clean(df)

        nominal_present = [c for c in NOMINAL_COLS if c in clean.columns]
        dummies = pd.get_dummies(clean[nominal_present], prefix=nominal_present)
        dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0)

        numeric_part = clean.drop(columns=nominal_present + ([TARGET] if TARGET in clean.columns else []))
        X = pd.concat([numeric_part, dummies], axis=1)
        X = X.reindex(columns=self.feature_columns_, fill_value=0)

        X_scaled = self.scaler_.transform(X.values).astype(np.float32)

        y_log = None
        if TARGET in df.columns:
            y_log = np.log1p(df[TARGET].values).astype(np.float32)

        return X_scaled, y_log

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def inverse_target(y_log: np.ndarray) -> np.ndarray:
        return np.expm1(y_log)


# ---------------------------------------------------------------------------
# Modelo general
# ---------------------------------------------------------------------------
ACTIVATION_MAP = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        act_cls = ACTIVATION_MAP[activation]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Entrenamiento de un modelo (una combinacion de hiperparametros, un fold)
# ---------------------------------------------------------------------------
def rmse_original_scale(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    y_true = Preprocessor.inverse_target(y_log_true)
    y_pred = Preprocessor.inverse_target(y_log_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Optimizador no soportado: {name}")


def train_one_model(X_train, y_train, X_val, y_val, hp: dict,
                     max_epochs: int, patience: int, verbose: bool = False):
    """Entrena un MLP con early stopping sobre (X_val, y_val).
    Devuelve: best_state_dict, history, best_val_rmse, best_train_rmse, best_epoch
    """
    set_seed(SEED)
    input_dim = X_train.shape[1]
    model = MLP(input_dim, hp["hidden_dims"], hp["activation"], hp["dropout"]).to(DEVICE)
    optimizer = make_optimizer(hp["optimizer"], model.parameters(), hp["lr"], hp["weight_decay"])
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)

    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    X_train_full_t = torch.tensor(X_train).to(DEVICE)
    y_train_full_t = torch.tensor(y_train).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_full_t)
            train_loss = criterion(train_pred, y_train_full_t).item()
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:4d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred_log = model(X_train_full_t).cpu().numpy()
        val_pred_log = model(X_val_t).cpu().numpy()

    best_train_rmse = rmse_original_scale(y_train, train_pred_log)
    best_val_rmse = rmse_original_scale(y_val, val_pred_log)

    return best_state, history, best_val_rmse, best_train_rmse, best_epoch


# ---------------------------------------------------------------------------
# Grid search con K-Fold CV
# ---------------------------------------------------------------------------
FULL_PARAM_GRID = {
    "hidden_dims": [[64], [128, 64], [128, 64, 32], [256, 128, 64], [64, 32, 16]],
    "activation": ["relu", "tanh"],
    "dropout": [0.0, 0.2],
    "lr": [1e-2, 1e-3],
    "weight_decay": [0.0, 1e-4],
    "batch_size": [32],
    "optimizer": ["adam"],
}

QUICK_PARAM_GRID = {                            ## este se usa solo cuando se le pasa el parámetro --quick ; solo es de prueba
    "hidden_dims": [[32], [64, 32]],
    "activation": ["relu"],
    "dropout": [0.0],
    "lr": [1e-3],
    "weight_decay": [0.0],
    "batch_size": [32],
    "optimizer": ["adam"],
}


def generate_combinations(grid: dict) -> list:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def run_grid_search(X_tv: np.ndarray, y_tv: np.ndarray, grid: dict, k_folds: int,
                     max_epochs: int, patience: int, outdir: str,
                     n_combos: int | None = None) -> pd.DataFrame:
    combos = generate_combinations(grid)
    if n_combos is not None and n_combos < len(combos):
        rng = random.Random(SEED)
        combos = rng.sample(combos, n_combos)

    print(f"\n[Grid Search] {len(combos)} combinaciones x {k_folds}-fold CV "
          f"= {len(combos) * k_folds} entrenamientos totales.\n")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    log_path = os.path.join(outdir, "iteration_results.csv")
    results = []

    for i, hp in enumerate(combos, start=1):
        t0 = time.time()
        fold_train_rmses, fold_val_rmses, fold_best_epochs = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tv), start=1):
            X_train, X_val = X_tv[train_idx], X_tv[val_idx]
            y_train, y_val = y_tv[train_idx], y_tv[val_idx]

            _, _, val_rmse, train_rmse, best_epoch = train_one_model(
                X_train, y_train, X_val, y_val, hp, max_epochs, patience
            )
            fold_train_rmses.append(train_rmse)
            fold_val_rmses.append(val_rmse)
            fold_best_epochs.append(best_epoch)

        elapsed = time.time() - t0
        row = {
            "iteration_id": i,
            "hidden_dims": str(hp["hidden_dims"]),
            "activation": hp["activation"],
            "dropout": hp["dropout"],
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "batch_size": hp["batch_size"],
            "optimizer": hp["optimizer"],
            "k_folds": k_folds,
            "mean_train_rmse": float(np.mean(fold_train_rmses)),
            "std_train_rmse": float(np.std(fold_train_rmses)),
            "mean_val_rmse": float(np.mean(fold_val_rmses)),
            "std_val_rmse": float(np.std(fold_val_rmses)),
            "mean_best_epoch": float(np.mean(fold_best_epochs)),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(row)

        print(f"[{i}/{len(combos)}] hidden={hp['hidden_dims']} act={hp['activation']} "
              f"drop={hp['dropout']} lr={hp['lr']} wd={hp['weight_decay']} "
              f"-> val_RMSE={row['mean_val_rmse']:.1f} (+/-{row['std_val_rmse']:.1f}) "
              f"| {elapsed:.1f}s")

        # Guardado incremental: si algo se interrumpe, no se pierde el progreso.
        pd.DataFrame(results).to_csv(log_path, index=False)

    results_df = pd.DataFrame(results).sort_values("mean_val_rmse").reset_index(drop=True)
    results_df.to_csv(log_path, index=False)
    print(f"\n Resultados de todas las iteraciones guardados en: {log_path}")
    return results_df


# ---------------------------------------------------------------------------
# Entrenamiento final y guardado de artefactos
# ---------------------------------------------------------------------------
def plot_training_curve(history: dict, outdir: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train loss (MSE, log-target)")
    ax.plot(history["val_loss"], label="Val loss (MSE, log-target)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE (escala log)")
    ax.set_title("Curva de entrenamiento - modelo final")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del MLP")
    parser.add_argument("--input", type=str, default="train.csv")
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--kfolds", type=int, default=9,
                         help="K para K-Fold Cross-Validation dentro de train+val (default=9 -> ~80/10 por fold)")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n-combos", type=int, default=None,
                         help="Si se especifica, muestrea aleatoriamente N combinaciones "
                              "del grid completo en vez de correrlo todo (para acotar el tiempo)")
    parser.add_argument("--quick", action="store_true",
                         help="Usa un grid muy pequeño y pocas épocas, solo para verificar "
                              "que el pipeline completo corre sin errores; solo es de prueba")
    parser.add_argument("--skip-search", action="store_true",    ## Este fue agregado después de un error con el LR
                         help="No repite el grid search: carga iteration_results.csv de "
                              "--outdir (de una corrida previa) y solo reentrena/guarda "
                              "el modelo final con la mejor configuracion encontrada.")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"[OK] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Separar test (10%) una sola vez. No se vuelve a tocar hasta el final.
    df_trainval, df_test = train_test_split(df, test_size=args.test_size, random_state=SEED)
    print(f"Train+Val: {len(df_trainval)} filas | Test (held-out): {len(df_test)} filas")

    # Preprocesamiento: se ajusta solo con train+val, se aplica igual a test.
    prep = Preprocessor()
    X_tv, y_tv = prep.fit_transform(df_trainval)
    X_test, y_test = prep.transform(df_test)
    print(f"Numero de features tras encoding: {X_tv.shape[1]}")

    grid = QUICK_PARAM_GRID if args.quick else FULL_PARAM_GRID
    max_epochs = 20 if args.quick else args.max_epochs
    patience = 5 if args.quick else args.patience
    k_folds = 3 if args.quick else args.kfolds

    # Grid search con K-Fold CV dentro de train+val (o cargar resultados
    #    previos si --skip-search, para no todo el entrenamiento).
    results_path = os.path.join(args.outdir, "iteration_results.csv")
    if args.skip_search:
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"--skip-search requiere que exista {results_path} de una corrida previa."
            )
        results_df = pd.read_csv(results_path).sort_values("mean_val_rmse").reset_index(drop=True)
        print(f"\n[Skip search] Cargando resultados previos desde: {results_path}")
    else:
        results_df = run_grid_search(
            X_tv, y_tv, grid, k_folds, max_epochs, patience, args.outdir,
            n_combos=args.n_combos,
        )

    print("\n Top 5 combinaciones (menor RMSE de validacion):")
    print(results_df.head(5).to_string(index=False))

    # Reentrenar la mejor combinacion sobre todo train+val (con un split
    #    interno 90/10 solo para decidir el early stopping), y evaluar una
    #    sola vez sobre el test set nunca antes visto.
    best_row = results_df.iloc[0]
    best_hp = {
        "hidden_dims": eval(best_row["hidden_dims"]),
        "activation": best_row["activation"],
        "dropout": float(best_row["dropout"]),
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "batch_size": int(best_row["batch_size"]),
        "optimizer": best_row["optimizer"],
    }
    print(f"\n[Reentrenamiento final] Mejor configuracion: {best_hp}")

    X_fit, X_es, y_fit, y_es = train_test_split(X_tv, y_tv, test_size=0.10, random_state=SEED)
    best_state, history, final_val_rmse, final_train_rmse, best_epoch = train_one_model(
        X_fit, y_fit, X_es, y_es, best_hp, max_epochs=max_epochs * 2, patience=patience,
        verbose=True,
    )

    final_model = MLP(X_tv.shape[1], best_hp["hidden_dims"], best_hp["activation"], best_hp["dropout"]).to(DEVICE)
    final_model.load_state_dict(best_state)
    final_model.eval()
    with torch.no_grad():
        test_pred_log = final_model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    final_test_rmse = rmse_original_scale(y_test, test_pred_log)

    print(f"\n [RESULTADO FINAL]")
    print(f"  RMSE train (fit):        {final_train_rmse:.2f}")
    print(f"  RMSE val (early stop):   {final_val_rmse:.2f}")
    print(f"  RMSE TEST (held-out):    {final_test_rmse:.2f}")

    expected_rmse = best_row["mean_val_rmse"]
    if final_test_rmse > 3 * expected_rmse:
        print(f"\n ADVERTENCIA: El RMSE final ({final_test_rmse:.0f}) es mucho peor que "
              f"el esperado por el grid search ({expected_rmse:.0f}). Es probable que el "
              f"entrenamiento haya divergido (usualmente LR altos). Intente "
              f"correr de nuevo con --skip-search (usando otra semilla de split) o elija "
              f"manualmente una configuracion más estable de iteration_results.csv "
              f"(menor 'std_val_rmse').")

    plot_training_curve(history, args.outdir, "final_model_training_curve.png")

    # Guardar artefactos: modelo, preprocesador, config y metricas.
    torch.save({
        "model_state_dict": best_state,
        "input_dim": X_tv.shape[1],
        "hidden_dims": best_hp["hidden_dims"],
        "activation": best_hp["activation"],
        "dropout": best_hp["dropout"],
    }, os.path.join(args.outdir, "final_model.pt"))

    joblib.dump(prep, os.path.join(args.outdir, "preprocessor.joblib"))

    with open(os.path.join(args.outdir, "best_config.json"), "w") as f:
        json.dump({
            "hyperparameters": best_hp,
            "train_rmse": final_train_rmse,
            "val_rmse_early_stopping_split": final_val_rmse,
            "test_rmse_held_out": final_test_rmse,
            "best_epoch": best_epoch,
            "k_folds_used_in_search": k_folds,
        }, f, indent=2)

    print(f"\n Artefactos guardados en '{args.outdir}/':")
    print("  - final_model.pt          (pesos + arquitectura del MLP)")
    print("  - preprocessor.joblib     (pipeline de preprocesamiento ajustado)")
    print("  - best_config.json        (hiperparámetros y métricas finales)")
    print("  - iteration_results.csv   (tabla completa de resultados del grid search)")
    print("  - final_model_training_curve.png              (imagen del Train-loss y Val-loss)")


if __name__ == "__main__":
    main()