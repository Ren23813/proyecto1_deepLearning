"""
Uso:
    python EDA.py --input train.csv --outdir eda_outputs
O simplemente correr este mismo archivo desde VSCode
    
Este script:
  1. Carga el dataset y reporta dimensiones y tipos de variables.
  2. Calcula estadisticas descriptivas de variables numericas y categoricas.
  3. Identifica valores nulos y variables con posibles outliers.
  4. Genera visualizaciones (distribuciones, correlaciones, relacion con target).
  5. Imprime un resumen de decisiones de preprocesamiento sugeridas.

  
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

TARGET = "SalePrice"
ID_COL = "Id"


#funciones auxiliares
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset cargado desde '{path}'")
    return df


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def save_fig(fig, outdir: str, name: str) -> None:
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Imagen guardada en: {path}")



# 1. Dimensiones y tipos de variables
def section_overview(df: pd.DataFrame) -> tuple[list, list]:
    print("1. DIMENSIONES Y TIPOS DE VARIABLES")
    print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if ID_COL in numeric_cols:
        numeric_cols.remove(ID_COL)
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    print(f"\nVariables numericas ({len(numeric_cols)}): {numeric_cols}")
    print(f"\nVariables categoricas ({len(categorical_cols)}): {categorical_cols}")
    print(f"\nVariable objetivo: {TARGET}")

    #las que son números, pero realmente son categóricas
    pseudo_categorical = ["MSSubClass", "OverallQual", "OverallCond",
                           "MoSold", "YrSold"]
    pseudo_categorical = [c for c in pseudo_categorical if c in df.columns]
    print(f"\n Columnas numéricas que realmente son categoricas/ordinales: "
          f"{pseudo_categorical}")

    return numeric_cols, categorical_cols



# 2. Estadistica descriptiva
def section_descriptive_stats(df: pd.DataFrame, numeric_cols: list,
                               categorical_cols: list, outdir: str) -> None:
    print("\n 2. ESTADÍSTICA DESCRIPTIVA")

    print("\nVariables numéricas (incluyendo variable objetivo)")
    desc_num = df[numeric_cols + [TARGET]].describe().T
    desc_num["skew"] = df[numeric_cols + [TARGET]].skew()
    print(desc_num)
    desc_num.to_csv(os.path.join(outdir, "descriptive_stats_numeric.csv"))

    print("\n Variables categóricas (moda, num. categorias unicas) ")
    desc_cat = pd.DataFrame({
        "n_unique": df[categorical_cols].nunique(),
        "moda": df[categorical_cols].mode().iloc[0],
        "freq_moda": [df[c].value_counts().iloc[0] if df[c].notna().any() else np.nan
                      for c in categorical_cols],
    })
    print(desc_cat)
    desc_cat.to_csv(os.path.join(outdir, "descriptive_stats_categorical.csv"))



# 3. Valores nulos y outliers
def section_missing_and_outliers(df: pd.DataFrame, numeric_cols: list,
                                  outdir: str) -> pd.DataFrame:
    print("\n 3. VALORES NULOS Y OUTLIERS")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})
    missing_table = missing_table[missing_table["n_missing"] > 0].sort_values(
        "pct_missing", ascending=False
    )
    print(f"\nColumnas con valores nulos: {len(missing_table)}")
    print(missing_table)
    missing_table.to_csv(os.path.join(outdir, "missing_values.csv"))

    # visualizacion nulos
    if len(missing_table) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_table) * 0.3)))
        sns.barplot(x=missing_table["pct_missing"], y=missing_table.index, ax=ax,
                    color="steelblue")
        ax.set_xlabel("% de valores nulos")
        ax.set_title("Porcentaje de valores nulos por columna")
        save_fig(fig, outdir, "missing_values.png")

    # Deteccion de outliers via IQR para variables numericas
    print("\n Deteccion de outliers (fuera de [Q1-1.5*IQR, Q3+1.5*IQR]) ")
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_outliers

    outlier_df = pd.Series(outlier_summary, name="n_outliers").sort_values(ascending=False)
    outlier_df = outlier_df[outlier_df > 0]
    print(outlier_df)
    outlier_df.to_csv(os.path.join(outdir, "outlier_counts.csv"))

    return missing_table



# 4. Visualizaciones
def section_visualizations(df: pd.DataFrame, numeric_cols: list,
                            categorical_cols: list, outdir: str) -> None:
    print("\n 4. VISUALIZACIONES")

    # Distribución de variable objetivo
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Distribucion de {TARGET}")
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=axes[1], color="darkorange")
    axes[1].set_title(f"Distribucion de log1p({TARGET})")
    save_fig(fig, outdir, "target_distribution.png")

    # distribuciones de variables numericas (grid)
    n_cols_grid = 5
    n_rows_grid = int(np.ceil(len(numeric_cols) / n_cols_grid))
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(n_cols_grid * 3, n_rows_grid * 2.5))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), ax=axes[i], color="teal")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    save_fig(fig, outdir, "numeric_distributions.png")

    # Matriz de correlacion (numéricas + target)
    corr = df[numeric_cols + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Matriz de correlacion (variables numericas)")
    save_fig(fig, outdir, "correlation_matrix.png")

    # Top features numéricas más correlacionadas con el target
    top_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(10)
    print(f"\nTop 10 variables numericas mas correlacionadas con {TARGET}:")
    print(top_corr)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_corr.index):
        sns.scatterplot(x=df[col], y=df[TARGET], ax=axes[i], alpha=0.5, s=15)
        axes[i].set_title(f"{col} (corr={corr[TARGET][col]:.2f})", fontsize=9)
    fig.tight_layout()
    save_fig(fig, outdir, "top_features_vs_target.png")

    # Variables categoricas más relevantes vs target (boxplots)
    cat_candidates = [c for c in categorical_cols if df[c].nunique() <= 10]
    cat_candidates = cat_candidates[:8]
    if cat_candidates:
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()
        for i, col in enumerate(cat_candidates):
            order = df.groupby(col)[TARGET].median().sort_values().index
            sns.boxplot(x=col, y=TARGET, data=df, order=order, ax=axes[i])
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(axis="x", rotation=45)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        save_fig(fig, outdir, "categorical_vs_target.png")



# Main
def main():
    parser = argparse.ArgumentParser(description="EDA para el dataset de entrenamiento")
    parser.add_argument("--input", type=str, default="train.csv",
                         help="Ruta al CSV de entrenamiento")
    parser.add_argument("--outdir", type=str, default="eda_outputs",
                         help="Carpeta donde se guardan tablas y figuras")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = load_data(args.input)

    numeric_cols, categorical_cols = section_overview(df)
    section_descriptive_stats(df, numeric_cols, categorical_cols, args.outdir)
    section_visualizations(df, numeric_cols, categorical_cols, args.outdir)

    print("\nEDA completo. Revisar carpeta:", args.outdir)


if __name__ == "__main__":
    main()
