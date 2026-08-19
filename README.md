# Proyecto 1: Competencia de Modelación — CC3092

MLP en PyTorch para predecir `SalePrice` sobre el dataset de Ames Housing.
RMSE final (conjunto de test): **$26,728.04**

## Estructura del repositorio

```
.
├── EDA.py                          # Analisis exploratorio de datos
├── training.py                     # Preprocesamiento, grid search y entrenamiento del MLP
├── implementacion.py               # Carga el modelo ya entrenado y genera predicciones
├── train.csv                       # Dataset de entrenamiento (no incluido en el repo, ver abajo)
├── eda_outputs/                    # Figuras y tablas generadas por EDA.py
├── artifacts/                      # Modelo, preprocesador y resultados generados por training.py
│   ├── final_model.pt              # Pesos + arquitectura del MLP ganador
│   ├── preprocessor.joblib         # Pipeline de preprocesamiento ajustado (fit solo con train)
│   ├── best_config.json            # Hiperparametros y metricas del modelo final
│   └── iteration_results.csv       # Tabla completa de las 80 combinaciones del grid search
└── README.md
```


## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch joblib
```

Se recomienda GPU (CUDA) para `training.py` — el grid search completo (80
combinaciones x 9-fold CV = 720 entrenamientos) es significativamente mas
rapido con GPU. Los tres scripts detectan y usan CUDA automaticamente si
esta disponible; si no, corren en CPU sin cambios.

## Cómo reproducir los resultados

### 1. EDA

```bash
python EDA.py --input train.csv --outdir eda_outputs
```

Genera estadisticas descriptivas, analisis de nulos/outliers y
visualizaciones (distribucion del target, correlaciones, relacion de
features con `SalePrice`) en `eda_outputs/`.

### 2. Entrenamiento

```bash
python training.py --input train.csv --outdir artifacts
```

Este script:
- Separa un test set del 10% que nunca se usa durante la busqueda de
  hiperparametros.
- Corre un grid search (80 combinaciones de arquitectura/activacion/dropout/
  learning rate/weight decay) usando K-Fold CV (K=9) sobre el 90% restante,
  de forma que cada fold reproduce la proporcion 80/10 de train/val.
- Reentrena la mejor combinacion sobre todo el 90% y evalua una unica vez
  sobre el test set held-out.
- Guarda el modelo, el preprocesador y las metricas en `artifacts/`.

**Advertencia de tiempo:** la corrida completa puede tardar varias horas en
CPU (bastante menos en GPU). Opciones utiles:

```bash
# Prueba rapida (grid chico, pocas epocas) para verificar que todo corre:
python training.py --input train.csv --outdir artifacts --quick

# Muestrear solo N combinaciones del grid en vez de correrlo completo:
python training.py --input train.csv --outdir artifacts --n-combos 20

# Reentrenar solo el modelo final a partir de un iteration_results.csv
# ya existente, sin repetir el grid search:
python training.py --input train.csv --outdir artifacts --skip-search
```

### 3. Predicciones sobre datos nuevos

Para realizar predicciones en base a un CSV:

```bash
python implementacion.py --input <csv_de_prueba> --artifacts artifacts --output predictions.csv
```

Carga `final_model.pt` y `preprocessor.joblib` (sin reentrenar nada),
aplica exactamente el mismo pipeline de preprocesamiento ajustado en train,
y genera `predictions.csv` con columnas `Id, Prediction` en el mismo orden
del archivo de entrada.

## Resumen de resultados

| Configuracion ganadora | Valor |
|---|---|
| Arquitectura | `[128, 64]` |
| Activacion | ReLU |
| Dropout | 0.2 |
| Learning rate | 0.01 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Optimizador | Adam |

| Metrica | RMSE |
|---|---|
| Train | $16,421.19 |
| Validation (early stopping) | $26,346.33 |
| **Test (held-out)** | **$26,728.04** |
