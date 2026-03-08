# 🏠 Kaggle — House Prices: Advanced Regression Techniques

![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-blue?logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)
![Status](https://img.shields.io/badge/Status-En%20progrés-yellow)

## 📋 Descripció

Competició de Kaggle per predir el preu de venda de cases a Ames, Iowa.  
**Tipus:** Regressió | **Mètrica:** RMSE logarítmic (log-RMSE)  
🔗 [Competició oficial](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📁 Estructura del projecte

```
kaggle-house-prices/
│
├── data/
│   ├── raw/            # Dades originals de Kaggle (NO al repo, vegeu .gitignore)
│   └── processed/      # Dades preprocessades
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_models.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
│
├── models/             # Models entrenats (.pkl) — NO al repo
├── submissions/        # Fitxers CSV de submissions — NO al repo
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Workflow

```
EDA → Neteja nuls → Feature Engineering → Encoding → Escalar → Model → Submit
```

---

## 📊 Resultats i submissions

| # | Model | CV log-RMSE | Public LB | Notes |
|---|-------|-------------|-----------|-------|
| 1 | Ridge (alpha=300) | 0.1166 | - | Baseline lineal |
| 2 | XGBoost tuned | 0.1137 | - | lr=0.01, depth=4 |
| 3 | Blend Ridge+XGBoost | 0.1100 | 0.1287 | 40% Ridge + 60% XGBoost |
| 4 | Blend Ridge+XGBoost | 0.1093 | 0.1275 | -50 features zero-importance|---

## 🛠️ Instal·lació

```bash
git clone https://github.com/EL_TEU_USER/kaggle-house-prices.git
cd kaggle-house-prices
pip install -r requirements.txt
```

### Descarregar les dades de Kaggle
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/raw/
```

---

## 📦 Dependències principals

- `pandas`, `numpy` — manipulació de dades
- `scikit-learn` — preprocessing i models
- `xgboost`, `lightgbm` — gradient boosting
- `matplotlib`, `seaborn` — visualització

---

## 📈 Aprenentatges clau

- [ ] EDA i detecció d'outliers
- [ ] Tractament de nuls semàntics (absent ≠ desconegut)
- [ ] Transformació log del target (skewness)
- [ ] Feature engineering amb 79 variables
- [ ] Regularització (Ridge/Lasso)
- [ ] Gradient Boosting (XGBoost/LightGBM)
- [ ] Stacking de models

---

## 📝 Notes

Projecte d'aprenentatge. Continua del repte [Titanic](https://www.kaggle.com/competitions/titanic).
