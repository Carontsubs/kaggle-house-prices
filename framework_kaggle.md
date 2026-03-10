# Framework d'Iteració de Model — Kaggle House Prices

## Cicle de treball
```
DIAGNOSTIC -> EDA DIRIGIT -> FEATURE ENGINEERING -> VALIDACIO -> (repetir)
```

---

## Pas 1 — Diagnòstic
Objectiu: Entendre on i com falla el model abans de fer cap canvi.

### 1.1 Residuals
| Tipus | Eina | Quan usar |
|---|---|---|
| Distribució | Histograma de residuals | Sempre — primer diagnòstic |
| Per rang de preu | Residuals vs preu predit (scatter) | Quan sospitem biaix per preu alt/baix |
| Casos extrems | Top 20 errors absoluts | Per identificar casos atípics |
| Temporal | Residuals vs YearBuilt | Quan cases antigues fallen més |
| Per barri | Residuals per Neighborhood (boxplot) | Quan sospitem efecte de barri |

```python
residuals = y - model.predict(X)
error_df = pd.DataFrame({'real': np.expm1(y), 'predit': np.expm1(pred), 'error': np.abs(residuals)})
error_df.sort_values('error', ascending=False).head(20)
```

### 1.2 SHAP Values
SHAP explica per què el model ha fet cada predicció concreta.
Diferència clau:
- Feature importance → quines features usa el model EN GENERAL
- SHAP → com afecta cada feature a CADA PREDICCIÓ CONCRETA

| Tipus SHAP | Funció | Quan usar |
|---|---|---|
| Summary Plot | shap.summary_plot() | Primer cop — panorama global |
| Local / Force Plot | shap.force_plot() | Per analitzar casos mal predits |
| Dependence Plot | shap.dependence_plot() | Per entendre relacions no lineals |
| Interaction Values | shap.interaction_values() | Quan sospitem interacció entre dues variables |
| Waterfall Plot | shap.waterfall_plot() | Per documentar un error concret |

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
```

---

## Pas 2 — EDA Dirigit
Objectiu: Un cop identificat on falla, tornar a les dades per entendre per què.
No és un EDA genèric — és focalitzat en els errors del Pas 1.

### 2.1 Heatmaps per Grups de Features
| Grup | Features | Objectiu |
|---|---|---|
| Superfícies | GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GarageArea | Detectar redundàncies entre m2 |
| Qualitats | OverallQual, ExterQual, KitchenQual, GarageQual, HeatingQC | Veure si totes aporten info nova |
| Temporals | YearBuilt, YearRemodAdd, GarageYrBlt | Efecte de l'edat i les reformes |
| Garatge | GarageCars, GarageArea, GarageFinish, GarageType | Possible redundància Cars vs Area |
| Soterrani | BsmtQual, TotalBsmtSF, BsmtFinSF1, BsmtExposure | Grup molt correlacionat internament |

### 2.2 Anàlisi del Con de Distribució
| Zona | Definició | Interpretació |
|---|---|---|
| Regressió lineal | np.polyfit(grau=1) | Diagnòstic ràpid — limitada per no-linearitats |
| Regressió parabòlica | np.polyfit(grau=2) | Millor per capturar creixement exponencial |
| Con interior 30% | y_pred +/- 0.3 * y_pred | Zona verda — mercat normal |
| Con exterior 60% | y_pred +/- 0.6 * y_pred | Zona groga — casos especials acceptables |
| Fora del con 60% | Punts fora del con exterior | Zona roja — anomalies, vendes atípiques |

### 2.3 Segmentació per Zona d'Error
Preguntes a respondre sobre els casos mal predits:
- Quins barris apareixen més?
- Quina SaleCondition tenen? (Normal, Abnorml, Family, Partial...)
- Quina franja d'edat (YearBuilt)?
- Quina qualitat general (OverallQual)?
- Hi ha una combinació recurrent de factors?

**Hipòtesis actives (House Prices):**
- Cases antigues sense reforma → probable venda familiar/atípica
- AnysSenseReforma = YearRemodAdd - YearBuilt: com més gran, més probable venda fora de mercat
- El venedor que vol preu de mercat s'esmera a reformar. El que no reforma vol vendre ràpid o a familiar

---

## Pas 3 — Feature Engineering
Objectiu: Crear features que capturin els patrons identificats. Eliminar les que aporten soroll.

**Regla fonamental:** la nova feature ha de superar en correlació la millor feature individual que la composa.

### 3.1 Crear Features Noves
| Tipus | Fórmula | Exemple | Quan usar |
|---|---|---|---|
| Suma | A + B | TotalSF = Bsmt + 1stFlr | Quan dos valors mesuren el mateix concepte |
| Producte | A * B | Qual_GrLivArea = OverallQual * GrLivArea | Quan hi ha interacció entre dues variables |
| Diferència | A - B | AnysSenseReforma = YearRemodAdd - YearBuilt | Per capturar canvi o distància temporal |
| Divisió | A / B | Preu_per_m2 = SalePrice / GrLivArea | Per normalitzar entre escales |
| Agrupació | Sum(binàries) | CheapNeighborhood = sum(barris barats) | Per combinar categories similars |
| Transformació | log1p(A) | log1p(LotArea) | Per reduir skewness |
| Booleana | (A > llindar).astype(int) | Has2ndFloor = (2ndFlrSF > 0) | Per convertir presència/absència en 0/1 |

### 3.2 Eliminar Features
| Mètode | Criteri | Limitació |
|---|---|---|
| Zero importance | XGBoost importance == 0 | Pot eliminar features útils per Ridge |
| Baixa importància | importance < llindar (ex: 0.001) | El sweet spot cal trobar-lo per CV |
| Alta correlació | Correlació entre features > 0.9 | Multicolinealitat — quedar la més interpretable |
| Correlació amb target | abs(corr SalePrice) < 0.1 | Criteri conservador per features molt febles |

**Paradoxa fonamental:**
- Ridge → funciona millor amb MOLTES features (la regularització filtra el soroll)
- XGBoost → funciona millor amb POQUES features ben escollides
- Blend Ridge+XGBoost → sweet spot entre els dos extrems

---

## Pas 4 — Validació
Objectiu: Mesurar si el canvi ha millorat el model de forma honesta.

### 4.1 Mètriques Clau
| Mètrica | Valor actual | Interpretació |
|---|---|---|
| CV log-RMSE (5 folds) | 0.1090 | Score local — qualitat del model |
| Public Leaderboard | 0.12754 | Score Kaggle — test real amb dades noves |
| Gap CV/Kaggle | 0.018 | Overfitting — com més gran, més memoritza el train |
| Std entre folds | 0.007 | Estabilitat del model |

### 4.2 Criteris de Submission
- Només submetre si el CV millora — mai per "veure què passa"
- Màxim 10 submissions per dia — cada una ha de tenir una hipòtesi clara
- Documentar sempre: model, features, CV score i resultat Kaggle
- Confiar més en el CV que en el Public LB (usa només el 30% del test)

### 4.3 Historial de Resultats
| # | Model | Features | CV | Public LB | Notes |
|---|---|---|---|---|---|
| 1 | Blend Ridge+XGBoost | 249 | 0.1100 | 0.12865 | Baseline blend |
| 2 | Blend Ridge+XGBoost | 199 (zero imp.) | 0.1093 | 0.12754 | Millor actual |
| 3 | Blend Ridge+XGBoost | 110 (imp>0.001) | 0.1090 | 0.12839 | CV millora, LB empitjora |

---

## Pendents i Hipòtesis

### EDA Pendent
- Heatmaps per grups (superfícies, qualitats, temporals, garatge, soterrani)
- Regressió parabòlica + cons de normalitat (30% i 60%) sobre GrLivArea vs SalePrice
- Validar hipòtesi: cases antigues sense reforma correlacionen amb SaleCondition Abnorml/Family

### Features Pendents
- AnysSenseReforma = YearRemodAdd - YearBuilt
- AnysSenseReforma * OverallCond (interacció temps sense reforma + condició)
- Zona del con: feature categòrica (0=normal, 1=especial, 2=anomalia)

### Models Pendents
- SHAP analysis complet (summary plot + local per casos extrems)
- SVR — model no lineal que pot capturar la corba parabòlica
- ElasticNet — combinació Ridge + Lasso
- Stacking — metamodel que aprèn els pesos òptims entre Ridge i XGBoost
- XGBoost amb 20-30 features + Ridge amb 199 features en blend separat

---

## Regles d'Or
1. La IA genera codi. Tu raones sobre les dades. Sense interpretació, el codi no val res.
2. Una feature ben pensada val més que 10 features generades automàticament.
3. El model aprèn correlacions. Tu entens causalitat. Aquesta diferència és el teu valor.
4. Més features no sempre és millor — el model òptim sol tenir 20-30 features molt ben escollides.
5. El gap CV/Kaggle és l'indicador més honest de la qualitat real del model.
6. Cada submission ha de tenir una hipòtesi clara darrere. Mai submetre per curiositat.
