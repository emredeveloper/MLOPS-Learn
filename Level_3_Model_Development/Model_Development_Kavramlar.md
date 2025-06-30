# Model Development Kavramlar Sözlüğü

Bu dosya Level 3: Model Development notebook'unda geçen tüm kavramları detaylı olarak açıklamaktadır.

## 📚 İçindekiler
- [Model Development Temelleri](#model-development-temelleri)
- [Model Selection ve Comparison](#model-selection-ve-comparison)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Cross-Validation](#cross-validation)
- [Model Ensembling](#model-ensembling)
- [Feature Importance ve Selection](#feature-importance-ve-selection)
- [Model Interpretability](#model-interpretability)
- [Performance Metrics](#performance-metrics)

---

## Model Development Temelleri

### 🧠 Model Development
**Tanım:** Veri bilimi projesinin kalbi olan model seçimi, eğitimi, optimizasyonu ve değerlendirmesi süreçlerinin tümünü kapsar.

**Ana Aşamaları:**
1. **Problem Definition:** İş probleminin ML problemine çevrilmesi
2. **Data Understanding:** Veri setinin analiz edilmesi
3. **Feature Engineering:** Özellik çıkarımı ve dönüştürme
4. **Model Selection:** Uygun algoritmaların seçilmesi
5. **Hyperparameter Tuning:** Model parametrelerinin optimizasyonu
6. **Model Validation:** Performans değerlendirmesi
7. **Model Selection:** En iyi modelin seçilmesi

### 🎯 Supervised Learning
**Tanım:** Labeled data kullanarak model eğitimi yapılan makine öğrenmesi yaklaşımı.

**Türleri:**
- **Classification:** Kategorik değişken tahmin etme
- **Regression:** Sürekli değişken tahmin etme

**Yaygın Algoritmalar:**
- Linear/Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines
- Neural Networks

### 📊 Training vs Test vs Validation
**Training Set (%60-80):** Modelin öğrendiği veri seti
**Validation Set (%10-20):** Hyperparameter tuning için kullanılan set
**Test Set (%10-20):** Final model performansını değerlendirmek için kullanılan set

```python
from sklearn.model_selection import train_test_split

# İlk split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# İkinci split: train vs validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)
```

---

## Model Selection ve Comparison

### 🔍 Model Comparison Framework
**Tanım:** Farklı modelleri sistematik ve objektif şekilde karşılaştırmak için kullanılan yapılandırılmış yaklaşım.

**Framework Bileşenleri:**
1. **Standardized Data Preparation:** Tüm modeller için aynı veri hazırlama
2. **Consistent Evaluation:** Aynı metrikler ve validation stratejisi
3. **Fair Comparison:** Aynı koşullarda model eğitimi
4. **Statistical Significance:** Sonuçların istatistiksel olarak anlamlı olması

### 🤖 Algorithm Families

#### Linear Models
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Linear Regression
lr = LinearRegression()

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=1.0)

# Elastic Net (L1 + L2 regularization)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

**Avantajları:**
- Hızlı eğitim
- Interpretable
- Low variance
- Good baseline

**Dezavantajları:**
- High bias
- Linear relationships only
- Feature engineering gerekli

#### Tree-Based Models
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Decision Tree
dt = DecisionTreeRegressor(max_depth=10)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
```

**Avantajları:**
- Non-linear relationships
- Feature interactions
- Robust to outliers
- No scaling required

**Dezavantajları:**
- Overfitting prone
- High variance (single trees)
- Black box (less interpretable)

#### Distance-Based Models
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# K-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=5)

# Support Vector Regression
svr = SVR(kernel='rbf', C=1.0)
```

**Avantajları:**
- Simple concept
- No assumptions about data distribution
- Good for complex decision boundaries

**Dezavantajları:**
- Computationally expensive
- Sensitive to scaling
- Curse of dimensionality

### 📊 Model Performance Comparison
```python
def compare_models(models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in models.items():
        # Model training
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    return pd.DataFrame(results).T
```

---

## Hyperparameter Tuning

### 🎛️ Hyperparameters vs Parameters
**Parameters:** Model tarafından öğrenilen değerler (weights, biases)
**Hyperparameters:** Model eğitimi öncesinde belirlenen konfigürasyonlar

**Yaygın Hyperparameters:**
- Learning rate
- Number of estimators
- Max depth
- Regularization strength
- Batch size
- Number of hidden layers

### 🔍 Hyperparameter Search Methods

#### Grid Search
**Tanım:** Tüm parametre kombinasyonlarını sistematik olarak test etme.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Avantajları:**
- Comprehensive search
- Guaranteed to find best combination
- Easy to implement

**Dezavantajları:**
- Computationally expensive
- Exponential time complexity
- Curse of dimensionality

#### Random Search
**Tanım:** Parametre uzayından rastgele örnekleme yaparak arama.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

**Avantajları:**
- More efficient than grid search
- Good for high-dimensional spaces
- Can find good solutions quickly

**Dezavantajları:**
- No guarantee of optimal solution
- May miss important regions

#### Bayesian Optimization
**Tanım:** Bayesian inference kullanarak akıllı parametre arama.

```python
# Optuna example
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, 
                           scoring='neg_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 🎯 Hyperparameter Tuning Strategies
1. **Start Simple:** Basit model ile başla
2. **One at a Time:** Tek parametre değiştir
3. **Coarse to Fine:** Geniş aralıktan başla, daralt
4. **Cross-Validation:** Always use CV for evaluation
5. **Computational Budget:** Time/resource constraints consider et

---

## Cross-Validation

### 🔄 Cross-Validation (CV)
**Tanım:** Model performansını güvenilir şekilde değerlendirmek için veri setini farklı şekillerde bölerek eğitim ve test yapma tekniği.

**Faydaları:**
- Robust performance estimation
- Reduced variance in estimates
- Better use of available data
- Overfitting detection

### 📊 CV Strategies

#### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, 
                        scoring='neg_mean_squared_error')

print(f"CV RMSE: {np.sqrt(-scores.mean()):.4f} ± {np.sqrt(scores.std()):.4f}")
```

#### Stratified K-Fold (Classification)
```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

#### Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, 
                        scoring='neg_mean_squared_error')
```

#### Leave-One-Out Cross-Validation
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, 
                        scoring='neg_mean_squared_error')
```

### 📈 Learning Curves
**Tanım:** Training set size'a karşı model performansının görselleştirilmesi.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

plt.plot(train_sizes, np.sqrt(-train_scores.mean(axis=1)), label='Training')
plt.plot(train_sizes, np.sqrt(-val_scores.mean(axis=1)), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.legend()
```

### 📊 Validation Curves
**Tanım:** Hyperparameter değerlerine karşı model performansının görselleştirilmesi.

```python
from sklearn.model_selection import validation_curve

param_range = [1, 5, 10, 20, 50, 100]
train_scores, val_scores = validation_curve(
    model, X, y, param_name='max_depth', param_range=param_range,
    cv=5, scoring='neg_mean_squared_error'
)

plt.plot(param_range, np.sqrt(-train_scores.mean(axis=1)), label='Training')
plt.plot(param_range, np.sqrt(-val_scores.mean(axis=1)), label='Validation')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.legend()
```

---

## Model Ensembling

### 🎭 Ensemble Methods
**Tanım:** Birden fazla modeli birleştirerek daha güçlü bir predictor oluşturma tekniği.

**Ensemble Türleri:**
1. **Bagging:** Bootstrap Aggregating (Random Forest)
2. **Boosting:** Sequential learning (Gradient Boosting)
3. **Voting:** Multiple model combinations
4. **Stacking:** Meta-learning approach

### 🗳️ Voting Ensemble
```python
from sklearn.ensemble import VotingRegressor

# Individual models
rf = RandomForestRegressor(n_estimators=100)
gb = GradientBoostingRegressor(n_estimators=100)
svr = SVR()

# Voting ensemble
voting_ensemble = VotingRegressor([
    ('rf', rf),
    ('gb', gb),
    ('svr', svr)
])

voting_ensemble.fit(X_train, y_train)
predictions = voting_ensemble.predict(X_test)
```

### 📚 Stacking Ensemble
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Base models
base_models = [
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor()),
    ('svr', SVR())
]

# Meta-learner
meta_learner = LinearRegression()

# Stacking ensemble
stacking_ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

stacking_ensemble.fit(X_train, y_train)
```

### 🎯 Blending
```python
# Manual blending
def blend_predictions(predictions_list, weights):
    """Weighted average of predictions"""
    weighted_preds = np.zeros_like(predictions_list[0])
    
    for i, preds in enumerate(predictions_list):
        weighted_preds += weights[i] * preds
    
    return weighted_preds

# Example usage
rf_preds = rf.predict(X_test)
gb_preds = gb.predict(X_test)
svr_preds = svr.predict(X_test)

blended_preds = blend_predictions(
    [rf_preds, gb_preds, svr_preds], 
    weights=[0.5, 0.3, 0.2]
)
```

### 🔧 Ensemble Best Practices
1. **Diversity:** Use diverse base models
2. **Quality:** Ensure base models are reasonably good
3. **Correlation:** Avoid highly correlated models
4. **Cross-Validation:** Use CV for ensemble validation
5. **Computational Cost:** Consider inference time

---

## Feature Importance ve Selection

### 📊 Feature Importance
**Tanım:** Her feature'ın model tahminlerine ne kadar katkıda bulunduğunun ölçülmesi.

### 🌲 Tree-Based Feature Importance
```python
# Random Forest feature importance
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.title('Random Forest Feature Importance')
```

### 🔄 Permutation Importance
**Tanım:** Her feature'ı shuffle ederek model performansındaki düşüşü ölçme.

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, random_state=42
)

perm_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)
```

### 🎯 Feature Selection Methods

#### Filter Methods
```python
# Correlation-based selection
correlation_matrix = X.corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j]
            ))

# Statistical tests
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

#### Wrapper Methods
```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE

rfe = RFE(
    estimator=RandomForestRegressor(),
    n_features_to_select=10
)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
```

#### Embedded Methods
```python
# LASSO regularization
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# Ridge regularization (for ranking)
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(cv=5)
ridge.fit(X, y)
feature_ranking = np.argsort(np.abs(ridge.coef_))[::-1]
```

---

## Model Interpretability

### 🔍 Model Interpretability
**Tanım:** Modelin nasıl karar verdiğinin anlaşılması ve açıklanması.

**Interpretability Levels:**
1. **Global Interpretability:** Modelin genel davranışını anlama
2. **Local Interpretability:** Specific predictions için açıklama
3. **Feature Importance:** Hangi feature'ların önemli olduğu
4. **Partial Dependence:** Feature'ların target üzerindeki etkisi

### 📊 SHAP (SHapley Additive exPlanations)
```python
import shap

# SHAP values calculation
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Local explanation for single prediction
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### 🍋 LIME (Local Interpretable Model-agnostic Explanations)
```python
import lime
import lime.lime_tabular

# LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    mode='regression'
)

# Local explanation
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict,
    num_features=10
)

explanation.show_in_notebook(show_table=True)
```

### 📈 Partial Dependence Plots
```python
from sklearn.inspection import PartialDependenceDisplay

# Single feature PDP
PartialDependenceDisplay.from_estimator(
    model, X, features=['feature_name']
)

# Two-feature interaction
PartialDependenceDisplay.from_estimator(
    model, X, features=[('feature1', 'feature2')]
)
```

---

## Performance Metrics

### 📊 Regression Metrics

#### Mean Squared Error (MSE)
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
```
**Formül:** `MSE = (1/n) * Σ(y_true - y_pred)²`
**Özellikler:** Büyük hataları penalize eder, outlier'lara duyarlı

#### Root Mean Squared Error (RMSE)
```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.4f}")
```
**Formül:** `RMSE = √MSE`
**Özellikler:** Orijinal birim cinsinden hata, interpretable

#### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
```
**Formül:** `MAE = (1/n) * Σ|y_true - y_pred|`
**Özellikler:** Outlier'lara daha robust, tüm hataları eşit ağırlık

#### R² (Coefficient of Determination)
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
```
**Formül:** `R² = 1 - (SS_res / SS_tot)`
**Özellikler:** Açıklanan varyans oranı, 0-1 arası (yüksek better)

---

## 🔗 İlgili Kaynaklar

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLxtend Library](http://rasbt.github.io/mlxtend/)
- [Feature Engineering Book](http://www.feat.engineering/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

---

*Bu döküman Level 3: Model Development notebook'unda geçen kavramları kapsamaktadır. Her kavram praktik örneklerle desteklenmiştir.* 