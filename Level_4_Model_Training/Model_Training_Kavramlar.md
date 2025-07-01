# Level 4: Model Training - Kavramlar

## 🎯 Advanced Model Training Kavramları

### 1. Cross-Validation Stratejileri

#### K-Fold Cross-Validation
- **Tanım**: Veri setini k eşit parçaya böler, k-1 parça ile eğitir, 1 parça ile test eder
- **Avantajlar**: Objektif model değerlendirmesi, overfitting tespiti
- **Kullanım**: Genel regresyon ve sınıflandırma problemleri

#### Stratified K-Fold
- **Tanım**: Sınıf dağılımını koruyarak veri setini böler
- **Avantajlar**: Dengesiz veri setlerinde güvenilir sonuçlar
- **Kullanım**: Sınıflandırma problemleri

#### Time Series Split
- **Tanım**: Zaman serisinde geçmişten geleceğe doğru split yapar
- **Avantajlar**: Temporal dependency'leri korur
- **Kullanım**: Zaman serisi tahmin problemleri

### 2. Hyperparameter Optimization

#### Grid Search
- **Tanım**: Tüm parametre kombinasyonlarını sistematik olarak dener
- **Avantajlar**: Garantili en iyi kombinasyon bulma
- **Dezavantajlar**: Exponentially artan hesaplama maliyeti
- **Kullanım**: Az parametreli modellerde

#### Random Search
- **Tanım**: Rastgele parametre kombinasyonları dener
- **Avantajlar**: Grid search'ten daha hızlı, iyi sonuçlar
- **Dezavantajlar**: Optimal sonuç garantisi yok
- **Kullanım**: Çok parametreli modellerde

#### Bayesian Optimization
- **Tanım**: Geçmiş deneyimleri kullanarak akıllı parametre seçimi
- **Avantajlar**: En az deneme ile en iyi sonuç
- **Tools**: Optuna, Hyperopt, Gaussian Processes
- **Kullanım**: Pahalı hesaplama gerektiren modellerde

### 3. Ensemble Methods

#### Voting
- **Hard Voting**: Çoğunluk oyuyla karar (sınıflandırma)
- **Soft Voting**: Olasılık ortalaması (sınıflandırma)
- **Averaging**: Tahmin ortalaması (regresyon)
- **Weighted Voting**: Ağırlıklı oylama

#### Stacking
- **Tanım**: Base modellerin çıktılarını meta-model ile öğrenir
- **Avantajlar**: Farklı model tiplerinin güçlü yanlarını birleştirir
- **Meta-learner**: Genelde basit model (Ridge, Linear)
- **Cross-validation**: Base model tahminlerini oluşturmak için

#### Blending
- **Tanım**: Holdout set ile meta-model eğitimi
- **Fark**: Stacking'den daha basit, daha az CV gerektirir
- **Avantajlar**: Hızlı uygulama, overfitting riski düşük
- **Kullanım**: Büyük veri setlerinde

### 4. Model Diagnostics

#### Learning Curves
- **Training Curve**: Training set boyutuna göre performans
- **Validation Curve**: Validation performansı karşılaştırması
- **Overfitting Detection**: Training-validation gap analizi
- **Underfitting Detection**: Her iki curve'ün de kötü performansı

#### Validation Curves
- **Tanım**: Tek parametrenin farklı değerlerinde performans
- **Bias-Variance Trade-off**: Optimal parametre noktası
- **Model Complexity**: Parametrenin model karmaşıklığına etkisi

#### Performance Metrics
- **Regression**: RMSE, MAE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC
- **Cross-validation**: Mean ± Std deviation

### 5. Model Versioning ve Tracking

#### MLflow Components
- **Tracking**: Experiment ve run yönetimi
- **Projects**: Reproducible runs
- **Models**: Model packaging ve deployment
- **Registry**: Model versioning ve lifecycle

#### Experiment Tracking
- **Parameters**: Model hyperparameters
- **Metrics**: Performance metrics
- **Artifacts**: Model files, plots, data
- **Tags**: Metadata ve organizasyon

#### Model Registry
- **Versioning**: Model versiyonları takibi
- **Stage Management**: None, Staging, Production, Archived
- **Model Lineage**: Model gelişim geçmişi
- **Collaboration**: Team çalışması desteği

### 6. Advanced Training Strategies

#### Regularization
- **L1 (Lasso)**: Feature selection, sparse models
- **L2 (Ridge)**: Weight shrinkage, smooth models
- **Elastic Net**: L1 + L2 kombinasyonu
- **Dropout**: Neural network regularization

#### Early Stopping
- **Tanım**: Validation loss artmaya başladığında eğitimi durdurma
- **Patience**: Kaç epoch beklenecek
- **Restore Best**: En iyi model ağırlıklarını geri yükleme
- **Monitoring**: Validation metric izleme

#### Data Augmentation
- **Image**: Rotation, flip, crop, noise
- **Text**: Synonym replacement, back-translation
- **Time Series**: Jittering, scaling, warping
- **Tabular**: SMOTE, noise injection

### 7. Performance Optimization

#### Feature Engineering
- **Selection**: Önemli feature'ları seçme
- **Creation**: Yeni feature'lar türetme
- **Transformation**: Scaling, encoding
- **Interaction**: Feature kombinasyonları

#### Model Selection
- **Complexity**: Model karmaşıklığı vs performance
- **Interpretability**: Açıklanabilirlik gereksinimleri
- **Speed**: Training ve inference hızı
- **Memory**: Model boyutu ve RAM kullanımı

#### Pipeline Optimization
- **Preprocessing**: Efficient data transformation
- **Parallel Processing**: Multi-core kullanımı
- **Memory Management**: RAM optimizasyonu
- **Caching**: Intermediate results kaydetme

### 8. Best Practices

#### Experiment Design
- **Baseline Models**: Basit modeller ile başlama
- **Incremental Improvement**: Adım adım geliştirme
- **Ablation Studies**: Her değişikliğin etkisini ölçme
- **Reproducibility**: Sonuçları tekrarlanabilir kılma

#### Model Validation
- **Multiple Metrics**: Tek metric'e bağımlı kalmama
- **Statistical Significance**: T-test, confidence intervals
- **Business Metrics**: Technical metrics'in business impact'i
- **Robustness**: Farklı veri setlerinde test

#### Documentation
- **Experiment Logs**: Detaylı kayıt tutma
- **Model Cards**: Model özellikleri dokümantasyonu
- **Code Comments**: Kodun açıklanması
- **Decision Rationale**: Neden bu yaklaşım seçildi

## 🔍 Önemli Notlar

### Overfitting vs Underfitting
- **Overfitting**: Training'de iyi, validation'da kötü
- **Underfitting**: Her ikisinde de kötü performans
- **Sweet Spot**: Optimal bias-variance trade-off

### Model Complexity
- **Simple Models**: Hızlı, yorumlanabilir, az veri
- **Complex Models**: Yüksek performans, çok veri gerektirir
- **Ensemble**: Complexity ve performance dengesinde

### Hyperparameter vs Parameter
- **Parameters**: Model tarafından öğrenilen (weights, biases)
- **Hyperparameters**: Kullanıcı tarafından belirlenen (learning rate, depth)
- **Meta-parameters**: Hyperparameter optimization'un parametreleri

## 🚀 Advanced Topics

### AutoML
- **Tanım**: Makine öğrenmesi pipeline'ının otomasyonu
- **Tools**: AutoSklearn, TPOT, H2O AutoML
- **Avantajlar**: Hızlı prototyping, domain expertise gerektirmez
- **Sınırlar**: Esnek olmayan, black-box yaklaşım

### Multi-objective Optimization
- **Tanım**: Birden fazla metric'i aynı anda optimize etme
- **Pareto Front**: Trade-off'lar arasında optimal noktalar
- **NSGA-II**: Multi-objective genetic algorithm
- **Applications**: Accuracy vs speed, performance vs interpretability

### Transfer Learning
- **Tanım**: Önceden eğitilmiş modeli yeni problem için kullanma
- **Fine-tuning**: Son katmanları yeniden eğitme
- **Feature Extraction**: Önceki katmanları feature extractor olarak kullanma
- **Domain Adaptation**: Farklı domain'lere uyarlama 