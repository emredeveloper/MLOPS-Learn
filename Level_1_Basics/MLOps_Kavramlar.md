# MLOps Temel Kavramlar Sözlüğü

Bu dosya Level 1: MLOps Giriş notebook'unda geçen tüm kavramları detaylı olarak açıklamaktadır.

## 📚 İçindekiler
- [MLOps ve Temel Kavramlar](#mlops-ve-temel-kavramlar)
- [Araçlar ve Teknolojiler](#araçlar-ve-teknolojiler)
- [Model Lifecycle](#model-lifecycle)
- [Metrikler ve Değerlendirme](#metrikler-ve-değerlendirme)

---

## MLOps ve Temel Kavramlar

### 🤖 MLOps (Machine Learning Operations)
**Tanım:** Makine öğrenmesi modellerinin geliştirme, dağıtım ve bakımını otomatikleştiren DevOps prensiplerine dayalı bir metodoloji.

**Ana Bileşenleri:**
- **Development (Geliştirme):** Model tasarımı, feature engineering, algoritma seçimi
- **Operations (Operasyonlar):** Model dağıtımı, monitoring, maintenance
- **Infrastructure (Altyapı):** Ölçeklenebilir hesaplama kaynakları, data storage

### 🔄 CI/CD Pipeline
**Tanım:** Continuous Integration/Continuous Deployment - Kod değişikliklerinin otomatik olarak test edilip production'a deploy edilmesi süreci.

**MLOps'taki Rolü:**
- Kod değişikliklerinin otomatik testi
- Model eğitiminin otomatikleştirilmesi
- Model deployment sürecinin otomatikleştirilmesi

### 📊 Experiment Tracking
**Tanım:** ML deneylerin sistematik olarak kaydedilmesi ve takip edilmesi süreci.

**Kaydedilecek Bilgiler:**
- Model parametreleri (hyperparameters)
- Eğitim metrikleri (accuracy, loss, etc.)
- Model artifacts (eğitilmiş model dosyaları)
- Veri versiyonları
- Kod versiyonları

### 🔧 Versiyonlama
**Model Versiyonlama:** Her model iterasyonunun unique identifier ile kaydedilmesi
**Veri Versiyonlama:** Veri setlerinin değişikliklerinin takip edilmesi
**Kod Versiyonlama:** Git benzeri araçlarla kod değişikliklerinin takibi

---

## Araçlar ve Teknolojiler

### 📈 MLflow
**Tanım:** Açık kaynak ML lifecycle yönetim platformu.

**Ana Fonksiyonları:**
- **MLflow Tracking:** Experiment logging ve comparison
- **MLflow Projects:** Reproducible ML code packaging
- **MLflow Models:** Model packaging ve deployment
- **MLflow Registry:** Centralized model store

**Temel Kullanım:**
```python
import mlflow
import mlflow.sklearn

# Experiment başlat
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # Parametreleri kaydet
    mlflow.log_param("n_estimators", 100)
    
    # Metrikleri kaydet
    mlflow.log_metric("accuracy", 0.95)
    
    # Model kaydet
    mlflow.sklearn.log_model(model, "model")
```

### 🗃️ DVC (Data Version Control)
**Tanım:** Veri setleri ve ML modelleri için Git-benzeri version control sistemi.

**Temel Komutları:**
- `dvc init`: Repository'yi DVC için initialize et
- `dvc add data.csv`: Veri dosyasını DVC tracking'e ekle
- `dvc push`: Verileri remote storage'a yükle
- `dvc pull`: Verileri remote'dan indir

### 🐳 Docker
**Tanım:** Uygulamaları containerize etmek için kullanılan platform.

**MLOps'taki Faydaları:**
- Environment consistency
- Reproducibility
- Easy deployment
- Scalability

### ☸️ Kubernetes
**Tanım:** Container orchestration platformu.

**MLOps Use Cases:**
- Model serving at scale
- Resource management
- Load balancing
- Auto-scaling

### 📊 Apache Airflow
**Tanım:** Workflow orchestration ve scheduling platformu.

**MLOps'taki Kullanımı:**
- Data pipeline orchestration
- Model training workflows
- Automated retraining
- ETL processes

---

## Model Lifecycle

### 🎯 Model Development
**Aşamaları:**
1. **Problem Definition:** Business problem'ın ML problem'ına çevrilmesi
2. **Data Collection:** Gerekli verilerin toplanması
3. **Exploratory Data Analysis (EDA):** Verinin anlaşılması
4. **Feature Engineering:** Özellik çıkarımı ve dönüştürme
5. **Model Selection:** Uygun algoritmanın seçilmesi
6. **Model Training:** Modelin eğitilmesi
7. **Model Evaluation:** Performans değerlendirmesi

### 🚀 Model Deployment
**Deployment Stratejileri:**
- **Blue-Green Deployment:** İki paralel environment kullanma
- **Canary Deployment:** Kademeli olarak traffic'i yeni modele yönlendirme
- **A/B Testing:** İki model versiyonunu karşılaştırma

### 📊 Model Monitoring
**İzlenecek Metrikler:**
- **Performance Metrics:** Accuracy, precision, recall
- **Business Metrics:** Revenue impact, user engagement
- **Technical Metrics:** Latency, throughput, error rates
- **Data Quality:** Input data drift, schema changes

### 🔄 Model Retraining
**Retraining Triggers:**
- Performance degradation
- Data drift detection
- Scheduled retraining
- Business rule changes

---

## Metrikler ve Değerlendirme

### 📊 Classification Metrics

#### Accuracy (Doğruluk)
**Formül:** `(TP + TN) / (TP + TN + FP + FN)`
**Açıklama:** Doğru tahminlerin toplam tahminlere oranı

#### Precision (Kesinlik)
**Formül:** `TP / (TP + FP)`
**Açıklama:** Pozitif tahmin edilenlerin ne kadarının gerçekten pozitif olduğu

#### Recall (Duyarlılık)
**Formül:** `TP / (TP + FN)`
**Açıklama:** Gerçek pozitiflerin ne kadarının doğru tahmin edildiği

#### F1-Score
**Formül:** `2 * (Precision * Recall) / (Precision + Recall)`
**Açıklama:** Precision ve Recall'ın harmonik ortalaması

### 📈 Regression Metrics

#### MSE (Mean Squared Error)
**Formül:** `Σ(y_true - y_pred)² / n`
**Açıklama:** Hataların karelerinin ortalaması

#### RMSE (Root Mean Squared Error)
**Formül:** `√(MSE)`
**Açıklama:** MSE'nin karekökü, orijinal birim cinsinden hata

#### MAE (Mean Absolute Error)
**Formül:** `Σ|y_true - y_pred| / n`
**Açıklama:** Mutlak hataların ortalaması

#### R² (R-squared)
**Formül:** `1 - (SS_res / SS_tot)`
**Açıklama:** Modelin açıkladığı varyansın oranı

---

## 🔧 Teknik Kavramlar

### 🎛️ Hyperparameters
**Tanım:** Model eğitimi öncesinde belirlenen ve eğitim süreci boyunca sabit kalan parametreler.

**Örnekler:**
- Learning rate
- Number of epochs
- Batch size
- Number of hidden layers
- Regularization parameters

### 🎯 Cross-Validation
**Tanım:** Model performansını güvenilir şekilde değerlendirmek için veri setini farklı şekillerde bölerek model eğitimi ve testi yapma tekniği.

**Türleri:**
- **K-Fold CV:** Veriyi K parçaya bölme
- **Stratified CV:** Sınıf dağılımını koruyarak bölme
- **Time Series CV:** Zaman serisi verileri için özel bölme

### 📊 Train-Validation-Test Split
**Train Set:** Modelin öğrendiği veri seti (%70-80)
**Validation Set:** Hyperparameter tuning için kullanılan set (%10-15)
**Test Set:** Final model performansını değerlendirmek için kullanılan set (%10-15)

### 🎭 Overfitting vs Underfitting
**Overfitting:** Modelin training data'ya aşırı uyum sağlaması, generalization yeteneğini kaybetmesi
**Underfitting:** Modelin yeterince complex olmaması, pattern'leri yakalayamaması

---

## 🛠️ Praktik Uygulamalar

### 📝 Experiment Logging
```python
# Temel MLflow logging
mlflow.log_param("parameter_name", value)
mlflow.log_metric("metric_name", value)
mlflow.log_artifact("file_path")
```

### 🔍 Model Comparison
```python
# Birden fazla model karşılaştırması
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
```

### 📊 Performance Visualization
```python
# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

---

## 🎓 Best Practices

### ✅ Experiment Organization
1. **Clear Naming:** Experiment ve run'lar için açıklayıcı isimler kullan
2. **Consistent Logging:** Her experiment'te aynı metrikleri logla
3. **Documentation:** Her experiment için açıklama ekle
4. **Reproducibility:** Random seed'leri kaydet

### 🔄 Version Control
1. **Code Versioning:** Her experiment için kod durumunu kaydet
2. **Data Versioning:** Kullanılan veri setinin versiyonunu belirt
3. **Model Versioning:** Model artifacts'ını version'la
4. **Environment Versioning:** Dependency'leri kaydet

### 📊 Monitoring
1. **Regular Checks:** Model performansını düzenli kontrol et
2. **Alerting:** Performans düşüşleri için alarm kur
3. **Logging:** Tüm önemli event'leri logla
4. **Dashboards:** Görsel monitoring dashboard'ları oluştur

---

## 🔗 İlgili Kaynaklar

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLOps Community](https://mlops.community/)
- [Google Cloud MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Azure MLOps](https://azure.microsoft.com/en-us/services/machine-learning/mlops/)
- [AWS MLOps](https://aws.amazon.com/machine-learning/mlops/)

---