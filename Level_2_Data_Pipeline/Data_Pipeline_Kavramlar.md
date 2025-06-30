# Data Pipeline Kavramlar Sözlüğü

Bu dosya Level 2: Data Pipeline notebook'unda geçen tüm kavramları detaylı olarak açıklamaktadır.

## 📚 İçindekiler
- [Data Pipeline Temelleri](#data-pipeline-temelleri)
- [Veri Versiyonlama](#veri-versiyonlama)
- [Veri Kalitesi ve Validasyon](#veri-kalitesi-ve-validasyon)
- [Veri Temizleme](#veri-temizleme)
- [Feature Engineering](#feature-engineering)
- [Veri Drift ve Monitoring](#veri-drift-ve-monitoring)

---

## Data Pipeline Temelleri

### 🔄 Data Pipeline
**Tanım:** Ham verinin toplanmasından model eğitimi için hazır hale getirilmesine kadar olan tüm veri işleme süreçlerinin otomatikleştirilmiş dizisi.

**Pipeline Bileşenleri:**
1. **Data Ingestion (Veri Alma):** Çeşitli kaynaklardan veri toplama
2. **Data Validation (Veri Doğrulama):** Veri kalitesi kontrolü
3. **Data Transformation (Veri Dönüştürme):** Veriyi uygun formata çevirme
4. **Feature Engineering (Özellik Mühendisliği):** Yeni özellikler oluşturma
5. **Data Storage (Veri Depolama):** İşlenmiş veriyi saklama

### 🎯 ETL vs ELT
**ETL (Extract, Transform, Load):**
- Önce transform, sonra load
- Geleneksel data warehouse yaklaşımı
- Structured data için uygun

**ELT (Extract, Load, Transform):**
- Önce load, sonra transform
- Modern big data yaklaşımı
- Unstructured data için uygun

### 📊 Data Lineage
**Tanım:** Verinin nereden geldiği, nasıl işlendiği ve nereye gittiğinin izlenmesi.

**Faydaları:**
- Troubleshooting
- Compliance
- Impact analysis
- Data governance

---

## Veri Versiyonlama

### 🗃️ DVC (Data Version Control)
**Tanım:** Büyük veri dosyaları ve ML modelleri için Git-benzeri version control sistemi.

**Temel Özellikler:**
- Large file versioning
- Remote storage integration
- Pipeline reproducibility
- Experiment tracking

**Temel Komutlar:**
```bash
# DVC başlatma
dvc init

# Veri dosyası ekleme
dvc add data/dataset.csv

# Pipeline tanımlama
dvc run -d data/raw -o data/processed python process.py

# Remote storage kurulumu
dvc remote add -d storage s3://my-bucket/dvc-storage

# Veriyi remote'a gönderme
dvc push

# Veriyi remote'dan çekme
dvc pull

# Pipeline yeniden çalıştırma
dvc repro
```

### 📝 dvc.yaml
**Tanım:** DVC pipeline'larını tanımlamak için kullanılan YAML dosyası.

**Örnek Yapı:**
```yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - data/raw
    outs:
      - data/prepared
      
  train:
    cmd: python train.py
    deps:
      - data/prepared
      - train.py
    outs:
      - models/model.pkl
    metrics:
      - metrics.json
```

### 🔄 Git vs DVC
**Git:** Kod için version control
**DVC:** Veri ve modeller için version control
**Birlikte kullanım:** Tam reproducibility için her ikisi gerekli

---

## Veri Kalitesi ve Validasyon

### 🔍 Data Validation
**Tanım:** Verinin beklenen kalite standartlarını karşılayıp karşılamadığının kontrol edilmesi.

**Validation Türleri:**
1. **Schema Validation:** Veri tiplerinin ve yapının kontrolü
2. **Range Validation:** Değerlerin beklenen aralıklarda olup olmadığı
3. **Pattern Validation:** Regex pattern'lerine uygunluk
4. **Business Rule Validation:** İş kurallarına uygunluk

### 📊 Data Quality Metrics
**Temel Kalite Boyutları:**
- **Completeness (Tamlık):** Eksik değer oranı
- **Accuracy (Doğruluk):** Verinin gerçeği yansıtma derecesi
- **Consistency (Tutarlılık):** Farklı kaynaklardaki verinin uyumu
- **Validity (Geçerlilik):** Tanımlı kurallara uygunluk
- **Uniqueness (Benzersizlik):** Duplikasyon oranı
- **Timeliness (Güncellik):** Verinin ne kadar güncel olduğu

### 🚨 Data Quality Issues
**Yaygın Problemler:**
```python
# Eksik değerler
df.isnull().sum()

# Duplikatlar
df.duplicated().sum()

# Aykırı değerler (IQR yöntemi)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)]

# Veri tipi uyumsuzlukları
df.dtypes
```

### 🛡️ Great Expectations
**Tanım:** Data validation ve documentation için Python kütüphanesi.

**Temel Kullanım:**
```python
import great_expectations as ge

# Dataset oluşturma
df_ge = ge.from_pandas(df)

# Expectations tanımlama
df_ge.expect_column_values_to_not_be_null('column_name')
df_ge.expect_column_values_to_be_between('numeric_column', 0, 100)
df_ge.expect_column_values_to_match_regex('email', r'^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$')
```

---

## Veri Temizleme

### 🧹 Data Cleaning
**Tanım:** Verinin kalitesini artırmak için hatalı, eksik veya tutarsız verilerin düzeltilmesi süreci.

**Temel Adımlar:**
1. **Missing Value Handling:** Eksik değerlerin işlenmesi
2. **Outlier Detection & Treatment:** Aykırı değerlerin tespit edilmesi ve işlenmesi
3. **Duplicate Removal:** Duplikatların kaldırılması
4. **Data Type Conversion:** Veri tiplerinin düzeltilmesi
5. **Standardization:** Verilerin standardizasyonu

### 🔧 Missing Value Strategies
**Silme Stratejileri:**
- **Listwise Deletion:** Eksik değeri olan tüm satırları silme
- **Pairwise Deletion:** Sadece ilgili analiz için eksik değerleri silme

**Doldurma Stratejileri:**
```python
# Mean/Median/Mode ile doldurma
df['column'].fillna(df['column'].mean())
df['column'].fillna(df['column'].median())
df['column'].fillna(df['column'].mode()[0])

# Forward/Backward fill
df['column'].fillna(method='ffill')
df['column'].fillna(method='bfill')

# Interpolation
df['column'].interpolate()

# Machine learning based imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

### 📈 Outlier Detection Methods
**İstatistiksel Yöntemler:**
```python
# Z-Score method
z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['column'] < Q1 - 1.5*IQR) | 
              (df['column'] > Q3 + 1.5*IQR)]
```

**Machine Learning Yöntemleri:**
```python
# Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(df)

# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(df)
```

### 🎯 Outlier Treatment
**Treatment Stratejileri:**
- **Removal:** Aykırı değerleri silme
- **Capping/Winsorizing:** Belirli değerlerde sınırlama
- **Transformation:** Log, sqrt gibi dönüşümler
- **Binning:** Kategorik gruplara ayırma

---

## Feature Engineering

### 🔧 Feature Engineering
**Tanım:** Ham veriden makine öğrenmesi algoritmalarının daha iyi performans gösterebileceği özellikler oluşturma süreci.

**Ana Kategoriler:**
1. **Feature Creation:** Yeni özellikler oluşturma
2. **Feature Transformation:** Mevcut özellikleri dönüştürme
3. **Feature Selection:** En önemli özellikleri seçme
4. **Feature Scaling:** Özellikleri ölçeklendirme

### 🎨 Feature Creation Techniques
**Numerical Features:**
```python
# Mathematical operations
df['feature_ratio'] = df['feature1'] / df['feature2']
df['feature_diff'] = df['feature1'] - df['feature2']
df['feature_product'] = df['feature1'] * df['feature2']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior'])
```

**Categorical Features:**
```python
# One-hot encoding
pd.get_dummies(df['category'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Target encoding
df.groupby('category')['target'].mean()
```

**DateTime Features:**
```python
# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

# Time-based features
df['days_since'] = (df['current_date'] - df['reference_date']).dt.days
```

### 📏 Feature Scaling
**Normalization (Min-Max Scaling):**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
```

**Standardization (Z-Score):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)
```

**Robust Scaling:**
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_robust = scaler.fit_transform(df)
```

### 🎯 Feature Selection
**Filter Methods:**
```python
# Correlation-based
correlation_matrix = df.corr()
high_corr_features = correlation_matrix[correlation_matrix > 0.9]

# Statistical tests
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
selected_features = selector.fit_transform(X, y)
```

**Wrapper Methods:**
```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=10)
selected_features = selector.fit_transform(X, y)
```

**Embedded Methods:**
```python
# LASSO regularization
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]
```

---

## Veri Drift ve Monitoring

### 📊 Data Drift
**Tanım:** Zaman içinde model input verilerinin istatistiksel özelliklerinin değişmesi.

**Drift Türleri:**
1. **Covariate Shift:** P(X) değişir, P(Y|X) sabit kalır
2. **Prior Probability Shift:** P(Y) değişir, P(X|Y) sabit kalır
3. **Concept Drift:** P(Y|X) değişir

### 🔍 Drift Detection Methods
**İstatistiksel Testler:**
```python
# Kolmogorov-Smirnov Test
from scipy import stats
ks_statistic, p_value = stats.ks_2samp(baseline_data, current_data)

# Chi-square Test (categorical data)
chi2, p_value = stats.chisquare(observed, expected)

# Population Stability Index (PSI)
def calculate_psi(expected, actual, buckets=10):
    expected_percents = pd.qcut(expected, buckets, duplicates='drop')
    actual_percents = pd.qcut(actual, buckets, duplicates='drop')
    
    expected_counts = expected_percents.value_counts()
    actual_counts = actual_percents.value_counts()
    
    psi_value = sum((actual_counts/len(actual) - expected_counts/len(expected)) * 
                   np.log((actual_counts/len(actual)) / (expected_counts/len(expected))))
    return psi_value
```

**Distance-based Methods:**
```python
# Wasserstein Distance
from scipy.stats import wasserstein_distance
distance = wasserstein_distance(baseline_data, current_data)

# KL Divergence
from scipy.stats import entropy
kl_div = entropy(baseline_data, current_data)
```

### 🚨 Evidently AI
**Tanım:** ML model ve data monitoring için açık kaynak kütüphane.

**Temel Kullanım:**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Data drift report
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=baseline_df, current_data=current_df)
data_drift_report.show()

# Data quality report
data_quality_report = Report(metrics=[DataQualityPreset()])
data_quality_report.run(reference_data=baseline_df, current_data=current_df)
```

### 📈 WhyLabs
**Tanım:** ML observability ve monitoring platformu.

**Özellikler:**
- Automated data profiling
- Drift detection
- Data quality monitoring
- Anomaly detection

---

## 🛠️ Pipeline Orchestration

### 📊 Apache Airflow
**Tanım:** Workflow scheduling ve monitoring platformu.

**DAG (Directed Acyclic Graph) Örneği:**
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Data processing pipeline',
    schedule_interval='@daily'
)

def extract_data():
    # Data extraction logic
    pass

def transform_data():
    # Data transformation logic
    pass

def load_data():
    # Data loading logic
    pass

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag
)

extract_task >> transform_task >> load_task
```

### 🔄 Prefect
**Tanım:** Modern workflow orchestration tool.

**Temel Kullanım:**
```python
import prefect
from prefect import Flow, task

@task
def extract_data():
    return "extracted_data"

@task
def transform_data(data):
    return f"transformed_{data}"

@task
def load_data(data):
    print(f"Loading {data}")

with Flow("data-pipeline") as flow:
    raw_data = extract_data()
    processed_data = transform_data(raw_data)
    load_data(processed_data)

flow.run()
```

---

## 🗄️ Data Storage Solutions

### 📊 Data Lake vs Data Warehouse
**Data Lake:**
- Schema-on-read
- Raw data storage
- Flexible structure
- Lower cost
- Big data friendly

**Data Warehouse:**
- Schema-on-write
- Processed data storage
- Structured data
- Higher cost
- Analytics optimized

### 🏗️ Modern Data Stack
**Typical Components:**
1. **Data Sources:** Applications, APIs, databases
2. **Data Ingestion:** Fivetran, Stitch, Airbyte
3. **Data Storage:** Snowflake, BigQuery, Redshift
4. **Data Transformation:** dbt, Dataform
5. **Business Intelligence:** Tableau, Looker, PowerBI
6. **Data Orchestration:** Airflow, Prefect
7. **Data Quality:** Great Expectations, Monte Carlo

---

## 🔧 Praktik Uygulamalar

### 📝 Pipeline Implementation
```python
class DataPipeline:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step_func, step_name):
        self.steps.append((step_name, step_func))
    
    def run(self, data):
        for step_name, step_func in self.steps:
            print(f"Running {step_name}...")
            data = step_func(data)
            print(f"{step_name} completed.")
        return data

# Pipeline oluşturma
pipeline = DataPipeline()
pipeline.add_step(validate_data, "Data Validation")
pipeline.add_step(clean_data, "Data Cleaning")
pipeline.add_step(engineer_features, "Feature Engineering")

# Pipeline çalıştırma
processed_data = pipeline.run(raw_data)
```

### 🔍 Data Quality Monitoring
```python
def monitor_data_quality(df):
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Quality alerts
    if quality_report['missing_values'] > len(df) * 0.1:
        print("⚠️  High missing value rate detected!")
    
    if quality_report['duplicate_rows'] > 0:
        print("⚠️  Duplicate rows detected!")
    
    return quality_report
```

---

## 🎓 Best Practices

### ✅ Pipeline Design
1. **Modularity:** Her step'i ayrı function olarak tasarla
2. **Idempotency:** Pipeline'ı birden çok kez çalıştırabilir ol
3. **Error Handling:** Robust error handling ve logging
4. **Testing:** Unit test'ler ve integration test'ler
5. **Documentation:** Her step'i dokümante et

### 🔄 Data Management
1. **Version Control:** Veri ve kod versiyonlarını takip et
2. **Backup Strategy:** Düzenli backup'lar al
3. **Access Control:** Data governance politikaları uygula
4. **Monitoring:** Sürekli monitoring ve alerting
5. **Scalability:** Büyüyen veri hacmi için ölçeklenebilir tasarım

### 📊 Quality Assurance
1. **Data Profiling:** Düzenli veri profilleme
2. **Validation Rules:** Comprehensive validation rules
3. **Drift Detection:** Sürekli drift monitoring
4. **Anomaly Detection:** Otomatik anomaly detection
5. **Quality Metrics:** KPI'lar ve dashboard'lar

---

## 🔗 İlgili Kaynaklar

- [DVC Documentation](https://dvc.org/doc)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

*Bu döküman Level 2: Data Pipeline notebook'unda geçen kavramları kapsamaktadır. Her kavram praktik örneklerle desteklenmiştir.* 