# MLOps Eğitim Süreci 🚀

Bu repository, MLOps (Machine Learning Operations) süreçlerini adım adım öğrenmek için tasarlanmış kapsamlı bir eğitim materyalidir.

## 📚 Eğitim Seviyeleri

### Level 1: Basics - MLOps Temelleri
- MLOps'un tanımı ve önemi
- Geleneksel ML vs MLOps karşılaştırması
- MLOps yaşam döngüsü
- Temel araçlar ve teknolojiler
- İlk MLflow denemesi

### Level 2: Data Pipeline - Veri Pipeline'ları
- Veri versiyonlama (DVC)
- Veri kalitesi kontrolü
- Feature engineering pipeline'ları
- Veri drift detection

### Level 3: Model Development - Model Geliştirme
- Experiment tracking
- Hyperparameter optimization
- Model validation
- Feature stores

### Level 4: Model Training - Model Eğitimi
- Distributed training
- Pipeline orchestration
- Model versioning
- Training monitoring

### Level 5: Model Deployment - Model Dağıtımı
- Model serving
- API development
- Containerization
- Cloud deployment

### Level 6: Monitoring - İzleme
- Model performance monitoring
- Data drift detection
- Alert systems
- Dashboard creation

### Level 7: Advanced MLOps - İleri Seviye
- Multi-model orchestration
- A/B testing
- Feature store management
- MLOps best practices

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- Git
- Docker (opsiyonel)

### Hızlı Başlangıç

1. **Repository'yi klonlayın:**
   ```bash
   git clone <your-repo-url>
   cd MLOPS-Learn
   ```

2. **Virtual environment oluşturun:**
   ```bash
   python -m venv mlops-env
   
   # Windows
   mlops-env\Scripts\activate
   
   # Linux/Mac
   source mlops-env/bin/activate
   ```

3. **İlk seviye için kütüphaneleri kurun:**
   ```bash
   cd Level_1_Basics
   pip install -r requirements.txt
   ```

4. **Jupyter notebook'u başlatın:**
   ```bash
   jupyter notebook 01_MLOps_Giris.ipynb
   ```

## 📋 Her Seviye için Kurulum

Her seviyenin kendi `requirements.txt` dosyası bulunmaktadır. Varsayılan dosya hafif bir "çekirdek" kurulum sunar ve aynı dizindeki `requirements-optional.txt` dosyası isteğe bağlı araçları listeler.

```bash
# Level 1
cd Level_1_Basics && pip install -r requirements.txt

# Level 2
cd Level_2_Data_Pipeline && pip install -r requirements.txt

# Level 3
cd Level_3_Model_Development && pip install -r requirements.txt

# Level 4
cd Level_4_Model_Training && pip install -r requirements.txt

# Level 5
cd Level_5_Model_Deployment && pip install -r requirements.txt

# Level 6
cd Level_6_Monitoring && pip install -r requirements.txt

# Level 7
cd Level_7_Advanced_MLOps && pip install -r requirements.txt
```

## 🎯 Öğrenme Yolculuğu

1. **Başlangıç:** Level 1'den başlayın
2. **Sıralı İlerleme:** Her seviyeyi sırasıyla tamamlayın
3. **Uygulama:** Her seviyedeki kod örneklerini çalıştırın
4. **Deneyim:** Kendi verilerinizle deneyin

## 📊 Kullanılan Teknolojiler

- **Experiment Tracking:** MLflow, Weights & Biases (opsiyonel)
- **Data Versioning:** DVC
- **Pipeline Orchestration:** Prefect, Apache Airflow (opsiyonel)
- **Model Serving:** FastAPI, BentoML (opsiyonel), Seldon Core (opsiyonel)
- **Monitoring:** Evidently, WhyLabs/whylogs, Prometheus
- **Dağıtık & GPU Eğitim:** Ray, Dask, Horovod (opsiyonel)
- **Feature Store:** Feast

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📖 Ek Kaynaklar

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [MLOps Community](https://mlops.community/)
- [DVC Documentation](https://dvc.org/doc)

## 📝 Lisans

Bu proje eğitim amaçlı hazırlanmıştır.

## ⭐ Destek

Bu projeyi faydalı bulduysanız, lütfen ⭐ vererek destekleyin!

---

**🎉 MLOps yolculuğunuza hoş geldiniz!**
