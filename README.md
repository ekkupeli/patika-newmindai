# Yelp Firma Yorumları Analizi
Bu proje, **Yelp academic veriseti** kullanılarak, firma yorumlarının **pozitif**, **nötr** veya **negatif** olarak sınıflandırılmasını ve bu analizlerin anlamlı sonuçlara dönüştürülmesini amaçlamaktadır. Çalışma, NLP (Doğal Dil İşleme) modelleri ve LLM (Large Language Model) entegrasyonlarını içermektedir.

## Proje Hedefleri
- Kullanıcı yorumlarını analiz ederek pozitif, nötr ve negatif kategorilere sınıflandırmak.
- Firmaların müşteri memnuniyet oranlarını belirlemek.
- LLM (Büyük Dil Modelleri) kullanarak, analiz sonuçlarından anlamlı raporlar ve öneriler oluşturmak.
- Firma performansı ve müşteri memnuniyeti hakkında genel bir değerlendirme sağlamak.

---

## Kullanılan Teknolojiler ve Araçlar
### Programlama Dili
- Python 3.10+

### Kütüphaneler
- **NLP ve LLM Modelleri**:
  - Transformers (Hugging Face) – GPT-2, Flan-T5, BART
  - nltk – Metin ön işleme
- **Veri İşleme**:
  - pandas
  - numpy
- **Görselleştirme**:
  - matplotlib
  - seaborn
- **API Geliştirme**:
  - FastAPI (isteğe bağlı)

---

## Kurulum
### Teknoloji ve Ortam Kurulumu
Proje Python programlama dili kullanılarak kodlanmıştır. Bu proje için gerekli kütüphaneler 'env' klasöründe listelenmiştir:
- **Anaconda** kullanıyorsanız gerekli ortamı oluşturmak için `environment.yml` dosyası kullanılabilir.
- Anaconda kullanıcısı olmayanlar için, ilgili kütüphane sürümleri için `environment_fh.yml` dosyası kullanılabilir.

#### LLM Modeli için (Örnek):

- **Tokenizer yükleme**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```
- **Flan-T5 modelini yükleme**
```python
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### Veri Kümeleri
Bu projede kullanılan veri setleri `datasets/custom` klasöründe bulunur. Bunlar şunları içerir:
- **Data**
- **Data with Classification**

## Görev Belgesi
Görev ayrıntıları ve gereksinimleri, proje dizininde bulunan bir `.docx` dosyasında belgelenmiştir.

## Görev 1: Veri Temizleme ve İşleme

