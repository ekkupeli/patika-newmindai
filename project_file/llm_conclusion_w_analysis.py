import pandas as pd
import json
with open("datasets\\custom\\data_w_class.json", encoding="utf-8") as json_file:      
    review_data = json_file.readlines()
    # Bu satır 315000 satırlık verinin tümünü işlediğinden yaklaşık 1 dk bir süre almaktadır. 
    review_data = list(map(json.loads, review_data)) 

review_w_class_df= pd.DataFrame(review_data)
json_file.close()

##Analysis

# En çok yorum alan firmaların ID"lerini ve yorum sayılarını belirleme
top_businesses = review_w_class_df["business_id"].value_counts().head(5)
top_businesses_ids = top_businesses.index.tolist()

# Yorumların sınıf bazında dağılımını hesaplama
sentiment_distribution = review_w_class_df.groupby("business_id")["class"].value_counts(normalize=True)

# En çok yorum alan firmalar için analiz çıktısı
analysis_results = []
for business_id in top_businesses_ids:
    sentiments = sentiment_distribution[business_id]
    analysis_results.append({
        "business_id": business_id,
        "positive": sentiments.get("Positive", 0),
        "neutral": sentiments.get("Neutral", 0),
        "negative": sentiments.get("Negative", 0),
    })

#print(analysis_results)

"""(print)
[{'business_id': '000104', 'positive': 0.8672752808988764, 'neutral': 0.024929775280898875, 'negative': 0.10779494382022473},
{'business_id': '000421', 'positive': 0.918552036199095, 'neutral': 0.01809954751131222, 'negative': 0.06334841628959276},
{'business_id': '000042', 'positive': 0.8729411764705882, 'neutral': 0.022745098039215685, 'negative': 0.10431372549019607},
{'business_id': '000036', 'positive': 0.7694728560188828, 'neutral': 0.03855232100708104, 'negative': 0.1919748229740362},
{'business_id': '000423', 'positive': 0.8126036484245439, 'neutral': 0.03897180762852405, 'negative': 0.148424543946932}]
"""
##LLM 

#prompt = (
#    "Bu, firmaların yorum analizi sonuçlarıdır:\n\n"
#    "Firma 000104: %87 olumlu, %2 nötr, %11 olumsuz yorum.\n"
#    "Firma 000036: %77 olumlu, %4 nötr, %19 olumsuz yorum.\n\n"
#    "Bu bilgilere dayanarak anlamlı bir sonuç üret. "
#    "Hangi firmanın genel müşteri memnuniyeti daha yüksek? Eksiklikleri özetle."
#)

#prompt = ("""These are the business review analysis results:\n\n"
#"Business 000104: 87% positive, 2% neutral, 11% negative reviews.\n"
#"Business 000036: 77% positive, 4% neutral, 19% negative reviews.\n\n"
#"Produce a meaningful result based on this information."
#"Which business has higher overall customer satisfaction? Summarize the shortcomings.""")

#prompt= ("""
#Yelp analiz sonuçları:
#
#Firma 000104:
#- %87 olumlu yorum
#- %2 nötr yorum
#- %11 olumsuz yorum
#
#Firma 000036:
#- %77 olumlu yorum
#- %4 nötr yorum
#- %19 olumsuz yorum
#
#Yukarıdaki verilere dayanarak:
#1. Hangi firmanın müşteri memnuniyeti daha yüksek?
#2. Hangi alanlarda iyileştirme yapılmalı?
#Sorularını cevapla.
#""")

#prompt = ("""
#Yelp analysis results:
#
#Business 000104:
#- 87% positive comments
#- 2% neutral comments
#- 11% negative comments
#
#Business 000036:
#- 77% positive comments
#- 4% neutral comments
#- 19% negative comments
#
#Based on the data above:
#1. Which business has higher customer satisfaction?
#2. In which areas should improvement be made?
#Answer the questions.
#""")

#prompt =("""
#         "Output:\n"
#    "1. Which business has higher customer satisfaction? Explain why.\n"
#    "2. What areas need improvement? Provide specific suggestions for both businesses.\n"
#         """)

### GPT2, Flan-T5 small, T5 small, BART Large, Flan-T5 Large denendi.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model ve tokenizer yükleme
model_name = "google/flan-t5-large"  # Alternatif: flan-t5-small, flan-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Instruction formatında prompt
prompt = (
    "Task: Analyze the customer satisfaction based on the provided Yelp review data.\n\n"
    "Input:\n"
    "Business 000104:\n"
    "- 87% positive comments\n"
    "- 2% neutral comments\n"
    "- 11% negative comments\n\n"
    "Business 000036:\n"
    "- 77% positive comments\n"
    "- 4% neutral comments\n"
    "- 19% negative comments\n\n"
    "Output:\n"
    "1. Which business has higher customer satisfaction?\n"
    "2. What areas need improvement? (how to improve)\n"
)

# Tokenize etme
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, padding=True, truncation=True)

# Çıktı üretme
outputs = model.generate(
    inputs["input_ids"],
    max_length=200,
    num_beams=4,        # Beam search ile daha çeşitli sonuçlar
    #temperature=0.7,    # Yaratıcılığı artırmak için ayarlandı
    #top_p=0.9,          # Daha odaklı yanıtlar için
    #do_sample=True,
    early_stopping=True
)

# Sonucu dekode etme
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)


"""(print)
Hangi firmanın müşteri memnuniyeti daha yüksek. Hangi alanlarda iyileştirme yapılmalı? HorrifiedSorularını cevapla.
"""

"""(print)
Business 000104: 86% positive comments, 2% neutral comments, 11% negative comments. 
Business 000036: 77% positive comment, 4% neutral comment, 19% negative comment. 
Based on the data above, which business has higher customer satisfaction? And which areas should improvement be made?
"""

"""(print)
1. Business 000104 
2. 77% positive comments
"""