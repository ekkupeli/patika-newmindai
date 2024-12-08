import pandas as pd
import numpy as np
import json
with open("datasets\\custom\\data.json", encoding="utf-8") as json_file:      
    review_data = json_file.readlines()
    # Bu satır 315000 satırlık verinin tümünü işlediğinden yaklaşık 1 dk bir süre almaktadır. 
    review_data = list(map(json.loads, review_data)) 

review_df= pd.DataFrame(review_data)
json_file.close()

print("""
#      -------------------------------------
#        Data (head(5)):
#      -------------------------------------
#      """)
print(review_df.head())

print("""
#      -------------------------------------
#        Data Info:
#      -------------------------------------
#      """)
print(review_df.info())

#Unique controls
print("""
#      -------------------------------------
#        Data Review Unique Info:
#      -------------------------------------
#      """)
print(len(review_df["review_id"].unique()))         #314941

print("""
#      -------------------------------------
#        Data Business Unique Info:
#      -------------------------------------
#      """)
print(len(review_df["business_id"].unique()))       #12819


#Koşulla sınıflama

# Koşulları ve atamaları tanımlama
conditions = [
    (review_df["roberta_pos"] > review_df["roberta_neu"]) & (review_df["roberta_pos"] > review_df["roberta_neg"]),
    (review_df["roberta_neg"] > review_df["roberta_pos"]) & (review_df["roberta_neg"] > review_df["roberta_neu"])
]
choices = ["Positive", "Negative"]

# Yeni sütunu oluşturma
review_df["class"] = np.select(conditions, choices, default="Neutral")
#print(review_df.head(10))


# JSON formatında kaydetme
#review_df.to_json("datasets\\custom\\data_w_class.json", orient="records", lines=True, force_ascii=False)
#print("JSON dosyasına kaydedildi.")

