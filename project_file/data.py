import pandas as pd
from sqlalchemy import create_engine
import json

# Veritabanı bağlantısı (örneğin, MySQL)
engine = create_engine("sqlite:///datasets/base/business_recommendations.db", echo=True)

# SQL sorgusunu çalıştır ve DataFrame'e dönüştür
df = pd.read_sql("SELECT * FROM review_scores", con=engine)

# Kullanılmayacak sütunları kaldır
columns_to_drop = ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "user_id", "funny", "cool"]
df = df.drop(columns=columns_to_drop)

#id sütunlarını elle numarala
df["review_id"] = range(len(df))

# business id sütununu altı basamaklı formatla
df["business_id"] = pd.factorize(df["business_id"])[0]
df["business_id"] = df["business_id"].astype(str).str.zfill(6)

# DataFrame'i JSON formatına dönüştür
# JSON dosyasına kaydet
with open("datasets\\custom\\data.json", "w", encoding="utf-8") as json_file:
    for row in df.to_dict(orient="records"):
        json_file.write(f"{json.dumps(row)}\n")

print("Pandas ile JSON dosyası başarıyla oluşturuldu.")
