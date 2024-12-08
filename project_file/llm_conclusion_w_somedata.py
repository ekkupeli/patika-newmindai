import pandas as pd
import json
with open("datasets\\custom\\data_w_class.json", encoding="utf-8") as json_file:      
    review_data = json_file.readlines()
    # Bu satır 315000 satırlık verinin tümünü işlediğinden yaklaşık 1 dk bir süre almaktadır. 
    review_data = list(map(json.loads, review_data)) 

review_w_class_df= pd.DataFrame(review_data)
json_file.close()

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
#import torch                           #with GPU
#print(torch.cuda.is_available())
#import datasets

# Veri setini yükleme (hata sanırım burada, veri setinde "input_ids": ..., "attention_mask": ..., "labels": ... yer almalı)

#dataset = datasets.load_dataset('json', data_files={'train': 'yelp_train.json', 'validation': 'yelp_valid.json'})
subset_t = review_w_class_df.groupby("business_id")["text"].head(1000)
subset_v = review_w_class_df.groupby("business_id")["text"].apply(lambda x: x[1002:1100])

# Model ve tokenizer yükleme
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Eğitim parametreleri
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=2,
)

# Eğitici
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=subset_t,
    eval_dataset=subset_v
)

# Eğitimi başlatma
trainer.train()

# Sonuç üretme
prompt = ("""
Based on the data:
1. Which business has higher customer satisfaction?
2. In which areas should improvement be made?
Answer the questions.
""")

# Tokenize etme ve mask ekleme
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# Çıktı üretme
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.pad_token_id,
)

# Sonucu dekode etme
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)