import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
local_dir = "C://Users/Diaa/PycharmProjects/Summarizing/Summarizing model"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)

# Function to summarize text
def summarize(title, article):
    full_text = str(title) + " " + str(article)
    inputs = tokenizer.encode("summarize: " + full_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Print the summary for debugging
    print(f"Title: {title}\nSummary: {summary}\n")
    return summary


# Load your DataFrame
Articles = pd.read_csv(r'C:\Users\Diaa\PycharmProjects\Summarizing\telecom Eg_2.csv')  # Change to your file path

# Apply the summarization function to the 'text_column'
Articles['Summary'] = Articles.apply(lambda row: summarize(row['Title'], row['Article']), axis=1)

Articles.to_csv('summarized_texts_from_telecom.csv', index=False)

print(Articles.head())
