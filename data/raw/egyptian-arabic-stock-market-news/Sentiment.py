from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
# Ensure NumPy is below version 2.0.0
# !pip install numpy<2.0.0

# Install openpyxl to handle Excel files
# !pip install openpyxl

# Local directory where the model and tokenizer files are saved
local_directory = "C://Users//Diaa//PycharmProjects//Summarizing//model"

# Load the model and tokenizer from the local directory
model = AutoModelForSequenceClassification.from_pretrained(local_directory)
tokenizer = AutoTokenizer.from_pretrained(local_directory)

df = pd.read_csv("summarized_texts_from_Ezz_Steel.csv")

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get the model outputs
    outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class (0 or 1) and its probability
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class_prob = probabilities[0, predicted_class_idx].item()

    # Determine the class label
    if predicted_class_idx == 0:
        predicted_class = "Negative"
    else:
        predicted_class = "Positive"
    print(predicted_class)
    return predicted_class, predicted_class_prob

# Apply the prediction function to each row in the DataFrame
df["Predicted Class"], df["Probability"] = zip(*df["Summary"].apply(predict))
df.to_excel('Ezz Steel data ready to pre-process.xlsx', index=False)
# Print the DataFrame with predictions
print(df)