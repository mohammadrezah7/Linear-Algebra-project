import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("df_file.csv")
words = pd.read_csv("words.csv")

print("=" * 60)
print("Our Dataset Info")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")
print(f"Vocabulary Shape: {words.shape}")

print("\nFirst 5 rows of dataset:")
print(df.head())


def preprocess(text):
    text = str(text)
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Processed_Text"] = df["Text"].apply(preprocess)

print("\n")
print("=" * 60)
print("Results")
print("=" * 60)

for i in range(5):
    print(f"\nSample {i+1}")

    print("Original Text:")
    print(df.loc[i, "Text"])

    print("\nProcessed Text:")
    print(df.loc[i, "Processed_Text"])

    print("-" * 60)

df.to_csv("processed_dataset.csv", index=False)

print("\nProcessed dataset saved successfully.")
print("File name: processed_dataset.csv")