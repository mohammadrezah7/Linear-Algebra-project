import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("Question 17")
print("=" * 60)

DOCUMENT_INDEX = 222

df = pd.read_csv("processed_dataset.csv")
words = pd.read_csv("words.csv")

U = np.load("U_trunc.npy")
S = np.load("S_trunc.npy")
Vt = np.load("Vt_trunc.npy")

vocabulary = words.iloc[:, 0].astype(str).tolist()

document_vectors = U @ np.diag(S)

word_vectors = Vt.T

print(f"Document Vector Shape : {document_vectors.shape}")
print(f"Word Vector Shape     : {word_vectors.shape}")


document_vector = document_vectors[DOCUMENT_INDEX].reshape(1, -1)

document_text = df.loc[DOCUMENT_INDEX, "Processed_Text"]

tokens = document_text.split()

print(f"\nSelected Document : {DOCUMENT_INDEX}")
print(f"Number of Words   : {len(tokens)}")

from collections import Counter

counter = Counter(tokens)

similarities = []
frequencies = []

for word in vocabulary:

    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    idx = word_to_index[word]

    word_vector = word_vectors[idx].reshape(1, -1)

    score = cosine_similarity(document_vector, word_vector)[0, 0]

    similarities.append(score)

    frequencies.append(counter.get(word, 0))


result = pd.DataFrame({

    "Word": vocabulary,

    "Similarity": similarities,

    "Frequency": frequencies

})

top_similarity = result.sort_values(

    by="Similarity",

    ascending=False

).head(20)

top_frequency = result.sort_values(

    by="Frequency",

    ascending=False

).head(20)

print("\nTop Similar Words\n")

print(top_similarity)

print("\nTop Frequent Words\n")

print(top_frequency)

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(10,8))

plt.barh(

    top_similarity["Word"],

    top_similarity["Similarity"]

)

plt.title("Top 20 Word Similarities")

plt.xlabel("Cosine Similarity")

plt.tight_layout()

plt.savefig(

    "images/document_similarity.png",

    dpi=300

)

plt.show()

plt.figure(figsize=(10,8))

plt.barh(

    top_frequency["Word"],

    top_frequency["Frequency"]

)

plt.title("Top 20 Word Frequencies")

plt.xlabel("Frequency")

plt.tight_layout()

plt.savefig(

    "images/document_frequency.png",

    dpi=300

)

plt.show()

result.to_csv(

    "document_word_similarity.csv",

    index=False

)

print("\nFiles Generated")

print("document_word_similarity.csv")

print("images/document_similarity.png")

print("images/document_frequency.png")

print("\nDone.")