import os
from collections import Counter
import pandas as pd
import numpy as np


print("Bag of Words")


df = pd.read_csv("processed_dataset.csv")
words = pd.read_csv("words.csv")

print(f"Dataset shape : {df.shape}")
print(f"Vocabulary shape : {words.shape}")
vocabulary = words.iloc[:, 0].astype(str).tolist()

print(f"\nVocabulary size : {len(vocabulary)}")

word_dictionary = {}
for index ,word in enumerate(vocabulary):
    word_dictionary[word] = index

num_documents = len(df)
num_words = len(vocabulary)

test1 = np.zeros((num_documents, num_words), dtype=int)

print("\nBuild Bag of Words Matrix ...")

for i,text in enumerate(df["Processed_Text"]):
    tokens = str(text).split()
    token_counts = Counter(tokens)
    for word,count in token_counts.items():
        if word in word_dictionary:
            column = word_dictionary[word]
            test1[i, column] = count

print("Done.")
newdf = pd.DataFrame(
    test1,
    columns=vocabulary
)

newdf.to_csv("bag_of_words.csv", index=False)
print("\nBag of Words saved")
print("\nMatrix Shape")
print(newdf.shape)
print("\nFirst 5 Documents")
print(newdf.head())

non_zero = np.count_nonzero(test1)
total = test1.size
density = (non_zero / total) * 100

print("\nMatrix info:")
print(f"Documents : {num_documents}")
print(f"Vocabulary Words : {num_words}")
print(f"Matrix Size : {test1.shape}")
print(f"Non-zero Elements : {non_zero}")
print(f"Density : {density:.4f}%")
print("\nFile Generated : Bag of Words")


bow_train = newdf.iloc[:2000, :].reset_index(drop=True)
bow_test = newdf.iloc[2000:, :].reset_index(drop=True)

df_train = df.iloc[:2000, :].reset_index(drop=True)
df_test = df.iloc[2000:, :].reset_index(drop=True)

bow_train.to_csv("bag_of_words_train.csv", index=False)
bow_test.to_csv("bag_of_words_test.csv", index=False)
df_train.to_csv("processed_dataset_train.csv", index=False)
df_test.to_csv("processed_dataset_test.csv", index=False)

print(f"Train set (first 2000 rows) shape : {bow_train.shape}")
print(f"Test set (remaining rows) shape    : {bow_test.shape}")
print("\nFiles Generated:")
print(" - bag_of_words_train.csv")
print(" - bag_of_words_test.csv")
print(" - processed_dataset_train.csv")
print(" - processed_dataset_test.csv")
print("\nNote: the test set is held out and should not be used again")
