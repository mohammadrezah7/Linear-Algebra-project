import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("processed_dataset.csv")


print("Question 8")

print(f"Dataset Shape: {df.shape}")

all_text = " ".join(df["Processed_Text"].astype(str))
words = all_text.split()

print(f"\nTotal Number of Words: {len(words)}")
word_counter = Counter(words)
words_30 = word_counter.most_common(30)
print("\nTop 30 Most Words:\n")

for i, (word, count) in enumerate(words_30, start=1):
    print(f"{i:2d}. {word:<20} {count}")

word_names = []
for item in words_30:
    word_names.append(item[0])

word_counts = []
for item in words_30:
    word_counts.append(item[1])

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(14, 7))

plt.bar(word_names, word_counts)

plt.title("Top 30 Most Words")

plt.xlabel("Words")

plt.ylabel("Frequency")

plt.xticks(rotation=60)

plt.tight_layout()

plt.savefig("images/top30_words.png", dpi=300)

plt.show()

print("\nChart saved successfully.")