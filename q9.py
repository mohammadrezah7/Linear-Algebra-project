import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("processed_dataset.csv")

print("=" * 60)
print("Question 9")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")

all_text = " ".join(df["Processed_Text"].astype(str))

print(f"\nTotal Characters: {len(all_text)}")


wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=300
).generate(all_text)


os.makedirs("images", exist_ok=True)

plt.figure(figsize=(16,8))

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.title("Word Cloud")

plt.tight_layout()

plt.savefig("images/wordcloud.png", dpi=300)

plt.show()

print("\nWord Cloud generated")