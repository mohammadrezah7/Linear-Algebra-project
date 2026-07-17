import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Question 15")

Vt_trunc = np.load("Vt_trunc.npy")
words = pd.read_csv("words.csv").iloc[:, 0].astype(str).tolist()

num_components, num_words = Vt_trunc.shape
print(f"Vt_trunc shape : {Vt_trunc.shape}")
print(f"Number of components : {num_components}")
print(f"Vocabulary size : {num_words}")

os.makedirs("images", exist_ok=True)

top_k = 5
component_top_words = []  

print("Top 5 words per component (by magnitude)")

for j in range(num_components):
    weights = Vt_trunc[j, :]
    top_indices = np.argsort(-np.abs(weights))[:top_k]

    top_words = []
    top_weights = []

    for i in top_indices:
        top_words.append(words[i])
        top_weights.append(weights[i])

    component_top_words.append(top_words)

    print(f"\nComponent {j + 1}:")
    for w, val in zip(top_words, top_weights):
        sign = "+" if val >= 0 else "-"
        print(f"  {w:<15} weight = {sign}{abs(val):.4f}")

    
    plt.figure(figsize=(8, 4))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in top_weights]
    plt.bar(top_words, top_weights, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Component {j + 1} - Top {top_k} Words")
    plt.ylabel("Weight")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"images/component_{j + 1}_top_words.png", dpi=200)
    plt.close()

print(f"\nSaved {num_components} component bar charts to images/component_<i>_top_words.png")


cols = 4
rows = int(np.ceil(num_components / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes = np.array(axes).reshape(-1)

for j in range(num_components):
    weights = Vt_trunc[j, :]
    top_indices = np.argsort(-np.abs(weights))[:top_k]
    top_words = [words[i] for i in top_indices]
    top_weights = [weights[i] for i in top_indices]
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in top_weights]

    ax = axes[j]
    ax.bar(top_words, top_weights, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Component {j + 1}", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)


for j in range(num_components, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("images/all_components_top_words.png", dpi=200)
plt.show()

themes = {
    "Politics": {"election", "government", "labour", "minister", "party", "law", "tax", "uk"},
    "Sport": {"win", "won", "match", "play", "club", "game"},
    "Technology": {"mobile", "phone", "digital", "online", "computer", "technology",
                   "user", "service", "market", "industry"},
    "Business": {"company", "firm", "market", "economy", "growth", "deal", "money",
                 "million", "sale", "office", "group", "new", "help", "expected", "number"},
    "Entertainment": {"film", "music", "director", "artist", "record"},
}

print("Discussion: guessing the latent meaning of each component")

print("""
Each component below is labeled with the theme its top-5 words
overlap with most (based on a rough grouping of our vocabulary into
politics / sport / technology / business / entertainment). This is
only a heuristic to help interpret the components - SVD components
are not required to align with human-defined categories, only to
capture directions of maximum variance in the data, so some
components mix words from different themes or are dominated by
generic words and are harder to label cleanly.
""")

for j, top_words in enumerate(component_top_words):
    top_words_set = set(top_words)
    scores = {theme: len(top_words_set & vocab) for theme, vocab in themes.items()}
    best_theme = max(scores, key=scores.get)
    best_score = scores[best_theme]

    if best_score == 0:
        guess = "no clear single theme (mixed / generic words)"
    else:
        guess = f"looks like a '{best_theme}' concept ({best_score}/{top_k} words match)"

    print(f"Component {j + 1:>2} - words: {top_words}")
    print(f"-> {guess}\n")
