import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

print("Question 16")

Vt = np.load("Vt_trunc.npy")

print(f"Vt shape : {Vt.shape}")

vocabulary = pd.read_csv("words.csv").iloc[:, 0].astype(str).tolist()

word_to_index = {
    word: idx
    for idx, word in enumerate(vocabulary)
}

word_pairs = [
    ("mobile", "technology"),
    ("director", "film"),
    ("win", "won"),
    ("play", "game"),
    ("play", "law"),
    ("government", "music")
]

print("\nResults\n")

print(f"{'Word 1':<15}{'Word 2':<15}{'Cosine':<15}{'Euclidean'}")

for w1, w2 in word_pairs:
    if w1 not in word_to_index or w2 not in word_to_index:
        print(f"{w1:<15}{w2:<15}Not Found")
        continue

    idx1 = word_to_index[w1]
    idx2 = word_to_index[w2]

    vec1 = Vt[:, idx1].reshape(1, -1)
    vec2 = Vt[:, idx2].reshape(1, -1)

    cos = cosine_similarity(vec1, vec2)[0, 0]

    dist = euclidean_distances(vec1, vec2)[0, 0]

    print(
        f"{w1:<15}"
        f"{w2:<15}"
        f"{cos:<15.4f}"
        f"{dist:.4f}"
    )
