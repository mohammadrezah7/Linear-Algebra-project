import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("Question 19")
print("=" * 60)

LABEL_NAMES = {
    0: "Politics",
    1: "Sport",
    2: "Technology",
    3: "Entertainment",
    4: "Business",
}

df_train = pd.read_csv("processed_dataset_train.csv")

U_trunc = np.load("U_trunc.npy")
S_trunc = np.load("S_trunc.npy")

print(f"Train documents shape : {df_train.shape}")
print(f"U_trunc shape          : {U_trunc.shape}")
print(f"S_trunc shape          : {S_trunc.shape}")


document_vectors = U_trunc @ np.diag(S_trunc)
num_concepts = document_vectors.shape[1]

print(f"Document vectors shape : {document_vectors.shape}")

labels = df_train["Label"].values

os.makedirs("images", exist_ok=True)

print("\n" + "-" * 60)
print("Average latent-space vector per category")
print("-" * 60)

category_ids = sorted(LABEL_NAMES.keys())
category_means = np.zeros((len(category_ids), num_concepts))

for row, label in enumerate(category_ids):
    mask = labels == label
    n_docs = int(mask.sum())
    category_means[row] = document_vectors[mask].mean(axis=0)
    print(f"{LABEL_NAMES[label]:<15} ({n_docs} train docs) : "
          f"{np.round(category_means[row], 3)}")

concept_cols = [f"C{i + 1}" for i in range(num_concepts)]
means_df = pd.DataFrame(
    category_means,
    index=[LABEL_NAMES[i] for i in category_ids],
    columns=concept_cols,
)

means_df.to_csv("category_mean_latent_vectors.csv")

plt.figure(figsize=(10, 5))
sns.heatmap(
    means_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Mean value along concept"},
)
plt.title("Average Latent-Space Vector per Category")
plt.xlabel("Concept (from Truncated SVD)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("images/category_mean_latent_heatmap.png", dpi=300)
plt.show()

print("\nHeat map saved to images/category_mean_latent_heatmap.png")
print("Mean vectors saved to category_mean_latent_vectors.csv")

print("\n" + "=" * 60)
print("Proposed labeling / classification method")
print("=" * 60)
print("""
We cannot rely on manually interpreting what each latent concept
"means" (as attempted, only heuristically, in Question 15) - that
does not scale to real applications. Instead we use the category
mean vectors themselves as reference points ("centroids") directly
in the latent space, without needing to know what any individual
concept represents:

  1. Project every document (train or test) into the same c-dimensional
     latent space used above: doc_vector = u_i * diag(S_trunc), where
     u_i is the row of U (or, for a brand-new/test document, its bag-of-
     words vector standardized with the SAME scaler fitted on the train
     set and then projected with doc_vector = bow_std @ Vt_trunc.T).
  2. For each of the 5 category centroids computed above, compute the
     Cosine Similarity (or, equivalently, the Euclidean Distance) between
     the document's latent vector and every centroid.
  3. Assign the document to the category whose centroid has the highest
     Cosine Similarity (lowest Euclidean Distance) to the document -
     i.e. a nearest-centroid classifier in latent space.

This "nearest centroid in latent space" approach is attractive because:
  - It only uses the *relative geometry* of the latent vectors (how
    close a document is to each category's average pattern), so it
    works even though we do not know the human-readable meaning of the
    individual concepts.
  - It is cheap: with c concepts and 5 categories, classifying a new
    document only costs a handful of dot products, instead of comparing
    against the full (much higher-dimensional and much larger) bag-of-
    words matrix.
  - It naturally generalizes: new documents just need to be embedded
    into the same latent space (via the fixed Vt_trunc / scaler learned
    on the training set) and compared to the 5 fixed centroids.

This method is implemented and evaluated on the held-out test set in
Question 20.
""")

print("Done.")
