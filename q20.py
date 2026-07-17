import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("Question 20")
print("=" * 60)

LABEL_NAMES = {
    0: "Politics",
    1: "Sport",
    2: "Technology",
    3: "Entertainment",
    4: "Business",
}

bow_train = pd.read_csv("bag_of_words_train.csv")
bow_test = pd.read_csv("bag_of_words_test.csv")
df_test = pd.read_csv("processed_dataset_test.csv")

Vt_trunc = np.load("Vt_trunc.npy")

means_df = pd.read_csv("category_mean_latent_vectors.csv", index_col=0)

print(f"Test documents shape : {bow_test.shape}")
print(f"Vt_trunc shape        : {Vt_trunc.shape}")
print(f"Centroids shape       : {means_df.shape}")

scaler = StandardScaler()
scaler.fit(bow_train.values.astype(float))
X_test_std = scaler.transform(bow_test.values.astype(float))

test_vectors = X_test_std @ Vt_trunc.T

category_ids = sorted(LABEL_NAMES.keys())
centroids = means_df.loc[[LABEL_NAMES[i] for i in category_ids]].values

similarities = cosine_similarity(test_vectors, centroids)
nearest = np.argmax(similarities, axis=1)
predicted_labels = np.array(category_ids)[nearest]

true_labels = df_test["Label"].values
correct = predicted_labels == true_labels

overall_accuracy = correct.sum() / len(true_labels)

print("\n" + "-" * 60)
print("Nearest-centroid labeling results on the test set")
print("-" * 60)
print(f"Overall Accuracy : {overall_accuracy:.4f}  ({correct.sum()}/{len(true_labels)})")

print("\nPer-category accuracy")
print("-" * 60)
for label in category_ids:
    mask = true_labels == label
    n_docs = int(mask.sum())
    if n_docs == 0:
        print(f"{LABEL_NAMES[label]:<15} (0 test docs)   : N/A")
        continue
    n_correct = int(correct[mask].sum())
    acc = n_correct / n_docs
    print(f"{LABEL_NAMES[label]:<15} ({n_docs} test docs) : {acc:.4f}  ({n_correct}/{n_docs})")

print("\nDone.")
