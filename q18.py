import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

print("Question 18")

doc_id = 222

df = pd.read_csv("processed_dataset.csv")

U = np.load("U_trunc.npy")
S = np.load("S_trunc.npy")

doc_vectors = U @ np.diag(S)

query = doc_vectors[doc_id].reshape(1, -1)

scores = cosine_similarity(query, doc_vectors)[0]

order = np.argsort(scores)[::-1]

order = order[order != doc_id]

top_docs = order[:10]

result = []

for i in top_docs:

    result.append({
        "Document": i,
        "Similarity": scores[i],
        "Label": df.loc[i, "Label"]
    })

result = pd.DataFrame(result)

print("\nMost Similar Documents\n")
print(result)

result.to_csv("latent_search_result.csv", index=False)

print("\nQuery Document")

print(df.loc[doc_id, "Processed_Text"])


for i in top_docs:

    print(f"\nDocument {i}")
    print(f"Similarity : {scores[i]:.4f}")
    print(f"Label : {df.loc[i,'Label']}")
    print(df.loc[i, "Processed_Text"])
   
