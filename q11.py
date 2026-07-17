import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


print("Question 11")



bow_train = pd.read_csv("bag_of_words_train.csv")

print(f"Bag of Words (train) shape : {bow_train.shape}")

X = bow_train.values.astype(float)


scaler = StandardScaler()
X_std = scaler.fit_transform(X)

np.save("X_standardized.npy", X_std)

print("\nStandardization done (mean=0, std=1 per column).")
print(f"Standardized matrix shape : {X_std.shape}")

U, S, Vt = np.linalg.svd(X_std, full_matrices=True)

print("\nSVD Matrix Shapes:")
print(f"U  (documents x documents) : {U.shape}")
print(f"S  (singular values, 1D) : {S.shape}")
print(f"Vt (words x words) : {Vt.shape}")


np.save("U.npy", U)
np.save("S.npy", S)
np.save("Vt.npy", Vt)

