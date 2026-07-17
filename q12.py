import os
import numpy as np
import matplotlib.pyplot as plt

print("Question 12")

X_std = np.load("X_standardized.npy")
U = np.load("U.npy")
S = np.load("S.npy")
Vt = np.load("Vt.npy")

print(f"Standardized matrix shape : {X_std.shape}")
print(f"Number of singular values : {S.shape[0]}")

os.makedirs("images", exist_ok=True)

ranks = np.arange(1, len(S) + 1)

plt.figure(figsize=(10, 6))
plt.plot(ranks, S, marker="o")
plt.title("Singular Values (Scree Plot)")
plt.xlabel("Component Index")
plt.ylabel("Singular Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/singular_values.png", dpi=300)
plt.show()


x1, y1 = ranks[0], S[0]
x2, y2 = ranks[-1], S[-1]

num = np.abs((y2 - y1) * ranks - (x2 - x1) * S + (x2 * y1 - y2 * x1))
den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
distances = num / den

index = int(np.argmax(distances))
c = index + 1  

print(f"\nSuggested elbow point (threshold): keep top {c} components")
print(f"Singular value at the elbow : {S[index]:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(ranks, S, marker="o", label="Singular values")
plt.axvline(c, color="red", linestyle="--", label=f"Suggested cut c = {c}")
plt.title("Singular Values with Suggested Elbow Cut")
plt.xlabel("Component Index")
plt.ylabel("Singular Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/elbow.png", dpi=300)
plt.show()


U_trunc = U[:, :c]
S_trunc = S[:c]
Vt_trunc = Vt[:c, :]

print("\nTruncated SVD Matrix Shapes:")
print(f"U_trunc  (documents x concepts) : {U_trunc.shape}")
print(f"S_trunc  (concepts, 1D) : {S_trunc.shape}")
print(f"Vt_trunc (concepts x words) : {Vt_trunc.shape}")


X_approx = U_trunc @ np.diag(S_trunc) @ Vt_trunc

reconstruction_error_abs = np.linalg.norm(X_std - X_approx, ord="fro")
reconstruction_error_rel = reconstruction_error_abs / np.linalg.norm(X_std, ord="fro")

print("\nError:")
print(f"Absolute : {reconstruction_error_abs:.4f}")
print(f"Relative : {reconstruction_error_rel:.4%}")


np.save("U_trunc.npy", U_trunc)
np.save("S_trunc.npy", S_trunc)
np.save("Vt_trunc.npy", Vt_trunc)

