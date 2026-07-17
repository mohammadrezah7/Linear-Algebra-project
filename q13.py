import numpy as np


def randomized_svd(A, rank, n_oversamples=10, n_iter=4, random_state=None):
    
    rng = np.random.default_rng(random_state)
    d, n = A.shape
    k = min(rank + n_oversamples, n)

    
    Omega = rng.standard_normal(size=(n, k))

    
    Y = A @ Omega

    
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Z = A.T @ Q
        Q_z, _ = np.linalg.qr(Z)
        Y = A @ Q_z

    
    Q, _ = np.linalg.qr(Y)

    
    B = Q.T @ A

    
    U_b, S, Vt = np.linalg.svd(B, full_matrices=False)

    
    U = Q @ U_b

    
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    return U, S, Vt


if __name__ == "__main__":
    print("=" * 60)
    print("Question 13")
    print("=" * 60)

    
    print("\nSanity check on a synthetic matrix")
    print("-" * 60)

    rng = np.random.default_rng(42)
    test_matrix = rng.standard_normal((200, 50))
    test_rank = 10

    U_exact, S_exact, Vt_exact = np.linalg.svd(test_matrix, full_matrices=False)
    U_rand, S_rand, Vt_rand = randomized_svd(
        test_matrix, rank=test_rank, n_oversamples=10, n_iter=4, random_state=0
    )

    print(f"Top {test_rank} singular values (exact SVD)     : {np.round(S_exact[:test_rank], 4)}")
    print(f"Top {test_rank} singular values (randomized SVD): {np.round(S_rand, 4)}")
    print(f"Max absolute difference                          : "
          f"{np.max(np.abs(S_exact[:test_rank] - S_rand)):.6f}")

    
    print("\nApplying Randomized SVD to our dataset")
    print("-" * 60)

    X_std = np.load("X_standardized.npy")
    S_trunc = np.load("S_trunc.npy")
    c = len(S_trunc)  

    print(f"Standardized matrix shape : {X_std.shape}")
    print(f"Target rank (from Q12)    : {c}")

    U_r, S_r, Vt_r = randomized_svd(X_std, rank=c, n_oversamples=10, n_iter=4, random_state=0)

    print("\nRandomized SVD Matrix Shapes:")
    print(f"U_random  (documents x concepts) : {U_r.shape}")
    print(f"S_random  (concepts, 1D)         : {S_r.shape}")
    print(f"Vt_random (concepts x words)     : {Vt_r.shape}")

    print("\nSingular values comparison (Truncated SVD vs Randomized SVD):")
    print(f"{'Idx':<5}{'Truncated SVD':<18}{'Randomized SVD':<18}")
    for i in range(c):
        print(f"{i+1:<5}{S_trunc[i]:<18.4f}{S_r[i]:<18.4f}")

    np.save("U_randomized.npy", U_r)
    np.save("S_randomized.npy", S_r)
    np.save("Vt_randomized.npy", Vt_r)

    print("\nFiles Generated:")
    print(" - U_randomized.npy")
    print(" - S_randomized.npy")
    print(" - Vt_randomized.npy")
