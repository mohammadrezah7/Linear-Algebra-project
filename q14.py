import time
import numpy as np
from q13 import randomized_svd

print("=" * 60)
print("Question 14")
print("=" * 60)


X_std = np.load("X_standardized.npy")

U_trunc = np.load("U_trunc.npy")
S_trunc = np.load("S_trunc.npy")
Vt_trunc = np.load("Vt_trunc.npy")

U_rand = np.load("U_randomized.npy")
S_rand = np.load("S_randomized.npy")
Vt_rand = np.load("Vt_randomized.npy")

c = len(S_trunc)

print(f"Standardized matrix shape : {X_std.shape}")
print(f"Rank used (elbow, Q12)    : {c}")


X_approx_trunc = U_trunc @ np.diag(S_trunc) @ Vt_trunc
err_trunc_abs = np.linalg.norm(X_std - X_approx_trunc, ord="fro")
err_trunc_rel = err_trunc_abs / np.linalg.norm(X_std, ord="fro")


X_approx_rand = U_rand @ np.diag(S_rand) @ Vt_rand
err_rand_abs = np.linalg.norm(X_std - X_approx_rand, ord="fro")
err_rand_rel = err_rand_abs / np.linalg.norm(X_std, ord="fro")

print("\n" + "-" * 60)
print("Reconstruction Error Comparison")
print("-" * 60)
print(f"{'Method':<20}{'Absolute (Frobenius)':<25}{'Relative':<10}")
print(f"{'Truncated SVD':<20}{err_trunc_abs:<25.4f}{err_trunc_rel:.4%}")
print(f"{'Randomized SVD':<20}{err_rand_abs:<25.4f}{err_rand_rel:.4%}")
print(f"\nDifference in relative error : {abs(err_rand_rel - err_trunc_rel):.4%}")


print("\n" + "-" * 60)
print("Runtime Comparison (on our dataset)")
print("-" * 60)

n_runs = 5

start = time.perf_counter()
for _ in range(n_runs):
    np.linalg.svd(X_std, full_matrices=True)
time_exact = (time.perf_counter() - start) / n_runs

start = time.perf_counter()
for _ in range(n_runs):
    randomized_svd(X_std, rank=c, n_oversamples=10, n_iter=4, random_state=0)
time_random = (time.perf_counter() - start) / n_runs

print(f"Exact / Full SVD   average time : {time_exact*1000:.2f} ms")
print(f"Randomized SVD     average time : {time_random*1000:.2f} ms")
print(f"Speedup (exact / randomized)    : {time_exact / time_random:.2f}x")


print("\n" + "-" * 60)
print("Runtime Comparison (large synthetic matrix, for scale intuition)")
print("-" * 60)

rng = np.random.default_rng(0)
big_matrix = rng.standard_normal((8000, 500))
big_rank = 20

start = time.perf_counter()
np.linalg.svd(big_matrix, full_matrices=False)
time_exact_big = time.perf_counter() - start

start = time.perf_counter()
randomized_svd(big_matrix, rank=big_rank, n_oversamples=10, n_iter=4, random_state=0)
time_random_big = time.perf_counter() - start

print(f"Matrix shape                    : {big_matrix.shape}, target rank = {big_rank}")
print(f"Exact SVD time                   : {time_exact_big:.3f} s")
print(f"Randomized SVD time              : {time_random_big:.3f} s")
print(f"Speedup (exact / randomized)     : {time_exact_big / time_random_big:.2f}x")


print("\n" + "=" * 60)
print("Discussion: Truncated SVD vs Randomized SVD at internet scale")
print("=" * 60)
print("""
On our dataset (2000 x 52), the two methods produce almost identical
reconstruction errors (see comparison above) - the randomized method
loses very little accuracy despite only needing to look at a few
random projections of the data instead of factorizing the full
matrix.

If our dataset consisted of all texts on the internet, the
document-word matrix would be enormous (millions/billions of rows,
a very large vocabulary of columns). Exact/Truncated SVD needs
O(d * n * min(d, n)) time and full O(d * n) memory to even start,
which quickly becomes infeasible at that scale. Randomized SVD only
requires a small number of matrix-vector products with the (often
sparse) data matrix plus a tiny SVD of a much smaller projected
matrix, so its cost scales roughly linearly in the data size instead
of quadratically/cubically, and it also parallelizes/streams well.

For this reason, at internet scale we would recommend Randomized
SVD: it gives a reconstruction error that is close enough to the
exact/truncated solution for practical purposes (as shown above),
while being dramatically cheaper in both time and memory - the
scale of the speedup here already grows noticeably with the size of
the synthetic large matrix above, and this gap widens further as the
matrix grows toward internet scale.
""")
