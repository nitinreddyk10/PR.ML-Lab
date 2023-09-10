import numpy as np
import math

# Define the histograms H1 and H2
H1 = np.array([0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04])
H2 = np.array([0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02])

# (a) KL Distance
def kl_distance(p, q):
    # Ensure both arrays have the same shape
    assert p.shape == q.shape, "Histograms must have the same shape"

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10

    # Calculate the KL distance
    kl_divergence = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    return kl_divergence

kl_distance_result = kl_distance(H1, H2)
print(f"(a) KL Distance: {kl_distance_result:.4f}")

# (b) Bhattacharyya Distance
def bhattacharyya_distance(p, q):
    # Ensure both arrays have the same shape
    assert p.shape == q.shape, "Histograms must have the same shape"

    # Calculate the Bhattacharyya coefficient
    bc_coefficient = np.sum(np.sqrt(p * q))

    # Calculate the Bhattacharyya distance
    bhattacharyya_dist = -np.log(bc_coefficient)

    return bhattacharyya_dist

bhattacharyya_distance_result = bhattacharyya_distance(H1, H2)
print(f"(b) Bhattacharyya Distance: {bhattacharyya_distance_result:.4f}")