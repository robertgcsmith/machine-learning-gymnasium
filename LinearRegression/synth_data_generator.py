import numpy as np

"""We construct synthetic datasets to test the system experimentally. Important in the experimental design is to
be able to manipulate n (number of samples), d (number of features), noise level (Gaussian noise added to y),
and conditioning (deliberately making X poorly conditioned).

The design philosophy is to take a rigorous approach using synthetic SVD reconstruction, 
which allows us to first generate random components and multiply them together to construct X.
"""

def generate_synthetic_data(n_samples, n_features, condition_number=1.0, scale=1.0, noise_std=0.1, 
                            seed=42, true_intercept=None, true_weights=None):
    
    rng = np.random.default_rng(seed)

    """Generate random orthogonal matrices U and Vt via QR decomposition
    (Note: QR of a random matrix yields a perfectly uniform orthogonal matrix)."""

    A = rng.normal(0, 1, (n_samples, n_features))
    U, _ = np.linalg.qr(A, mode='reduced')
    
    B = rng.normal(0, 1, (n_features, n_features))
    V, _= np.linalg.qr(B, mode='reduced') # V needs to be orthogonal
    V_T = V.T # Take transpose for SVD
    
    """Construct Singular Values based on the targeted condition number
    We logarithmically space values between 1 and (1 / condition_number)."""

    if condition_number < 1.0:
        raise ValueError("Condition number must be >= 1.0")
        
    singular_values = np.logspace(0, -np.log10(condition_number), num=n_features)
    Sigma = np.diag(singular_values)
     
    X = (U @ Sigma @ V_T) * scale # Reconstruct X and apply external feature scaling
    
    """Generate ground-truth weights (including an implicit intercept).
    The true model is y = true_intercept + X @ true_weights."""
    if true_intercept is None:
        true_intercept = rng.uniform(-5,5)
    if true_weights is None:
        true_weights = rng.uniform(-5,5,(n_features,1))
    else:
        true_weights = np.asarray(true_weights, dtype=np.float64).reshape(-1,1)

    """Target vector y with standard measurement noise"""
    noise = rng.normal(0, noise_std, (n_samples, 1)) # Compute y with added Gaussian noise
    y = true_intercept + (X @ true_weights) + noise
    
    return X, y.flatten(), true_intercept, true_weights.flatten()
