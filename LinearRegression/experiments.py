import numpy as np
import time
import matplotlib.pyplot as plt
from ols_regression import LinearRegression
from synth_data_generator import generate_synthetic_data

SEED = 42
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_condition_sweep():
    kappa_list = [1, 10, 100, 1000, 10000, 1000000]
    
    print("\n--- Running Condition Number Sweep ---")
    print(f"{'Requested k':<12}{'Augmented k':<15}{'QR MSE':<12}{'GD MSE':<12}{'||w_GD - w_QR||':<15}")
    
    for k in kappa_list:
        X, y, _, true_w = generate_synthetic_data(n_samples=500, n_features=5, condition_number=k, seed=SEED)
        
        model = LinearRegression(fit_intercept=True)
        X_design = model._prepare_design_matrix(X)
        aug_k = np.linalg.cond(X_design)
        
        # Fit models
        model.fit_analytical(X, y)
        w_qr = model.w.copy()
        
        _, losses, distances = model.fit_gradient_descent(X, y, learning_rate=0.01, epochs=5000)
        w_gd = model.w.copy()
        
        mse_qr = 0.5*np.mean((X_design @ w_qr - y.reshape(-1,1))**2)
        mse_gd = 0.5*np.mean((X_design @ w_gd - y.reshape(-1,1))**2)
        
        print(f"{k:<12.1e}{aug_k:<15.2f}{mse_qr:<12.5f}{mse_gd:<12.5f}{distances[-1]:<15.5f}")

def run_learning_rate_experiment():
    print("\n--- Running Learning Rate Limit Experiment ---")
    X, y, _, _ = generate_synthetic_data(n_samples=500, n_features=5, condition_number=10, seed=SEED)
    
    model = LinearRegression(fit_intercept=True)
    X_design = model._prepare_design_matrix(X)
    
    
    eigenvalues = np.linalg.eigvalsh(X_design.T @ X_design) # Calculate fixed stability limit for loss representation
    lambda_max = eigenvalues[-1]
    eta_crit = (2.0 * X_design.shape[0]) / lambda_max
    
    lr_factors = [0.1, 0.9, 1.01, 1.5]
    plt.figure(figsize=(10, 6))
    
    for factor in lr_factors:
        lr = factor * eta_crit

        epochs = 100 if factor > 1.0 else 2000 # Cap epochs for unstable settings to avoid massive numerical explosion
        
        _, losses, _ = model.fit_gradient_descent(X, y, learning_rate=lr, epochs=epochs)
        
        if np.isnan(losses[-1]) or np.isinf(losses[-1]) or losses[-1] > losses[0]: # Check for NaN / Inf divergence safely
            print(f"Factor {factor}x eta_crit (lr={lr:.4f}) -> DIVERGED (NaN/Inf)")
            plt.plot(losses[:10], label=f"{factor}x eta_crit -> (Diverging)", linestyle='--')
        else:
            print(f"Factor {factor}x eta_crit (lr={lr:.4f}) -> Converged. Final Loss: {losses[-1]:.5f}")
            plt.plot(losses, label=f"{factor}x eta_crit")
            
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss ($L_2$ Empirical Risk)')
    plt.title('GD Convergence Profiles Relative to Spectral Bound eta_crit')
    plt.legend()
    plt.savefig('gd_learning_rates.png')
    plt.close()

def run_feature_scaling_experiment():
    print("\n--- Running Feature Scaling Experiment ---")
    X, y, _, _ = generate_synthetic_data(n_samples=500, n_features=3, condition_number=5, seed=SEED)
    
    
    X_badly_scaled = X.copy() # Intentionally malform scales
    X_badly_scaled[:, 0] *= 1e4
    X_badly_scaled[:, 1] *= 1e-4
    
    model = LinearRegression(fit_intercept=True)
    
    print("Fitting Unscaled Matrix via GD...") # Try GD on badly scaled matrix
    _, losses_unscaled, _ = model.fit_gradient_descent(X_badly_scaled, y, learning_rate=1e-9, epochs=3000)
    
    X_scaled = (X_badly_scaled - np.mean(X_badly_scaled, axis=0)) / np.std(X_badly_scaled, axis=0)
    print("Fitting Z-Score Standardised Matrix via GD...")
    _, losses_scaled, _ = model.fit_gradient_descent(X_scaled, y, learning_rate=0.05, epochs=3000)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_unscaled, label='Badly Scaled Data (lr=1e-9)', color='crimson')
    plt.plot(losses_scaled, label='Standardised Data (lr=0.05)', color='teal')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Impact of Feature Standardisation on GD Trajectory')
    plt.legend()
    plt.savefig('feature_scaling_impact.png')
    plt.close()

def run_noise_sweep():
    print("\n--- Running Noise Sweep (Parameter Recovery vs. Prediction Error) ---")
    noise_levels = [0.0, 0.1, 1.0, 10.0]
    # Set matrix to an interesting ill-conditioned geometry
    k_ill = 500 
    
    print(f"{'Noise Std':<12}{'QR Param Dist':<18}{'QR Test MSE':<15}")
    
    for noise in noise_levels:
        X_train, y_train, true_b, true_w = generate_synthetic_data(n_samples=400, n_features=4, 
        condition_number=k_ill, noise_std=noise, seed=SEED
        )
        X_test, _, _, _ = generate_synthetic_data(n_samples=100, n_features=4, 
        condition_number=k_ill, noise_std=noise, seed=SEED+1)
        
        model = LinearRegression(fit_intercept=True)
        model.fit_analytical(X_train, y_train)

        eps_test = np.random.default_rng(SEED+1).normal(0,noise,(100,1))
        y_test = (true_b + (X_test @true_w.reshape(-1,1)) + eps_test).flatten()
        
        """Account for design matrix trick shapes during distance calculation.
        true_w doesn't contain true intercept from generator, skip intercept for parameter comparison."""
        param_dist = np.linalg.norm(model.w[1:].flatten() - true_w)
        
        y_pred = model.predict(X_test)
        test_mse = np.mean((y_pred - y_test)**2)
        
        print(f"{noise:<12.1f}{param_dist:<18.5f}{test_mse:<15.5f}")

def run_runtime_comparison():
    print("\n--- Running Computational Cost Benchmark ---")
    configurations = [(1000, 10), (10000, 50), (50000, 100)]
    
    print(f"{'Dimensions (n x d)':<25}{'QR Time (s)':<15}{'GD Time (500 e) (s)':<20}")
    
    for n, d in configurations:
        X, y, _, _ = generate_synthetic_data(n_samples=n, n_features=d, condition_number=5, seed=SEED)
        model = LinearRegression(fit_intercept=True)
        
        # Benchmark QR
        start = time.perf_counter()
        model.fit_analytical(X, y)
        t_qr = time.perf_counter() - start
        
        # Benchmark GD
        start = time.perf_counter()
        model.fit_gradient_descent(X, y, learning_rate=0.01, epochs=500)
        t_gd = time.perf_counter() - start
        
        print(f"{f'{n} x {d}':<25}{t_qr:<15.4f}{t_gd:<20.4f}")

if __name__ == '__main__':
    run_condition_sweep()
    run_learning_rate_experiment()
    run_feature_scaling_experiment()
    run_noise_sweep()
    run_runtime_comparison()
    print("\nNumerical study complete. Performance figures saved to working directory.")
