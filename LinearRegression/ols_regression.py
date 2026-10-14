# Implement multivariate ordinary least squares (OLS) regression from scratch using numpy.
import numpy as np

"""Instead of the OLS objective function minimized with respect to two separate entities ($w$ and $b$), 
requiring one compute two separate sets of partial derivatives, the design philosophy will be based on using 
the Design Matrix Trick turning the affine system $y = Xw + b$ into a pure linear system $y = Xw$. 
More detail in the associated notebook.
"""

class LinearRegression:
    def __init__(self, fit_intercept=True):
        """Multivariate OLS Linear Regression from scratch."""
        self.fit_intercept = fit_intercept
        self.w = None # Single parameter vector that combines weights and bias

    def _prepare_design_matrix(self, X):
        """Internal helper to force float64 and prepend intercept if required. 
        Can input a nested Python list, a Pandas DataFrame, or a standard NumPy array"""
        X = np.asarray(X, dtype=np.float64)

        """Apply the matrix design trick to simplify model design.
        We use X.shape[0] to count rows (samples) in the data, 
        and then np.ones to create a column vector of ones with np.hstack to glue column 
        onto left-hand side of feature matrix X.

        This has the benefit of transforming the matrix from an input shape of (n, d) 
        to a design matrix shape of (n, d + 1).
        """
        if self.fit_intercept:
            n_samples = X.shape[0]
            return np.hstack((np.ones((n_samples, 1)), X))
        return X

    def _solve_analytical(self, X_design, y):
        """Fit model using QR Decomposition such that X = Q * R
        with Q shape (n, d+1) and Q^T Q = I, and R upper-triangular matrix of shape (d+1, d+1).
                
        Since n > d, we set mode='reduced' so that all non-square matrices are handled efficiently 
        with (n_samples > n_features)."""

        Q,R = np.linalg.qr(X_design, mode='reduced')
        Qty = Q.T @ y # Project target vector onto orthogonal space created by Q.
        return np.linalg.solve(R, Qty).reshape(-1,1)

    def fit_analytical(self, X, y):
        X_design = self._prepare_design_matrix(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)  # Force column vector (n, 1)

        """Delegate computation to shared helper."""

        self.w = self._solve_analytical(X_design,y)
        return self
    
    def fit_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        """Fit the model using custom batch gradient descent."""
        X_design = self._prepare_design_matrix(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        
        n_samples, n_features = X_design.shape
        
        """Initialize weights (zeros is a standard baseline)."""
        self.w = np.zeros((n_features, 1))

        """Reuse shared diagnostic helper for target diagnostic tracking"""
        w_target = self._solve_analytical(X_design,y)

        loss_history = []
        distance_history = []

        """Implement the gradient descent loop."""
        for epoch in range(epochs):
            y_pred = X_design @ self.w # Compute predictions: shape (n,1)
            error = y_pred - y

            """Track mean squared loss"""
            current_loss = 0.5*np.mean(error **2)
            loss_history.append(current_loss)

            """Track weight distance ||w_t - w_optimal||_2"""
            dist = np.linalg.norm(self.w - w_target)
            distance_history.append(dist)

            """latex Compute the vectorized gradient of Mean Squared Error"""
            gradient = (1.0 / n_samples) * (X_design.T @ error)
            self.w -= learning_rate * gradient # Update unified parameter vector w
            
        return self, loss_history, distance_history

    def predict(self, X_design):
        """
        Predict targets using pure matrix multiplication: y_hat = Xw
        X_design: shape (m, d)
        """
        X_design = self._prepare_design_matrix(X_design)
        
        if self.w is None:
            raise ValueError("Model must be fitted before making predictions.")
        return (X_design @ self.w).flatten() # Flatten back to shape (m,)