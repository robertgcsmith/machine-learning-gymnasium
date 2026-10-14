# Section 1: Linear regression and other notes

Linear regression is a statistical model to relate one or more explanatory variables $x_1, \dots, x_p$ to a single (scalar) response variable $y$, including an intercept term $\beta_0$. The assumption of the model is that the response variable is a linear combination of the explanatory variables plus a bias $ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p $.

By prepending a $1$ to our input vector to form $x_{new} = (1, x_1, \dots, x_p)^T$ and creating a unified parameter vector $\theta = (\beta_0, \beta_1, \dots, \beta_p)^T$, the above is written compactly in matrix notation as $y = x_{new}^T \theta$. Any disagreement between that and the reality is modelled through an error variable $\varepsilon$ so that the full model takes the form:

$$y = x_{new}^T \theta + \varepsilon.$$

If we have $n$ observations of the data, we prepend a column of ones to our original data matrix to form an $n \times (p+1)$ matrix $X_{new}$ and write

$$y = X_{new}\theta + \varepsilon.$$

That, at its most basic, is the linear regression model. From here on, there are several ways to estimate the values for the model parameters $\theta$. All of them require one to make certain assumptions on the data and/or the errors. The most popular and easiest approach concerns the ordinary least squares (OLS) estimate. To estimate the parameters using this method we make the following assumptions:

   1. The design matrix $X_{new}$ has full column rank.
   2. All the errors are independent and identically distributed according to a zero-mean normal distribution with variance $\sigma^2$ (i.e. $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$). We'll assume the value for $\sigma^2$ is known (or you can estimate it from the data).

The OLS method finds estimates for $\theta$ by minimising

$$S(\theta) = \Vert{}y - X_{new}\theta\Vert{}^2.$$

Under the above assumptions, the optimal estimate is

$$\hat{\theta} = (X_{new}^T X_{new})^{-1} X_{new}^T y.$$

## Design Matrix Trick

Instead of the OLS objective function minimised with respect to two separate entities $w$ and $b$, requiring one compute two separate sets of partial derivatives, in the associated implementation the design philosophy will be based on using the *design matrix trick*, wherein we prepend or append a column of ones such that the bias $b$ becomes just another weight $w_{0}$ inside a unified vector $\theta = [b, w_1, w_2, \dots, w_d]^T$.
This folds the separate affine transformation $y = Xw + b$ into a pure linear mapping $y = X_{new}\theta$. 

Consequently, the fundamental OLS normal equation can be written and solved in a single step

$$\theta =(X_{new}^{T}X_{new})^{-1}X_{new}^{T}y.$$

An added benefit of taking this approach is that, instead of writing custom formulas to handle the intercept separately, the exact same mathematical operation elegantly solves for both the weights and the bias simultaneously.

## Statistical Correction: OLS vs. Probabilistic Models

It is worth noting that OLS is fundamentally a geometric projection problem. To define the estimator $\hat{w} = \arg\min_w \Vert y - Xw \Vert_2^2$, no probabilistic assumptions regarding data distributions, error independence, or Gaussian noise are required. 

* **The Optimization View:** OLS strictly seeks to minimise the residual sum of squares by projecting the target vector onto the column space of the design matrix.
* **The Statistical View:** The assumption of independent and identically distributed (i.i.d.) Gaussian errors is only introduced downstream when interpreting the OLS estimator as a Maximum Likelihood Estimator (MLE), or when constructing finite-sample confidence intervals and hypothesis tests in classical inference.

# Section 2: Theoretical research 
## Derivation of Optimal Learning Rate and Contraction Bound for Symmetric Positive-Definite Quadratic Forms

We consider the optimisation of the quadratic objective function:
$$ L(w) = \frac{1}{2} w^\top H w - b^\top w + c, $$
where $H \in \mathbb{R}^{d \times d}$ is a symmetric positive-definite (SPD) matrix with real eigenvalues ordered as
$$ 0 < \lambda_{\min} = \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_d = \lambda_{\max}. $$

The gradient of $L(w)$ with respect to $w$ is
$$ \nabla L(w) = Hw - b. $$
Setting the gradient to zero yields the unique global analytical minimum $w^\star$:
$$ Hw^\star = b \implies w^\star = H^{-1}b. $$

Let us define the parameter error vector at step $t$ as $e_t = w_t - w^\star$. The fixed-step gradient descent update rule is
$$ w_{t+1} = w_t - \eta \nabla L(w_t) = w_t - \eta (Hw_t - b). $$
Substituting $b = Hw^\star$ yields
$$ w_{t+1} = w_t - \eta H(w_t - w^\star). $$
Subtracting $w^\star$ from both sides translates the optimisation update entirely into error coordinates
$$ e_{t+1} = e_t - \eta He_t = (I - \eta H)e_t .$$

Because $H$ is a symmetric matrix, it can be diagonalised via an orthogonal matrix of eigenvectors $Q$ (i.e., $Q^\top Q = I$) such that $H = Q \Lambda Q^\top$, where $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$. Substituting this into our error equation gives
$$ e_{t+1} = (I - \eta Q \Lambda Q^\top)e_t = Q(I - \eta \Lambda)Q^\top e_t. $$
Let us transform the error vector into the eigenbasis coordinates by defining $\tilde{e}_t = Q^\top e_t$. Multiplying from the left by $Q^\top$ uncouples the system completely
$$ \tilde{e}_{t+1} = (I - \eta \Lambda)\tilde{e}_t. $$

Because $(I - \eta \Lambda)$ is a purely diagonal matrix, the error dynamics decouple into $d$ independent scalar equations (one for each eigendirection/mode $i$):
$$ \tilde{e}_{t+1, i} = (1 - \eta \lambda_i)\tilde{e}_{t, i}. $$
The absolute value $|1 - \eta \lambda_i|$ represents the contraction factor of the error along the $i$-th eigendirection during a single update step. For the error to vanish in all directions, we must ensure $|1 - \eta \lambda_i| < 1$ for all $i$. 

### Minimising the Worst-Case Contraction Factor
The overall convergence rate of the system is governed entirely by the slowest-converging component—the worst-case contraction factor across all modes
$$ \rho(\eta) = \max_{i} |1 - \eta \lambda_i|. $$
Because the function $f(\lambda) = 1 - \eta \lambda$ is strictly linear and decreasing with respect to $\lambda$ (for $\eta > 0$), the maximum absolute value must occur at one of the two extreme structural boundaries of the spectrum: $\lambda_{\min}$ or $\lambda_{\max}$. Thus
$$ \rho(\eta) = \max \left( |1 - \eta \lambda_{\min}|, |1 - \eta \lambda_{\max}| \right). $$

To find the optimal learning rate $\eta^\star$ that minimises this worst-case boundary envelope, we observe the behavior of the two components:

1. As $\eta$ increases from zero, $1 - \eta \lambda_{\min}$ decreases down from $1$, improving convergence for the slow mode.
2. Simultaneously, $1 - \eta \lambda_{\max}$ drops rapidly, crosses zero, and becomes increasingly negative.

The optimal trade-off $\eta^\star$ occurs at the exact intersection point where the lower bound of the fast mode equals the upper bound of the slow mode, balanced with opposing signs:
$$ 1 - \eta^\star \lambda_{\min} = -(1 - \eta^\star \lambda_{\max}) $$
$$ 1 - \eta^\star \lambda_{\min} = -1 + \eta^\star \lambda_{\max} $$
$$ 2 = \eta^\star (\lambda_{\max} + \lambda_{\min}) $$
$$ \eta^\star = \frac{2}{\lambda_{\max} + \lambda_{\min}} $$

### Deriving the Optimal Contraction Factor ($\rho^\star$)
We substitute our optimal step size $\eta^\star$ back into the slow-mode contraction equation to find the guaranteed worst-case rate $\rho^\star$

$$ \rho^\star = 1 - \eta^\star \lambda_{\min} = 1 - \left( \frac{2}{\lambda_{\max} + \lambda_{\min}} \right) \lambda_{\min} $$
Combining the fractions over a common denominator
$$ \rho^\star = \frac{\lambda_{\max} + \lambda_{\min} - 2\lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}. $$

Dividing both the numerator and denominator by $\lambda_{\min}$, and substituting the definition of the condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$:
$$ \rho^\star = \frac{\frac{\lambda_{\max}}{\lambda_{\min}} - 1}{\frac{\lambda_{\max}}{\lambda_{\min}} + 1} = \frac{\kappa - 1}{\kappa + 1}. $$

#### Optimisation Insight
This mathematical relationship establishes a direct fundamental bridge between matrix conditioning and optimisation complexity. As $\kappa \to 1$ (a perfectly spherical bowl), $\rho^\star \to 0$, meaning gradient descent converges to the exact global minimum in a single step. However, if the condition number is large ($\kappa \to \infty$), we can approximate the rate using a Taylor expansion:
$$ \rho^\star = \frac{1 - \frac{1}{\kappa}}{1 + \frac{1}{\kappa}} \approx 1 - \frac{2}{\kappa} $$
This indicates that the contraction factor approaches $1$ exceptionally closely. The error decreases by a tiny fraction each iteration, providing mathematical proof for why poorly conditioned problems require $\mathcal{O}(\kappa)$ iterations to achieve convergence.

# Section 3: Comprehensive Empirical Analysis

In the associated implementation, we executed a comprehensive five-part numerical study to evaluate our solvers under varying parameters of sample volume ($n$), feature dimensionality ($d$), conditioning ($\kappa$), learning rate ($\eta$), scaling, and noise volatility $\sigma$.

## Numerical Solver Limitations & Boundary Edge Cases

Our implemented QR Decomposition method (`np.linalg.qr` with `mode='reduced'`) combined with `np.linalg.solve(R, Qty)` provides numerical precision for overdetermined, full-column-rank systems, it introduces explicit boundary failures under alternative mathematical constraints:

**Underdetermined Systems d > n:** When the number of features exceeds the data points, the reduced matrix $R$ is no longer square; it takes a rectangular shape of $n \times d$. Because `np.linalg.solve` strictly demands a square, non-singular coefficient matrix to complete back-substitution, the line `np.linalg.solve(R, Qty)` will throw a `LinAlgError` dimensional mismatch and crash entirely.

**Exact Rank Deficiency / Perfect Collinearity:** If two or more feature columns are perfectly collinear, the matrix $R$ will contain one or more zeros (or values within machine precision limits of zero) along its main diagonal. This makes $R$ structurally singular, causing the back-substitution algorithm to fail due to division-by-zero constraints.

*Production Alternative:* A general-purpose production-grade linear solver (such as `scikit-learn`'s backend) circumvents these edge cases by utilising SVD combined with a truncation threshold to compute the Moore-Penrose pseudoinverse $X^+$, allowing it to naturally select the unique minimum $L_2$-norm solution in rank-deficient scenarios.

## The Condition Number Sweep & Parameter Recovery Traps
We subjected both solvers to a geometric sweep where the matrix conditioning was varied across six orders of magnitude $\kappa \in \{1, 10^6\}$.

| Requested $\kappa(X)$ | Augmented $\kappa(X_{\text{design}})$ | QR Final MSE | GD Final MSE | Parameter Distance $\Vert w_{GD} - w_{QR} \Vert_2$ |
| :--- | :--- | :--- | :--- | :--- |
| $1.0 \times 10^0$ | $22.51$ | $0.01033$ | $0.07355$ | $5.63360$ |
| $1.0 \times 10^1$ | $224.43$ | $0.01033$ | $0.01842$ | $5.93733$ |
| $1.0 \times 10^2$ | $2,244.31$ | $0.01033$ | $0.01166$ | $8.12536$ |
| $1.0 \times 10^3$ | $22,443.14$ | $0.01033$ | $0.01065$ | $75.50815$ |
| $1.0 \times 10^4$ | $224,431.40$ | $0.01033$ | $0.01045$ | $766.29301$ |
| $1.0 \times 10^6$ | $22,443,139.56$ | $0.01033$ | $0.01040$ | $76,532.37349$ |

## Empirical Diagnosis:

The experiments show a noticeable inflation of the condition number when moving from the raw matrix $\kappa(X)$ to the augmented design matrix $\kappa(X_{\text{design}})$. 

Rather than an inherent mathematical pathology of linear regression intercepts, this shift is an artifact of scale discrepancy unique to this synthetic data construction. The largest singular value of our generated feature matrix $X$ is normalized near $1$, whereas the appended column of ones holds a vector norm of $\sqrt{500} \approx 22.36$. The observed condition inflation is primarily a reflection of this relative scale mismatch, illustrating how unscaled structural column additions can subtly alter the conditioning profile of an optimisation problem.

**Optimisation Velocity Collapse:** While the QR solver resolves the system analytically at the global minimum instantly (MSE $= 0.01033$) regardless of the matrix conditioning, Batch Gradient Descent becomes progressively paralyzed. At $\kappa = 10^6$, the parameter divergence vectors explode to $76,532.37$. This occurs because the minimum eigenvalue ($\lambda_{\min}$) of the Hessian drops towards zero, forcing the gradient floor slope to turn incredibly flat and preventing the optimisation updates from moving down the valley corridor.

## Learning Rate Dynamics and Spectral Boundary Verification
Using our corrected Hessian formulation ($H = \frac{1}{n}X^\top X$), we calculated the exact theoretical maximum stable step size boundary:
$$\eta_{\text{crit}} = \frac{2}{\lambda_{\max}(H)} = \frac{2n}{\lambda_{\max}(X^\top X)}$$
We tested four learning rates relative to this exact critical bound:
* **$0.1 \times \eta_{\text{crit}}$ ($\eta = 0.2000$):** Smooth, monotonic convergence. Reached a final loss baseline of $0.01686$.
* **$0.9 \times \eta_{\text{crit}}$ ($\eta = 1.8000$):** Faster convergence, aggressively riding the step size boundary. Achieved a superior loss of $0.01182$.
* **$1.01 \times \eta_{\text{crit}}$ ($\eta = 2.0200$):** **Divergence Initiated.** Over 100 epochs, the step size consistently overshot the quadratic loss bowl walls, inflating the final training error to $467.82227$.
* **$1.5 \times \eta_{\text{crit}}$ ($\eta = 3.0000$):** **Catastrophic Geometric Explosion.** The system entered a recursive feedback oscillation loop, causing the empirical risk to explode catastrophically to a magnitude of $\approx 3.7 \times 10^{60}$.


## Feature Scaling as a Loss Landscape Preconditioner
Our experiment tested an intentionally poorly scaled matrix where individual columns were multiplied by conflicting geometric factors ($10^4$ and $10^{-4}$).
* **Mechanics:** Transforming the features via Z-score normalisation scales columns to a uniform mean of 0 and variance of 1. 
* **Theoretical Softening:** It is important to emphasise that feature standardisation does **not** inherently guarantee an isotropic Hessian ($\kappa \approx 1$). While standardisation equalizes marginal feature scales and often drastically reduces conditioning problems caused by disparate feature scales, it does not remove underlying correlations between features. 


## Noise Sweeps: Decoupling Parameter Recovery from Prediction Error
Noise sweep demonstrates that when measurement noise is exactly zero $\sigma = 0$, parameter recovery error and generalisation test MSE reach numerical zero. 

As noise scales up under an ill-conditioned regime $\kappa = 500$, the parameter distance vector expands dramatically:

| Noise Standard Deviation $\sigma$ | Parameter Recovery Error $\Vert \hat{w} - w^\star \Vert_2$ | Prediction Generalisation Error (Test MSE) |
| :--- | :--- | :--- |
| \(0.0\) | \(0.00000\) | \(0.00000\) |
| \(0.1\) | \(26.04355\) | \(1.89892\) |
| \(1.0\) | \(260.43548\) | \(189.89173\) |
| \(10.0\) | \(2,604.35479\) | \(18,989.17333\) |

### Conceptual Insights
This sweep outlines the clear boundary between the **optimisation problem**, the **statistical model**, and the **estimator**. OLS acts purely as a projection matrix onto a column subspace. Under high noise paired with near-collinear features, the model can still project vector configurations to achieve stable predictions. However, because multiple linear combinations can generate the exact same structural output, identifying the true underlying data coordinates ($w^\star$) degrades exponentially by a scale factor of **$2,604.35$**.

## Empirical Computational Runtime Metrics
We measured the execution speed across three distinct matrix scaling configurations:

--- Running Computational Cost Benchmark ---
Dimensions (n x d)       QR Time (s)    GD Time (500 e) (s) 
1000 x 10                0.0003         0.0104              
10000 x 50               0.0182         0.1754              
50000 x 100              0.1984         0.6865 

## Complexity Trade-Offs:
The QR decomposition solves the system analytically in a single deterministic pass with a complexity scale of $\mathcal{O}(nd^2)$. For configurations where $d$ is relatively small, it remains incredibly fast. However, because its complexity scales quadratically with feature volume ($d^2$), it becomes highly resource-intensive as feature dimensions scale. Conversely, Batch Gradient Descent scales strictly linearly per epoch ($\mathcal{O}(\text{epochs} \cdot nd)$), highlighting its utility as an alternative optimiser choice for high-dimensional feature systems.

# Section 4: Further experimental observations and notes

1. Derive the OLS estimator from the empirical risk 

Define your empirical risk function strictly with the 

$$L(w)=\frac{1}{2n}\|{}Xw-y\|{}_{2}^{2}=\frac{1}{2n}(Xw-y)^{\top }(Xw-y).$$

Expanding this yields

$$L(w)=\frac{1}{2n}\left(w^{\top }X^{\top }Xw-2w^{\top }X^{\top }y+y^{\top }y\right).$$

Taking the gradient with respect to the vector $w$

$$\nabla _{w}L(w)=\frac{1}{2n}\left(2X^{\top }Xw-2X^{\top }y\right)=\frac{1}{n}X^{\top }(Xw-y).$$

2. The gradient used by the optimiser 

Following the derivation above, the gradient of your loss function is exactly the first derivative vector

$$\nabla _{w}L=\frac{1}{n}X^{\top }(Xw-y)$$

3. Explicitly computing $(X^\top X)^{-1}$: Squaring $X$ to form the normal equations matrix $X^{\top }X$ squares the condition number: $\kappa(X^\top X) = \kappa(X)^2$. For your poorly conditioned matrix where $\kappa(X) = 10^7$, computing the normal equations inflates the condition number to $10^{14}$. This is right at the edge of 64-bit floating-point precision limit $\sim 10^{16}$, causing numerical round-off errors and loss of precision.

4. Uniqueness of the ordinary least squares (OLS) solution: The OLS solution is unique if and only if the design matrix $X \in \mathbb{R}^{n \times d}$ has full column rank, meaning

$$\text{rank}(X)=d$$

**Proof**: The critical step in finding the unique analytical solution requires inverting the matrix $X^\top X \in \mathbb{R}^{d \times d}$. By basic algebra, the product matrix $X^{\top }X$ is invertible if and only if $X$ has linearly independent columns (full column rank). If $X$ does not have full column rank (i.e., $\text{rank}(X) < d$), it implies that at least one feature column can be written as a perfect linear combination of the others. In this case, $X^{\top }X$ becomes singular (non-invertible). The null space of $X$ becomes non-trivial $\text{null}(X) \neq \{0\}$, meaning there exists an infinite number of parameter vectors $w$ that yield the exact same predictions $\hat{y} = Xw$, resulting in infinitely many global minima.  

5. How does the condition number of $X$ relate to that of $X^{\top }X$?

The condition number of the normal equations matrix is the square of the condition number of the design matrix

$$\kappa (X^{\top }X)=\kappa (X)^{2}$$

**Proof via SVD**: Let the Singular Value Decomposition (SVD) of $X$ be

$$X=U\Sigma V^{\top },$$

where $U$ and $V$ are orthogonal matrices $U^\top U = I$, $V^\top V = I$, and $\Sigma $ is a diagonal matrix containing the singular values of $X$ sorted in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_d$.

By definition, the condition number of $X$ under the $L_{2}$ norm is the ratio of its largest singular value to its smallest non-zero singular value

$$\kappa (X)=\frac{\sigma _{\max }}{\sigma _{\min }}=\frac{\sigma _{1}}{\sigma _{d}}.$$

Computing $X^{\top }X$ gives

$$X^{\top }X=(U\Sigma V^{\top })^{\top }(U\Sigma V^{\top })=V\Sigma ^{\top }U^{\top }U\Sigma V^{\top }.$$

Since $U$ is orthogonal $U^\top U = I$, this simplifies to

$$X^{\top }X=V\Sigma ^{2}V^{\top }.$$

This is the eigendecomposition of $X^{\top }X$. The eigenvalues $\lambda$ of $X^{\top }X$ are exactly the squares of the singular values of $X$: $\lambda_i = \sigma_i^2$.The condition number of this symmetric matrix is the ratio of its maximum eigenvalue to its minimum eigenvalue

$$\kappa (X^{\top }X)=\frac{\lambda _{\max }}{\lambda _{\min }}=\frac{\sigma _{1}^{2}}{\sigma _{d}^{2}}=\left(\frac{\sigma _{1}}{\sigma _{d}}\right)^{2}=\kappa (X)^{2}. $$

Empirical Note: This explains why your Experiment 2 broke down so dramatically. When you set $\kappa(X) = 10^7$, forming the normal equations would yield $\kappa(X^\top X) = 10^{14}$, which immediately destroys numerical precision in standard float64 operations.

6. What determines the largest stable learning rate for gradient descent on this problem?

The maximum stable learning rate is strictly determined by the maximum eigenvalue $\lambda_{\max}$ of the Hessian matrix of the objective function.

Under the unified convention matching our empirical implementation:

$$L(w) = \frac{1}{2n} \Vert Xw - y \Vert_2^2$$

Taking the derivative with respect to \(w\) yields the gradient vector:
$$nabla_w L(w) = \frac{1}{n} X^\top (Xw - y)$$

Differentiating a second time isolates the true Hessian matrix

$$H = \nabla^2_w L(w) = \frac{1}{n} X^\top X$$

Fixed-step gradient descent forms a linear dynamical system around the global minimum. To ensure the parameter errors vanish rather than oscillating to infinity, the step size $\eta$ must be bounded by the spectral radius of the Hessian:

$$0 < \eta < \frac{2}{\lambda_{\max}(H)} \implies 0 < \eta < \frac{2n}{\lambda_{\max}(X^\top X)}$$

If $\eta$ crosses this strict upper theoretical threshold, the update dynamics will overshoot the boundary walls of the quadratic loss bowl, triggering immediate numerical overflow errors (`NaN` / `Inf`).


7. How scaling affect gradient-descent convergence while leaving the underlying OLS problem essentially unchanged: 

Feature scaling behaves as a geometric preconditioner that balances the landscape of the loss function. 

**Effect on the OLS Problem:** Scaling features is mathematically equivalent to multiplying your design matrix from the right by a diagonal scaling matrix $D$, transforming it to $\tilde{X} = XD$. Because $D$ is invertible, the column space remains entirely unchanged. The analytical minimum updates adaptively $\tilde{w} = D^{-1}w_{OLS}$, meaning the final underlying model achieves the exact same predictions and predictive error.

**Effect on Gradient Descent:** Without scaling, if one feature spans a range of $[0, 10^6]$ and another spans $[0, 1]$, the singular values of $X$ diverge massively due purely to scale mismatch, inflating the condition number. The loss landscape turns into an incredibly steep, narrow elliptical canyon, causing gradient descent to bounce violently back and forth across the steep walls rather than traveling efficiently down the gentle canyon floor. 

Feature scaling often dramatically reduces conditioning problems caused by disparate feature scales by regularising these axes. However, it is important to qualify that standardisation equalizes only the *marginal* feature scales; it does not remove underlying structural correlations between features. Therefore, it does not inherently guarantee an isotropic Hessian ($\kappa \approx 1$) or perfectly spherical loss bowls in general settings, though it remains essential for enabling larger, stable learning rates and accelerating initial convergence.
 

8. If $d > n$, what changes mathematically?

When $d > n$, the system contains more free parameters than observational constraints. However, the presence of more features than data points alone **does not guarantee** that an interpolating solution with zero training error exists. 

* **The Rank Condition:** For an interpolating solution ($L(w) = 0$) to exist, the target vector $y$ must reside completely within the column space of the design matrix: $y \in \operatorname{col}(X)$. 
* **Full Row Rank Scenario:** If $\operatorname{rank}(X) = n$ (full row rank), then the column space spans the entire space $\mathbb{R}^n$ ($\operatorname{col}(X) = \mathbb{R}^n$). In this case, an infinite number of zero-error interpolating solutions are guaranteed to exist for *any* target vector $y$.
* **Rank-Deficient Scenario:** If $\operatorname{rank}(X) < n$ (due to perfectly collinear rows or dead features), the column space is a lower-dimensional subspace of $\mathbb{R}^n$. If the noise or target vector $y$ contains components orthogonal to this subspace ($y \notin \operatorname{col}(X)$), it is mathematically impossible to achieve an interpolation of zero training error, even though $d > n$. In this edge case, the system remains a traditional least-squares projection problem on a rank-deficient space.


9. If infinitely many interpolating solutions exist, which one does the Moore–Penrose pseudoinverse select?

The Moore–Penrose pseudoinverse $X^{+}$ explicitly selects the unique solution that possesses the minimum $L_{2}$ norm. Mathematical Definition: Out of the infinite set of solutions 
$S = {w : Xw = y}$, the pseudoinverse returns

$$w^{+}=X^{+}y=\arg \min _{w\in S}\|w\|_{2}.$$

This occurs due to its structural geometric properties. The pseudoinverse maps the target vector $y$ entirely onto the row space of $X$ $\text{row}(X)$. Because it sets any coordinate vector pointing into the null space $\text{null}(X)$ cleanly to zero, it strips away any unnecessary scalar weight additions, leaving the structurally shortest possible weight vector. 


10. Suppose gradient descent starts at $w_0 = 0$ in an underdetermined linear regression problem and converges to an interpolating solution. Which solution does it select, and why?

Gradient descent will converge to the exact same minimum $L_2$-norm solution selected by the Moore-Penrose pseudoinverse ($w_{GD} = w^+$).

**Proof**: Under our implemented loss convention 

$L(w) = \frac{1}{2n}| Xw - y |_2^2$,

the gradient is defined as

$ \nabla_w L = \frac{1}{n} X^\top(Xw - y) = X^\top \left[ \frac{1}{n}(Xw - y) \right]. $

Notice that the term $\frac{1}{n}(Xw - y)$ is simply a vector in $\mathbb{R}^n$. Let us denote this vector as $\alpha$. Therefore, any gradient vector can be expressed as
$ \nabla_w L = X^\top \alpha. $

This indicates that every single gradient step is formed strictly by a linear combination of the rows of $X$ (since $X^\top \alpha \in \text{row}(X)$). 

If we initialise gradient descent at the origin ($w_0 = 0$), then $w_0$ naturally sits within the row space of $X$. Because every subsequent optimisation update $-\eta \nabla_w L$ is also confined to $\text{row}(X)$, the weight vector can never accumulate any components inside the null space ($\text{null}(X)$). It is mathematically locked into the row space for its entire lifespan, implicitly solving the exact norm-minimisation problem handled by the pseudoinverse.