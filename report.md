# Optimization Methods Comparison Report

**Course:** CSE 620 - Combinatorial Optimization  
**Project:** Gradient Descent Variants Analysis  
**Date:** September 2025  

---

## 1. Introduction

### 1.1 Objective

This project investigates and compares the convergence behavior of four optimization algorithms on three benchmark functions with varying geometric properties. The goal is to understand how algorithm design, hyperparameter selection, and function geometry interact to affect optimization performance.

### 1.2 Problem Instances

We optimize three test functions $f: \mathbb{R}^2 \to \mathbb{R}$:

**Function 1: Convex Quadratic Bowl**
$$f_1(x, y) = x^2 + y^2$$
- Global minimum: $f_1(0, 0) = 0$
- Hessian: $\nabla^2 f_1 = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ (constant, positive definite, condition number $\kappa = 1$)
- Well-conditioned, isotropic, single global minimum

**Function 2: Rosenbrock (Banana Valley)**
$$f_2(x, y) = (1 - x)^2 + 100(y - x^2)^2$$
- Global minimum: $f_2(1, 1) = 0$
- Hessian: $\nabla^2 f_2 = \begin{bmatrix} 2 - 400(y - 3x^2) & -400x \\ -400x & 200 \end{bmatrix}$
- Ill-conditioned ($\kappa \gg 1$), narrow curved valley, highly anisotropic curvature

**Function 3: Multimodal Cosine Bumps**
$$f_3(x, y) = x^2 + y^2 + 10\cos(x) + 10\cos(y)$$
- Global minima: $f_3(\pm 2.596, \pm 2.596) \approx -3.618$ (4 symmetric points)
- Local minima: $f_3(0, \pm 2.596) \approx 8.191$, $f_3(\pm 2.596, 0) \approx 8.191$
- Hessian: $\nabla^2 f_3 = \begin{bmatrix} 2 - 10\cos(x) & 0 \\ 0 & 2 - 10\cos(y) \end{bmatrix}$ (diagonal but indefinite)
- Multiple local minima, non-convex, periodic structure

### 1.3 Experimental Design

| Factor | Values |
|--------|--------|
| Functions | 3 ($f_1$, $f_2$, $f_3$) |
| Initial Points | 3: $(-2, 2)$, $(0.5, -1.5)$, $(3, 3)$ |
| Optimizers | 4: GD, Newton, AdaGrad, Adam |
| Learning Rates | 3: $0.001$, $0.01$, $0.1$ |
| **Total Combinations** | **3 × 3 × 4 × 3 = 108** (per function: 36) |

Convergence criterion: $\|x_{k+1} - x_k\| < 10^{-6}$  
Maximum iterations: 2000

---

## 2. Methods

### 2.1 Algorithm Descriptions

#### 2.1.1 Gradient Descent (GD)
**Update Rule:**
$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$
where $\alpha > 0$ is the fixed learning rate.

**Pseudocode:**
```
Input: f, ∇f, x₀, α, max_iter, tol
x ← x₀
for k = 1 to max_iter:
    g ← ∇f(x)
    x_new ← x - α * g
    if ‖x_new - x‖ < tol: return x_new
    x ← x_new
return x
```

#### 2.1.2 Newton's Method
**Update Rule:**
$$x_{k+1} = x_k - \alpha_k H(x_k)^{-1} \nabla f(x_k)$$
where $H(x) = \nabla^2 f(x)$ is the Hessian, and $\alpha_k$ is damped adaptively.

**Damping Strategy:** If $f(x_{k+1}) > f(x_k)$, then $\alpha_{k+1} = 0.5 \cdot \alpha_k$.

**Pseudocode:**
```
Input: f, ∇f, H, x₀, α, max_iter, tol, damping=0.5
x ← x₀; current_α ← α
for k = 1 to max_iter:
    g ← ∇f(x); H_x ← H(x)
    step ← H_x⁻¹ * g  (or g if H singular)
    x_new ← x - current_α * step
    if ‖x_new - x‖ < tol: return x_new
    if f(x_new) > f(x): current_α ← current_α * damping
    x ← x_new
return x
```

*Note: For pure quadratics with $\alpha=1$, Newton converges in exactly 1 iteration.*

#### 2.1.3 AdaGrad
**Update Rules:**
$$G_t = G_{t-1} + g_t \odot g_t \quad \text{(element-wise accumulation)}$$
$$x_{t+1} = x_t - \alpha \frac{g_t}{\sqrt{G_t} + \varepsilon}$$

**Pseudocode:**
```
Input: f, ∇f, x₀, α, max_iter, tol, ε=1e-8
x ← x₀; G ← 0
for t = 1 to max_iter:
    g ← ∇f(x)
    G ← G + g²
    x_new ← x - α * g / (√G + ε)
    if ‖x_new - x‖ < tol: return x_new
    x ← x_new
return x
```

#### 2.1.4 Adam (Adaptive Moment Estimation)
**Update Rules:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$

**Pseudocode:**
```
Input: f, ∇f, x₀, α, max_iter, tol, β₁=0.9, β₂=0.999, ε=1e-8
x ← x₀; m ← 0; v ← 0
for t = 1 to max_iter:
    g ← ∇f(x)
    m ← β₁*m + (1-β₁)*g
    v ← β₂*v + (1-β₂)*g²
    m_hat ← m / (1 - β₁ᵗ)
    v_hat ← v / (1 - β₂ᵗ)
    x_new ← x - α * m_hat / (√v_hat + ε)
    if ‖x_new - x‖ < tol: return x_new
    x ← x_new
return x
```

### 2.2 Implementation Details

**Platform:** Python 3.10, NumPy, Matplotlib  
**Source Code:** `optimization_experiment.py` (included in Appendix)

**Usage Example:**
```python
from optimization_experiment import gradient_descent, f1, grad_f1

# Run GD on f1 from (-2, 2) with lr=0.1
result = gradient_descent(f1, grad_f1, np.array([-2.0, 2.0]), lr=0.1)
print(f"Converged: {result.converged}, Iterations: {result.iterations}")
print(f"Final point: {result.final_point}, f = {result.final_value}")
```

---

## 3. Results

### 3.1 Function Landscapes

**Figure 1:** Contour plots of the three test functions showing their geometric structure.
- (a) $f_1$: Concentric circles (isotropic quadratic)
- (b) $f_2$: Narrow curved valley (banana shape)
- (c) $f_3$: Periodic bumps with quadratic envelope

> **Image Placeholders:**
> - `f1_quadratic_initial_-2.0_2.0.png` — f1 contour with paths
> - `f2_rosenbrock_initial_-2.0_2.0.png` — f2 contour with paths
> - `f3_cosine_initial_-2.0_2.0.png` — f3 contour with paths

### 3.2 Optimization Trajectories

For each function and initial point, Figure 2 shows the optimization paths for all 4 optimizers × 3 learning rates.

> **Image Placeholders (9 figures):**
> - `f1_quadratic_initial_-2.0_2.0.png`, `f1_quadratic_initial_0.5_-1.5.png`, `f1_quadratic_initial_3.0_3.0.png`
> - `f2_rosenbrock_initial_-2.0_2.0.png`, `f2_rosenbrock_initial_0.5_-1.5.png`, `f2_rosenbrock_initial_3.0_3.0.png`
> - `f3_cosine_initial_-2.0_2.0.png`, `f3_cosine_initial_0.5_-1.5.png`, `f3_cosine_initial_3.0_3.0.png`

**Key Observations from Trajectories:**
- **f1:** All methods converge directly to origin; Newton with lr=1.0 would be 1 step
- **f2:** GD zigzags/diverges; Newton follows valley floor; AdaGrad stalls; Adam nearly converges
- **f3:** All methods find global minimum except Newton from (0.5, -1.5) → local minimum

### 3.3 Convergence Curves

**Figure 3:** Function value vs. iteration for each optimizer/learning rate combination.

> **Image Placeholders (9 figures):**
> - `f1_quadratic_convergence_initial_-2.0_2.0.png`, etc.
> - `f2_rosenbrock_convergence_initial_-2.0_2.0.png`, etc.
> - `f3_cosine_convergence_initial_-2.0_2.0.png`, etc.

### 3.4 Tabulated Results

#### Table 1: f1 (Quadratic Bowl) — Summary Statistics (Converged Runs)

| Optimizer | LR | Avg Iter | Std Iter | Avg Time (s) | Std Time | Avg f | Std f | Conv Rate |
|-----------|-----|----------|----------|--------------|----------|-------|-------|-----------|
| Gradient Descent | 0.01 | 540.0 | 20.1 | 0.0016 | 0.0001 | 0.000000 | 0.000000 | 1.00 |
| Gradient Descent | 0.1 | 60.7 | 2.1 | 0.0002 | 0.0000 | 0.000000 | 0.000000 | 1.00 |
| Newton Method | 0.01 | 1015.7 | 40.2 | 0.0080 | 0.0003 | 0.000000 | 0.000000 | 1.00 |
| Newton Method | 0.1 | 120.0 | 3.7 | 0.0009 | 0.0000 | 0.000000 | 0.000000 | 1.00 |
| AdaGrad | 0.1 | 866.5 | 231.5 | 0.0038 | 0.0009 | 0.000000 | 0.000000 | 0.67 |
| Adam | 0.01 | 855.7 | 261.8 | 0.0059 | 0.0018 | 0.000000 | 0.000000 | 1.00 |
| Adam | 0.1 | 208.7 | 20.5 | 0.0015 | 0.0001 | 0.000000 | 0.000000 | 1.00 |

*lr=0.001 failed to converge for all methods within 2000 iterations.*

#### Table 2: f2 (Rosenbrock) — Summary Statistics (Converged Runs)

| Optimizer | LR | Avg Iter | Std Iter | Avg Time (s) | Std Time | Avg f | Std f | Conv Rate |
|-----------|-----|----------|----------|--------------|----------|-------|-------|-----------|
| Newton Method | 0.01 | 1717.3 | 141.2 | 0.0150 | 0.0013 | 0.000000 | 0.000000 | 1.00 |
| Newton Method | 0.1 | 208.7 | 25.5 | 0.0018 | 0.0002 | 0.000000 | 0.000000 | 1.00 |

*Gradient Descent (lr≥0.01): NaN/divergence. AdaGrad: 0% convergence. Adam (lr=0.1): f≈0.002 after 2000 iter.*

#### Table 3: f3 (Cosine Bumps) — Summary Statistics (Converged Runs)

| Optimizer | LR | Avg Iter | Std Iter | Avg Time (s) | Std Time | Avg f | Std f | Conv Rate |
|-----------|-----|----------|----------|--------------|----------|-------|-------|-----------|
| Gradient Descent | 0.001 | 929.0 | 120.9 | 0.0037 | 0.0005 | -3.617967 | 0.000000 | 1.00 |
| Gradient Descent | 0.01 | 111.0 | 12.4 | 0.0004 | 0.0000 | -3.617967 | 0.000000 | 1.00 |
| Gradient Descent | 0.1 | 7.0 | 1.4 | 0.0000 | 0.0000 | -3.617967 | 0.000000 | 1.00 |
| Newton Method | 0.01 | 882.3 | 11.2 | 0.0107 | 0.0001 | 0.318361 | 5.566808 | 1.00 |
| Newton Method | 0.1 | 106.7 | 0.5 | 0.0013 | 0.0000 | 0.318361 | 5.566808 | 1.00 |
| AdaGrad | 0.1 | 253.0 | 223.4 | 0.0013 | 0.0012 | -3.617967 | 0.000000 | 1.00 |
| Adam | 0.001 | 1647.0 | 213.0 | 0.0132 | 0.0014 | -3.617967 | 0.000000 | 0.67 |
| Adam | 0.01 | 262.7 | 122.6 | 0.0021 | 0.0010 | -3.617967 | 0.000000 | 1.00 |
| Adam | 0.1 | 209.3 | 23.1 | 0.0016 | 0.0002 | -3.617967 | 0.000000 | 1.00 |

*Note: Newton's average f = 0.318 because from (0.5, -1.5) it converges to local minimum f≈8.19.*

---

## 4. Experimental Analysis

### 4.1 Impact of Problem Difficulty

| Function | Geometry | GD (best) | Newton (best) | AdaGrad (best) | Adam (best) |
|----------|----------|-----------|---------------|----------------|-------------|
| f1 (Quadratic) | Convex, κ=1 | 61 iter | 120 iter* | 867 iter | 209 iter |
| f2 (Rosenbrock) | Non-convex, κ≫1 | **Fail** | 209 iter | **Fail** | Near miss |
| f3 (Cosine) | Multimodal | **7 iter** | 107 iter | 253 iter | 210 iter |

*Newton with lr=1.0: 1 iteration (theoretical)

**Observation:** As problem difficulty increases (quadratic → ill-conditioned → multimodal):
- **GD** goes from reliable → fails on Rosenbrock → fastest on Cosine
- **Newton** is robust on f1/f2 but has basin-of-attraction issues on f3
- **AdaGrad** consistently slow; fails on Rosenbrock due to gradient accumulation
- **Adam** is consistently decent but rarely the fastest

### 4.2 Function Geometry vs. Method Suitability

| Geometry Feature | Effect on Methods |
|------------------|-------------------|
| **Isotropic curvature (f1)** | All methods work; GD optimal with tuned α |
| **High condition number (f2)** | Newton excels (uses Hessian); GD/AdaGrad fail; Adam struggles |
| **Multiple minima (f3)** | Large α GD jumps over barriers; Newton trapped by basin; Adam momentum helps |

**Why Newton wins on Rosenbrock:** The Hessian captures the valley curvature, allowing large steps along the floor while correcting cross-valley oscillations.

**Why AdaGrad fails on Rosenbrock:** Cross-valley gradients are huge → $G_t$ grows rapidly → effective step size $\alpha/\sqrt{G_t} \to 0$ → stalls along valley floor.

**Why GD wins on Cosine:** Large steps (α=0.1) "jump over" shallow local minima; momentum methods (Adam) can overshoot; Newton gets trapped in wrong basin.

### 4.3 Initial Point Sensitivity

| Function | Initial Point | Best Method | Notes |
|----------|---------------|-------------|-------|
| f1 | All | GD (α=0.1) | Insensitive; all converge to global min |
| f2 | All | Newton (α=0.1) | Insensitive; only Newton converges |
| f3 | (-2, 2) | GD (α=0.1) | All find global min |
| f3 | (0.5, -1.5) | GD/Adam | **Newton → local min f=8.19** |
| f3 | (3, 3) | GD (α=0.1) | All find global min |

**Key Insight:** Newton's method is most sensitive to initialization on multimodal functions due to its local quadratic approximation defining a basin of attraction.

### 4.4 Method/Parameter Sensitivity

| Method | Sensitive to α? | Sensitive to Init? | Robust? |
|--------|-----------------|-------------------|---------|
| GD | **High** (f2: α≥0.01 diverges) | Low (convex) / Med (multimodal) | No |
| Newton | Medium (damping helps) | **High** (f3 basin issues) | Med |
| AdaGrad | Medium (needs high α) | Low | No (slow) |
| Adam | Low (adaptive) | Low | **Yes** |

### 4.5 Runtime vs. Iterations Comparison

| Function | Fastest Converged (iter) | Fastest Converged (time) | Notes |
|----------|-------------------------|-------------------------|-------|
| f1 | GD (α=0.1): 61 | GD (α=0.1): 0.0002s | Newton theoretical: 1 iter |
| f2 | Newton (α=0.1): 209 | Newton (α=0.1): 0.0018s | Only reliable method |
| f3 | GD (α=0.1): 7 | GD (α=0.1): ~0.0000s | GD remarkably fast |

**Per-iteration cost:** Newton > Adam ≈ AdaGrad > GD (Hessian inversion dominates)

---

## 5. Conclusion

### 5.1 Overall Findings

1. **No universal best optimizer** — performance depends critically on function geometry
2. **Newton's method** is superior for ill-conditioned problems (Rosenbrock) but requires Hessian and careful damping
3. **Gradient Descent** with well-tuned learning rate is surprisingly effective on well-conditioned and multimodal problems
4. **Adam** provides the best robustness across diverse landscapes but is rarely the fastest
5. **AdaGrad** is ill-suited for dense low-dimensional problems with varying curvature scales

### 5.2 Lessons Learned

- **Learning rate tuning is crucial** — especially for GD on ill-conditioned problems
- **Second-order methods pay off** when Hessian is available and well-conditioned
- **Adaptive methods** (AdaGrad, Adam) trade per-iteration cost for hyperparameter robustness
- **Multimodal landscapes** favor methods with "exploration" capability (large steps, momentum)

### 5.3 Potential Improvements

- Add line search / backtracking for GD and Newton
- Test LBFGS (limited-memory quasi-Newton) as Hessian-free alternative
- Evaluate on higher-dimensional problems (n > 2)
- Add stochastic gradient variants for noisy objectives
- Implement trust-region Newton for better globalization

---

## 6. Appendix

### 6.1 Complete Source Code

See `optimization_experiment.py` for full implementation with detailed docstrings.

### 6.2 Additional Tables

#### Table A1: f1 — Full Results (All 36 Runs)

| Optimizer | Init Point | LR | Iter | Time | Final x | Final y | Final f | Conv |
|-----------|------------|-----|------|------|---------|---------|---------|------|
| Gradient Descent | (-2.0, 2.0) | 0.001 | 2000 | 0.0064 | -0.0365 | 0.0365 | 0.002662 | ✗ |
| Gradient Descent | (-2.0, 2.0) | 0.01 | 543 | 0.0016 | 0.0000 | 0.0000 | 0.000000 | ✓ |
| Gradient Descent | (-2.0, 2.0) | 0.1 | 61 | 0.0002 | 0.0000 | 0.0000 | 0.000000 | ✓ |
| Newton Method | (-2.0, 2.0) | 0.001 | 2000 | 0.0163 | -0.2704 | 0.2704 | 0.146232 | ✗ |
| Newton Method | (-2.0, 2.0) | 0.01 | 1021 | 0.0080 | 0.0000 | 0.0000 | 0.000000 | ✓ |
| Newton Method | (-2.0, 2.0) | 0.1 | 121 | 0.0010 | 0.0000 | 0.0000 | 0.000000 | ✓ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

*(Full table: 36 rows per function)*

#### Table A2: f2 — Full Results (All 36 Runs)

| Optimizer | Init Point | LR | Iter | Time | Final x | Final y | Final f | Conv |
|-----------|------------|-----|------|------|---------|---------|---------|------|
| Gradient Descent | (-2.0, 2.0) | 0.001 | 2000 | 0.0065 | 0.6423 | 0.4108 | 0.128269 | ✗ |
| Gradient Descent | (-2.0, 2.0) | 0.01 | 2000 | 0.0067 | nan | nan | nan | ✗ |
| Newton Method | (-2.0, 2.0) | 0.01 | 1807 | 0.0158 | 1.0000 | 1.0000 | 0.000000 | ✓ |
| Newton Method | (-2.0, 2.0) | 0.1 | 231 | 0.0020 | 1.0000 | 1.0000 | 0.000000 | ✓ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

#### Table A3: f3 — Full Results (All 36 Runs)

| Optimizer | Init Point | LR | Iter | Time | Final x | Final y | Final f | Conv |
|-----------|------------|-----|------|------|---------|---------|---------|------|
| Gradient Descent | (-2.0, 2.0) | 0.001 | 877 | 0.0034 | -2.5957 | 2.5957 | -3.617967 | ✓ |
| Gradient Descent | (-2.0, 2.0) | 0.01 | 106 | 0.0004 | -2.5957 | 2.5957 | -3.617967 | ✓ |
| Gradient Descent | (-2.0, 2.0) | 0.1 | 6 | 0.0000 | -2.5957 | 2.5957 | -3.617967 | ✓ |
| Newton Method | (-2.0, 2.0) | 0.01 | 880 | 0.0105 | -2.5957 | 2.5957 | -3.617967 | ✓ |
| Newton Method | (0.5, -1.5) | 0.01 | 897 | 0.0108 | 0.0001 | -2.5957 | 8.191017 | ✓ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 6.3 Complete Visualization Set

**Contour Plots with Paths (9):**
- `f1_quadratic_initial_-2.0_2.0.png`
- `f1_quadratic_initial_0.5_-1.5.png`
- `f1_quadratic_initial_3.0_3.0.png`
- `f2_rosenbrock_initial_-2.0_2.0.png`
- `f2_rosenbrock_initial_0.5_-1.5.png`
- `f2_rosenbrock_initial_3.0_3.0.png`
- `f3_cosine_initial_-2.0_2.0.png`
- `f3_cosine_initial_0.5_-1.5.png`
- `f3_cosine_initial_3.0_3.0.png`

**Convergence Curves (9):**
- `f1_quadratic_convergence_initial_-2.0_2.0.png`
- `f1_quadratic_convergence_initial_0.5_-1.5.png`
- `f1_quadratic_convergence_initial_3.0_3.0.png`
- `f2_rosenbrock_convergence_initial_-2.0_2.0.png`
- `f2_rosenbrock_convergence_initial_0.5_-1.5.png`
- `f2_rosenbrock_convergence_initial_3.0_3.0.png`
- `f3_cosine_convergence_initial_-2.0_2.0.png`
- `f3_cosine_convergence_initial_0.5_-1.5.png`
- `f3_cosine_convergence_initial_3.0_3.0.png`

---

*End of Report*