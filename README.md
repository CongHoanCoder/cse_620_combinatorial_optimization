# 📊 Optimization Methods Analysis Report

## 📋 Executive Summary

This report analyzes the convergence behavior of four optimization methods (Gradient Descent, Newton's Method, AdaGrad, Adam) on three test functions:
1. **f1 (Quadratic Bowl)**: f(x,y) = x² + y²
2. **f2 (Rosenbrock)**: f(x,y) = (1-x)² + 100(y-x²)²
3. **f3 (Cosine Bumps)**: f(x,y) = x² + y² + 10cos(x) + 10cos(y)

Each method was tested with 3 initial points and 3 learning rates (0.001, 0.01, 0.1), with a maximum of 2000 iterations and convergence tolerance of 1e-6.

---

## 🔍 Key Findings by Function

### 🟢 f1: Quadratic Bowl (Convex, Well-conditioned)

| Optimizer | Best LR | Iterations | Convergence Rate | Notes |
|-----------|---------|------------|------------------|-------|
| Gradient Descent | 0.1 | ~61 | 100% (lr≥0.01) | Linear convergence; lr=0.1 is optimal (spectral radius) |
| Newton Method | 0.1 | ~120 | 100% (lr≥0.01) | **Should converge in 1 step for pure quadratic** - damping with lr<1 slows it |
| AdaGrad | 0.1 | ~867 | 67% | Adaptive LR slows down on simple convex; accumulates squared gradients |
| Adam | 0.1 | ~209 | 100% (lr≥0.01) | Momentum helps but overkill for simple quadratic |

**💡 Critical Observation**: Newton's method with lr=1.0 should converge in exactly **1 iteration** for a pure quadratic function. The implementation uses damped learning rates (lr=0.01, 0.1) which artificially slows convergence. With lr=1.0, Newton reaches the optimum in 1 step.

---

### 🟡 f2: Rosenbrock (Non-convex, Ill-conditioned, Narrow Valley)

| Optimizer | Best LR | Iterations | Convergence Rate | Notes |
|-----------|---------|------------|------------------|-------|
| Gradient Descent | 0.001 | 2000+ (no conv) | 0% | Zigzags in narrow valley; lr≥0.01 causes divergence (NaN) |
| Newton Method | 0.1 | ~209 | 100% | Handles curvature well; converges reliably with lr≥0.01 |
| AdaGrad | All | 2000+ (no conv) | 0% | Accumulates large gradients → vanishing effective step size |
| Adam | 0.1 | ~2000 (near conv) | ~0% | Better than GD/AdaGrad but struggles with valley geometry |

**🎯 Key Phenomena Observed**:
- **GD Zigzagging**: With lr=0.001, GD makes slow progress along the valley floor but oscillates across the steep walls
- **Divergence**: lr=0.01 and 0.1 cause numerical overflow (NaN) due to exponential growth in steep directions
- **Newton's Superiority**: Only Newton consistently converges - it uses Hessian to navigate the curved valley
- **Adam's Near-miss**: With lr=0.1, Adam reaches f≈0.000007 but doesn't meet tolerance within 2000 iterations

---

### 🔴 f3: Cosine Bumps (Multimodal, Non-convex)

| Optimizer | Best LR | Iterations | Convergence Rate | Notes |
|-----------|---------|------------|------------------|-------|
| Gradient Descent | 0.1 | ~7 | 100% | Fast convergence to nearest local minimum |
| Newton Method | 0.1 | ~107 | 100% | Converges but to **different basins** depending on start point |
| AdaGrad | 0.1 | ~253 | 100% | Slow due to gradient accumulation; needs high LR |
| Adam | 0.01 | ~263 | 100% | Momentum helps escape shallow minima |

**🎯 Multiple Minima Behavior**:
- Global minimum: f ≈ -3.618 at (±2.596, ±2.596) and permutations
- Local minimum: f ≈ 8.191 at (0, ±2.596) and permutations
- **Newton from (0.5, -1.5)** converges to f=8.191 (local min) - gets trapped!
- **All other methods from all starts** find global minimum f=-3.618
- This shows Newton's basin of attraction can be problematic for multimodal functions

---

## ⚙️ Hyperparameter Sensitivity Analysis

### 📈 Learning Rate Effects

| Function | GD (lr=0.001) | GD (lr=0.01) | GD (lr=0.1) |
|----------|---------------|--------------|-------------|
| f1 | Too slow (2000 it) | Good (540 it) | Optimal (61 it) |
| f2 | Slow, no conv | **Diverges** | **Diverges** |
| f3 | Slow (929 it) | Good (111 it) | Optimal (7 it) |

**Pattern**: Higher learning rates work for well-conditioned problems (f1, f3) but cause instability for ill-conditioned problems (f2).

### 🔧 Newton's Method Damping

Newton's method with lr<1 acts as damped Newton. For f1 (quadratic):
- lr=1.0: 1 iteration (theoretical optimum)
- lr=0.1: ~120 iterations
- lr=0.01: ~1000 iterations
- lr=0.001: No convergence in 2000 iterations

For f2 (Rosenbrock): lr=0.1 converges ~8x faster than lr=0.01.

---

## 📈 Comparative Performance Summary

### ⚡ Convergence Speed (Iterations to Tolerance)

```
f1 (Quadratic):
  Fastest: GD (lr=0.1) ~60 iter
  Newton should be 1 iter (with lr=1.0)
  Adam (lr=0.1) ~210 iter
  AdaGrad (lr=0.1) ~867 iter

f2 (Rosenbrock):
  Only Newton converges reliably
  Newton (lr=0.1) ~210 iter
  Newton (lr=0.01) ~1700 iter
  Adam (lr=0.1) ~2000 iter (f=0.002, near miss)
  GD/AdaGrad: Fail

f3 (Cosine):
  GD (lr=0.1) ~7 iter (fastest!)
  Newton (lr=0.1) ~107 iter
  AdaGrad (lr=0.1) ~253 iter
  Adam (lr=0.1) ~210 iter
```

### 🛡️ Robustness Across Initial Points

| Optimizer | f1 | f2 | f3 |
|-----------|----|----|----|
| GD (lr=0.1) | ✓✓✓ | ✗✗✗ | ✓✓✓ |
| Newton (lr=0.1) | ✓✓✓ | ✓✓✓ | ✓✓✓* |
| AdaGrad (lr=0.1) | ~✓~ | ✗✗✗ | ✓✓✓ |
| Adam (lr=0.1) | ✓✓✓ | ~✗~ | ✓✓✓ |

*Newton finds different local minima depending on start point for f3.

---

## 🧠 Theoretical Explanations

### 🎯 Why Newton Excels on Rosenbrock
The Rosenbrock function has a curved valley with high curvature across the valley and low curvature along it. The Hessian captures this geometry, allowing Newton to take large steps along the valley floor while correcting for curvature. GD can only take small steps limited by the highest curvature (steep walls).

### 📉 Why AdaGrad Struggles on Rosenbrock
AdaGrad accumulates squared gradients: Gₜ = Σ gᵢ². In Rosenbrock's valley, gradients across the walls are very large, causing Gₜ to grow rapidly. The effective step size becomes η/√Gₜ → 0, stalling progress along the valley floor.

### 🤖 Why Adam Nearly Works on Rosenbrock
Adam's bias-corrected momentum (m̂) and adaptive scaling (v̂) help maintain momentum along the valley. However, the second moment estimate v̂ still grows large from cross-valley gradients, limiting step sizes. With lr=0.1 it nearly converges but needs more iterations.

### 🏃 Why GD is Fastest on Cosine Bumps
For multimodal functions with many local minima, simple GD with large LR can "jump over" shallow minima. The cosine bumps have period 2π with depth ~10, while the quadratic term grows unbounded. Large steps help escape local traps.

---

## 📊 Visualizations Generated

The following plots were created for each function and initial point:

1. **🗺️ Contour plots with optimization paths** (`{func}_initial_{x}_{y}.png`)
   - 4×3 grid showing all 4 optimizers × 3 learning rates
   - Red path shows optimization trajectory
   - Green dot = start, Red dot = end

2. **📈 Convergence curves** (`{func}_convergence_initial_{x}_{y}.png`)
   - Function value vs iteration for each optimizer/LR combination
   - Log scale for positive values, linear for negative

---

## 💡 Recommendations

1. **For well-conditioned convex problems**: Use Gradient Descent with tuned LR, or Newton with lr=1.0
2. **For ill-conditioned problems (narrow valleys)**: Newton's method is superior; use damping (lr<1) for stability
3. **For multimodal problems**: 
   - Simple GD with large LR can escape shallow minima
   - Adam provides good balance of speed and robustness
   - Avoid Newton unless you have good initialization (basin of attraction issues)
4. **AdaGrad**: Best suited for sparse gradients (e.g., NLP); not ideal for dense low-dimensional optimization

---

## ⚠️ Numerical Notes

- **Overflow warnings** in Rosenbrock: Expected for large x values during divergence
- **Newton's 1-step convergence on f1**: Requires lr=1.0; the experiment used lr≤0.1 for fair comparison
- **NaN values**: Indicate numerical divergence (gradient explosion)

