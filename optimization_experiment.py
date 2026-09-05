"""
Optimization Methods Comparison Experiment

This script implements and compares four optimization methods on three test functions:
1. Gradient Descent (fixed learning rate α)
2. Newton's Method (uses Hessian inverse with damping)
3. AdaGrad (per-parameter adaptive step sizes)
4. Adam (β1=0.9, β2=0.999, ε=1e-8)

Test Functions:
1. f1: Convex bowl (quadratic) - f(x,y) = x^2 + y^2
2. f2: Rosenbrock (banana valley) - f(x,y) = (1-x)^2 + 100(y-x^2)^2
3. f3: Multimodal non-convex (cosine bumps) - f(x,y) = x^2 + y^2 + 10cos(x) + 10cos(y)

Experiment Setup (per requirements):
- 3 initial points: (-2, 2), (0.5, -1.5), (3, 3)
- 3 learning rates: 0.001, 0.01, 0.1
- Convergence: ||x_{k+1} - x_k|| < 1e-6
- Max iterations: 2000
- Newton's method uses damping (decreasing rate if f increases)

Outputs:
- Contour plots with optimization paths overlaid
- Convergence curves (f(x) vs iterations)
- Tabulated results: iterations, time, final point, final f-value, convergence rate
- Statistical summary (mean, std over multiple runs per configuration)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any
import json

@dataclass
class OptimizationResult:
    """Container for optimization run results"""
    optimizer_name: str           # Name of optimizer used
    function_name: str            # Name of function optimized
    initial_point: np.ndarray     # Starting point (x0, y0)
    learning_rate: float          # Learning rate / step size used
    final_point: np.ndarray       # Converged point (x*, y*)
    final_value: float            # Function value at final point f(x*)
    iterations: int               # Number of iterations to converge (or max_iter)
    converged: bool               # Whether converged within tolerance
    path: List[np.ndarray]        # Full optimization trajectory
    time_elapsed: float           # Wall-clock time in seconds

# ============================================================
# TEST FUNCTIONS AND THEIR DERIVATIVES
# ============================================================

def f1(x: np.ndarray) -> float:
    """Convex bowl (quadratic): f1(x, y) = x^2 + y^2
    Global minimum at (0, 0) with f=0. Well-conditioned, isotropic."""
    return x[0]**2 + x[1]**2

def grad_f1(x: np.ndarray) -> np.ndarray:
    """Gradient: ∇f = [2x, 2y]^T"""
    return np.array([2*x[0], 2*x[1]])

def hess_f1(x: np.ndarray) -> np.ndarray:
    """Hessian: H = [[2, 0], [0, 2]] (constant, positive definite)"""
    return np.array([[2, 0], [0, 2]])

def f2(x: np.ndarray) -> float:
    """Rosenbrock (banana valley): f2(x, y) = (1 - x)^2 + 100(y - x^2)^2
    Global minimum at (1, 1) with f=0. Ill-conditioned, narrow curved valley."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_f2(x: np.ndarray) -> np.ndarray:
    """Gradient: ∇f = [-2(1-x) - 400x(y-x^2), 200(y-x^2)]^T"""
    x1, x2 = x[0], x[1]
    return np.array([
        -2*(1 - x1) - 400*x1*(x2 - x1**2),
        200*(x2 - x1**2)
    ])

def hess_f2(x: np.ndarray) -> np.ndarray:
    """Hessian: H = [[2 - 400(y - 3x^2), -400x], [-400x, 200]]
    Highly variable curvature - large across valley, small along valley."""
    x1, x2 = x[0], x[1]
    return np.array([
        [2 - 400*(x2 - 3*x1**2), -400*x1],
        [-400*x1, 200]
    ])

def f3(x: np.ndarray) -> float:
    """Multimodal non-convex (cosine bumps): f3(x, y) = x^2 + y^2 + 10cos(x) + 10cos(y)
    Many local minima. Global min ≈ -3.618 at (±2.596, ±2.596) and permutations.
    Local min ≈ 8.191 at (0, ±2.596) and permutations."""
    return x[0]**2 + x[1]**2 + 10*np.cos(x[0]) + 10*np.cos(x[1])

def grad_f3(x: np.ndarray) -> np.ndarray:
    """Gradient: ∇f = [2x - 10sin(x), 2y - 10sin(y)]^T"""
    return np.array([2*x[0] - 10*np.sin(x[0]), 2*x[1] - 10*np.sin(x[1])])

def hess_f3(x: np.ndarray) -> np.ndarray:
    """Hessian: H = [[2 - 10cos(x), 0], [0, 2 - 10cos(y)]]
    Diagonal but indefinite in regions (can be negative)"""
    return np.array([
        [2 - 10*np.cos(x[0]), 0],
        [0, 2 - 10*np.cos(x[1])]
    ])

# Function registry: name -> (f, grad_f, hess_f)
FUNCTIONS = {
    'f1_quadratic': (f1, grad_f1, hess_f1),
    'f2_rosenbrock': (f2, grad_f2, hess_f2),
    'f3_cosine': (f3, grad_f3, hess_f3)
}

# ============================================================
# EXPERIMENT CONFIGURATION (per requirements)
# ============================================================
# 3 different initial points as specified
INITIAL_POINTS = [
    np.array([-2.0, 2.0]),   # Initial point 1
    np.array([0.5, -1.5]),   # Initial point 2
    np.array([3.0, 3.0])     # Initial point 3
]

# 3 learning rates as specified: 0.001, 0.01, 0.1
LEARNING_RATES = [0.001, 0.01, 0.1]
MAX_ITERATIONS = 2000       # Max iterations (limit for non-convergence)
TOLERANCE = 1e-6            # Convergence criterion: ||x_{k+1} - x_k|| < 1e-6

# ============================================================
# OPTIMIZER IMPLEMENTATIONS
# ============================================================

def gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray, 
                     lr: float, max_iter: int = MAX_ITERATIONS, 
                     tol: float = TOLERANCE) -> OptimizationResult:
    """
    Gradient Descent with fixed learning rate α.
    
    Update rule: x_{k+1} = x_k - α * ∇f(x_k)
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        lr: Fixed learning rate α (0.001, 0.01, or 0.1)
        max_iter: Maximum iterations (2000)
        tol: Convergence tolerance (1e-6)
    
    Returns:
        OptimizationResult with trajectory, iterations, final value, etc.
    """
    x = x0.copy()
    path = [x.copy()]
    start_time = time.time()
    
    for i in range(max_iter):
        g = grad_f(x)
        # Fixed step size update
        x_new = x - lr * g
        path.append(x_new.copy())
        
        # Check convergence: ||x_{k+1} - x_k|| < 1e-6
        if np.linalg.norm(x_new - x) < tol:
            return OptimizationResult(
                optimizer_name='Gradient Descent',
                function_name='',
                initial_point=x0,
                learning_rate=lr,
                final_point=x_new,
                final_value=f(x_new),
                iterations=i + 1,
                converged=True,
                path=path,
                time_elapsed=time.time() - start_time
            )
        x = x_new
    
    # Did not converge within max_iter
    return OptimizationResult(
        optimizer_name='Gradient Descent',
        function_name='',
        initial_point=x0,
        learning_rate=lr,
        final_point=x,
        final_value=f(x),
        iterations=max_iter,
        converged=False,
        path=path,
        time_elapsed=time.time() - start_time
    )

def newton_method(f: Callable, grad_f: Callable, hess_f: Callable, 
                  x0: np.ndarray, lr: float = 1.0, 
                  max_iter: int = MAX_ITERATIONS, tol: float = TOLERANCE,
                  damping: float = 0.5) -> OptimizationResult:
    """
    Newton's Method with Hessian inverse and damping.
    
    Update rule: x_{k+1} = x_k - α_k * H(x_k)^{-1} * ∇f(x_k)
    
    For pure quadratic (f1), with lr=1.0, converges in exactly 1 step.
    Damping: if f(x_new) > f(x), reduce learning rate by factor 0.5.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        hess_f: Hessian function
        x0: Initial point
        lr: Base learning rate (damped adaptively)
        max_iter: Maximum iterations (2000)
        tol: Convergence tolerance (1e-6)
        damping: Factor to reduce lr when f increases (0.5)
    
    Returns:
        OptimizationResult with trajectory, iterations, final value, etc.
    """
    x = x0.copy()
    path = [x.copy()]
    current_lr = lr
    start_time = time.time()
    
    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        
        # Compute Newton step: H^{-1} * g
        try:
            H_inv = np.linalg.inv(H)
            step = H_inv @ g
        except np.linalg.LinAlgError:
            # Fallback to gradient if Hessian singular
            step = g
        
        # Apply step with current learning rate (damping)
        # For lr=1.0 and pure quadratic, this is exact Newton step
        if current_lr == 1.0:
            x_new = x - step
        else:
            x_new = x - current_lr * step
        path.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < tol:
            return OptimizationResult(
                optimizer_name='Newton Method',
                function_name='',
                initial_point=x0,
                learning_rate=lr,
                final_point=x_new,
                final_value=f(x_new),
                iterations=i + 1,
                converged=True,
                path=path,
                time_elapsed=time.time() - start_time
            )
        
        # Damping: if function value increases, reduce step size
        if f(x_new) > f(x):
            current_lr *= damping
        x = x_new
    
    return OptimizationResult(
        optimizer_name='Newton Method',
        function_name='',
        initial_point=x0,
        learning_rate=lr,
        final_point=x,
        final_value=f(x),
        iterations=max_iter,
        converged=False,
        path=path,
        time_elapsed=time.time() - start_time
    )

def adagrad(f: Callable, grad_f: Callable, x0: np.ndarray, 
            lr: float, max_iter: int = MAX_ITERATIONS, 
            tol: float = TOLERANCE, eps: float = 1e-8) -> OptimizationResult:
    """
    AdaGrad (Adaptive Gradient) - per-parameter adaptive learning rates.
    
    Update rules:
        G_t = G_{t-1} + g_t^2          (accumulate squared gradients)
        x_{t+1} = x_t - α * g_t / (√G_t + ε)
    
    Adapts step size per parameter: frequent/large gradients get smaller steps.
    Good for sparse gradients, but can stall on dense problems (G_t grows monotonically).
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        lr: Global learning rate α (0.001, 0.01, or 0.1)
        max_iter: Maximum iterations (2000)
        tol: Convergence tolerance (1e-6)
        eps: Numerical stability constant (1e-8)
    
    Returns:
        OptimizationResult with trajectory, iterations, final value, etc.
    """
    x = x0.copy()
    path = [x.copy()]
    G = np.zeros_like(x)  # Accumulator for squared gradients
    start_time = time.time()
    
    for i in range(max_iter):
        g = grad_f(x)
        G += g**2  # Accumulate squared gradients element-wise
        # Adaptive step: divide by sqrt of accumulated squared gradients
        x_new = x - lr * g / (np.sqrt(G) + eps)
        path.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < tol:
            return OptimizationResult(
                optimizer_name='AdaGrad',
                function_name='',
                initial_point=x0,
                learning_rate=lr,
                final_point=x_new,
                final_value=f(x_new),
                iterations=i + 1,
                converged=True,
                path=path,
                time_elapsed=time.time() - start_time
            )
        x = x_new
    
    return OptimizationResult(
        optimizer_name='AdaGrad',
        function_name='',
        initial_point=x0,
        learning_rate=lr,
        final_point=x,
        final_value=f(x),
        iterations=max_iter,
        converged=False,
        path=path,
        time_elapsed=time.time() - start_time
    )

def adam(f: Callable, grad_f: Callable, x0: np.ndarray, 
         lr: float, max_iter: int = MAX_ITERATIONS, 
         tol: float = TOLERANCE, beta1: float = 0.9, 
         beta2: float = 0.999, eps: float = 1e-8) -> OptimizationResult:
    """
    Adam (Adaptive Moment Estimation) - combines momentum + adaptive learning rates.
    
    Update rules (with bias correction):
        m_t = β1 * m_{t-1} + (1-β1) * g_t          (1st moment - momentum)
        v_t = β2 * v_{t-1} + (1-β2) * g_t^2        (2nd moment - variance)
        m̂_t = m_t / (1 - β1^t)                     (bias-corrected 1st moment)
        v̂_t = v_t / (1 - β2^t)                     (bias-corrected 2nd moment)
        x_{t+1} = x_t - α * m̂_t / (√v̂_t + ε)
    
    Default hyperparameters: β1=0.9, β2=0.999, ε=1e-8 (per requirements).
    Momentum helps escape shallow minima; adaptive rates handle varying curvature.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        lr: Learning rate α (0.001, 0.01, or 0.1)
        max_iter: Maximum iterations (2000)
        tol: Convergence tolerance (1e-6)
        beta1: Exponential decay rate for 1st moment (0.9)
        beta2: Exponential decay rate for 2nd moment (0.999)
        eps: Numerical stability constant (1e-8)
    
    Returns:
        OptimizationResult with trajectory, iterations, final value, etc.
    """
    x = x0.copy()
    path = [x.copy()]
    m = np.zeros_like(x)  # 1st moment (momentum)
    v = np.zeros_like(x)  # 2nd moment (uncentered variance)
    start_time = time.time()
    
    for i in range(max_iter):
        g = grad_f(x)
        t = i + 1  # Time step (1-indexed for bias correction)
        
        # Update biased moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        
        # Bias correction (important early in training)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Adam update
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < tol:
            return OptimizationResult(
                optimizer_name='Adam',
                function_name='',
                initial_point=x0,
                learning_rate=lr,
                final_point=x_new,
                final_value=f(x_new),
                iterations=i + 1,
                converged=True,
                path=path,
                time_elapsed=time.time() - start_time
            )
        x = x_new
    
    return OptimizationResult(
        optimizer_name='Adam',
        function_name='',
        initial_point=x0,
        learning_rate=lr,
        final_point=x,
        final_value=f(x),
        iterations=max_iter,
        converged=False,
        path=path,
        time_elapsed=time.time() - start_time
    )

# Optimizer registry for experiment loop
OPTIMIZERS = {
    'Gradient Descent': gradient_descent,
    'Newton Method': newton_method,
    'AdaGrad': adagrad,
    'Adam': adam
}

def run_experiments():
    """
    Run all experiments: 3 functions × 3 initial points × 4 optimizers × 3 learning rates = 108 runs.
    
    For each configuration:
    - Run optimizer until convergence (||Δx|| < 1e-6) or max_iter (2000)
    - Record iterations, time, final point, final value, convergence status, full path
    - Print progress with convergence status (✓/✗)
    
    Returns:
        List of all OptimizationResult objects
    """
    all_results = []
    
    for func_name, (f, grad_f, hess_f) in FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"Testing on {func_name}")
        print(f"{'='*60}")
        
        for x0 in INITIAL_POINTS:
            print(f"\n  Initial point: {x0}")
            
            for opt_name, optimizer in OPTIMIZERS.items():
                for lr in LEARNING_RATES:
                    # Newton needs Hessian, others only need gradient
                    if opt_name == 'Newton Method':
                        result = optimizer(f, grad_f, hess_f, x0, lr=lr)
                    else:
                        result = optimizer(f, grad_f, x0, lr=lr)
                    
                    result.function_name = func_name
                    all_results.append(result)
                    
                    status = "✓" if result.converged else "✗"
                    print(f"    {opt_name:15s} lr={lr:.4f}: iter={result.iterations:4d} "
                          f"f={result.final_value:.6f} {status}")
    
    return all_results

def create_contour_plot(f: Callable, x_range=(-4, 4), y_range=(-4, 4), levels=50):
    """
    Create contour plot data for a 2D function.
    
    Args:
        f: Function to plot
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        levels: Number of contour levels
    
    Returns:
        X, Y meshgrid and Z function values
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate function on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    return X, Y, Z

def plot_optimization_paths(results: List[OptimizationResult]):
    """
    Generate contour plots with optimization paths overlaid.
    
    Creates one figure per function per initial point.
    Each figure is a 3×4 grid: 3 learning rates × 4 optimizers.
    Shows contour lines, optimization trajectory (red), start (green), end (red).
    
    Args:
        results: List of all OptimizationResult objects
    """
    func_groups = {}
    for r in results:
        if r.function_name not in func_groups:
            func_groups[r.function_name] = []
        func_groups[r.function_name].append(r)
    
    for func_name, func_results in func_groups.items():
        f, _, _ = FUNCTIONS[func_name]
        
        # Set plot ranges per function for good visualization
        if func_name == 'f1_quadratic':
            x_range, y_range = (-4, 4), (-4, 4)
        elif func_name == 'f2_rosenbrock':
            x_range, y_range = (-2, 2), (-1, 3)
        else:
            x_range, y_range = (-6, 6), (-6, 6)
        
        X, Y, Z = create_contour_plot(f, x_range, y_range)
        
        unique_initials = list(set(tuple(r.initial_point) for r in func_results))
        unique_lrs = sorted(set(r.learning_rate for r in func_results))
        
        for x0 in unique_initials:
            fig, axes = plt.subplots(len(unique_lrs), 4, figsize=(20, 5*len(unique_lrs)))
            if len(unique_lrs) == 1:
                axes = axes.reshape(1, -1)
            
            for lr_idx, lr in enumerate(unique_lrs):
                for opt_idx, opt_name in enumerate(['Gradient Descent', 'Newton Method', 'AdaGrad', 'Adam']):
                    ax = axes[lr_idx, opt_idx]
                    
                    # Find matching result for this config
                    matching = [r for r in func_results 
                               if np.array_equal(r.initial_point, x0) 
                               and r.learning_rate == lr 
                               and r.optimizer_name == opt_name]
                    
                    if not matching:
                        ax.set_visible(False)
                        continue
                    
                    result = matching[0]
                    path = np.array(result.path)
                    
                    # Plot contours
                    ax.contour(X, Y, Z, levels=30, alpha=0.5, cmap='viridis')
                    # Plot optimization path
                    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=3, linewidth=1)
                    # Mark start and end
                    ax.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
                    ax.plot(result.final_point[0], result.final_point[1], 'ro', markersize=10, label='End')
                    
                    ax.set_xlim(x_range)
                    ax.set_ylim(y_range)
                    ax.set_title(f'{opt_name}\nlr={lr}, iter={result.iterations}')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{func_name} - Initial point: {x0}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{func_name}_initial_{x0[0]}_{x0[1]}.png', dpi=150, bbox_inches='tight')
            plt.close()

def generate_tables(results: List[OptimizationResult]):
    """
    Print detailed tables of results for each function.
    
    Two tables per function:
    1. Full results: each run with optimizer, init point, lr, iterations, time, final point, f-value, converged
    2. Summary statistics (converged runs only): mean/std of iterations, time, f-value, convergence rate
    
    Args:
        results: List of all OptimizationResult objects
    """
    func_groups = {}
    for r in results:
        if r.function_name not in func_groups:
            func_groups[r.function_name] = []
        func_groups[r.function_name].append(r)
    
    for func_name, func_results in func_groups.items():
        print(f"\n{'='*100}")
        print(f"Results for {func_name}")
        print(f"{'='*100}")
        
        # Table 1: Full detailed results
        print(f"{'Optimizer':<15} {'Init Point':<20} {'LR':<8} {'Iter':<6} {'Time(s)':<8} "
              f"{'Final x':<15} {'Final y':<15} {'Final f':<12} {'Converged':<10}")
        print("-" * 100)
        
        for r in func_results:
            init_str = f"({r.initial_point[0]:.1f}, {r.initial_point[1]:.1f})"
            final_str_x = f"{r.final_point[0]:.6f}"
            final_str_y = f"{r.final_point[1]:.6f}"
            print(f"{r.optimizer_name:<15} {init_str:<20} {r.learning_rate:<8.4f} "
                  f"{r.iterations:<6} {r.time_elapsed:<8.4f} {final_str_x:<15} "
                  f"{final_str_y:<15} {r.final_value:<12.6f} {str(r.converged):<10}")
        
        # Table 2: Summary statistics (only converged runs)
        print("\nSummary Statistics (converged runs only):")
        print(f"{'Optimizer':<15} {'LR':<8} {'Avg Iter':<10} {'Std Iter':<10} "
              f"{'Avg Time':<10} {'Std Time':<10} {'Avg f':<12} {'Std f':<12} {'Conv Rate':<10}")
        print("-" * 100)
        
        for opt_name in ['Gradient Descent', 'Newton Method', 'AdaGrad', 'Adam']:
            for lr in LEARNING_RATES:
                # Filter converged runs for this optimizer/lr combo
                matching = [r for r in func_results 
                           if r.optimizer_name == opt_name and r.learning_rate == lr and r.converged]
                
                if matching:
                    iters = [r.iterations for r in matching]
                    times = [r.time_elapsed for r in matching]
                    fvals = [r.final_value for r in matching]
                    # Convergence rate = converged / total runs for this config
                    total_runs = len([r for r in func_results 
                                      if r.optimizer_name == opt_name and r.learning_rate == lr])
                    conv_rate = len(matching) / total_runs
                    
                    print(f"{opt_name:<15} {lr:<8.4f} {np.mean(iters):<10.1f} {np.std(iters):<10.2f} "
                          f"{np.mean(times):<10.4f} {np.std(times):<10.4f} "
                          f"{np.mean(fvals):<12.6f} {np.std(fvals):<12.6f} {conv_rate:<10.2f}")

def plot_convergence_curves(results: List[OptimizationResult]):
    """
    Generate convergence curves: f(x) vs iteration for each optimizer.
    
    Creates one figure per function per initial point.
    Each figure has 4 subplots (one per optimizer), each showing 3 curves (one per lr).
    Uses log scale for positive f-values, linear for negative (f3 has negative minima).
    
    Args:
        results: List of all OptimizationResult objects
    """
    func_groups = {}
    for r in results:
        if r.function_name not in func_groups:
            func_groups[r.function_name] = []
        func_groups[r.function_name].append(r)
    
    for func_name, func_results in func_groups.items():
        f, _, _ = FUNCTIONS[func_name]
        
        unique_initials = list(set(tuple(r.initial_point) for r in func_results))
        
        for x0 in unique_initials:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            for opt_idx, opt_name in enumerate(['Gradient Descent', 'Newton Method', 'AdaGrad', 'Adam']):
                ax = axes[opt_idx]
                
                for lr in LEARNING_RATES:
                    matching = [r for r in func_results 
                               if np.array_equal(r.initial_point, x0) 
                               and r.learning_rate == lr 
                               and r.optimizer_name == opt_name]
                    
                    if matching:
                        result = matching[0]
                        path = np.array(result.path)
                        f_vals = [f(p) for p in path]
                        ax.plot(f_vals, label=f'lr={lr}', alpha=0.7)
                
                # Use log scale for positive values, linear for negative (f3 cosine)
                f_min = min(min(f(p) for p in r.path) for r in func_results 
                           if np.array_equal(r.initial_point, x0))
                if f_min > 0:
                    ax.set_yscale('log')
                    ax.set_ylabel('f(x) (log scale)')
                else:
                    ax.set_ylabel('f(x)')
                ax.set_xlabel('Iteration')
                ax.set_title(f'{opt_name}\nStart: {x0}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{func_name} - Convergence Curves - Initial: {x0}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{func_name}_convergence_initial_{x0[0]}_{x0[1]}.png', dpi=150, bbox_inches='tight')
            plt.close()

def main():
    """
    Main entry point: run experiments, generate visualizations, print tables.
    
    Workflow:
    1. Run all 108 optimization configurations (3 functions × 3 init points × 4 optimizers × 3 lrs)
    2. Generate contour plots with paths (9 figures: 3 functions × 3 init points)
    3. Generate convergence curves (9 figures: 3 functions × 3 init points)
    4. Print detailed results tables with statistics
    """
    print("Starting optimization experiments...")
    results = run_experiments()
    
    print("\n\nGenerating visualizations...")
    plot_optimization_paths(results)
    plot_convergence_curves(results)
    
    print("\n\nGenerating tables...")
    generate_tables(results)
    
    print("\n\nDone! Check the generated PNG files for visualizations.")

if __name__ == '__main__':
    main()