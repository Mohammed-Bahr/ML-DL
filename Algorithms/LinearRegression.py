import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ---------------- Data ----------------
training_set = pd.read_csv('weight-height.csv')

x_train = training_set['Height'].values.astype(float)   # shape (m,)
y_train = training_set['Weight'].values.astype(float)   # shape (m,)

# Standardize so gradient descent converges fast
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
X = (x_train - x_mean) / x_std
Y = (y_train - y_mean) / y_std


# ---------------- Model ----------------
def predict(x, w, b):
    return w * x + b


def CostFunction(y_hat, y):
    """Half mean squared error."""
    m = len(y)
    return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)      # FIXED


def GradientDescent(x, y, y_hat, w, b, learning_rate):
    """Single gradient descent update step."""           # FIXED
    m = len(y)
    dj_dw = (1 / m) * np.sum((y_hat - y) * x)
    dj_db = (1 / m) * np.sum(y_hat - y)

    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db
    return w, b


def train(x, y, iterations=2000, learning_rate=0.05):
    w, b = 0.0, 0.0
    cost_history = []
    for i in range(iterations):
        y_hat = predict(x, w, b)
        cost_history.append(CostFunction(y_hat, y))
        w, b = GradientDescent(x, y, y_hat, w, b, learning_rate)
        
        if i % 250 == 0:
            if len(cost_history) > 2 and cost_history[-1] >= cost_history[-2]:
                print(f"Warning: Cost increased at iteration {i} (Cost = {cost_history[-1]:.6f})")
                break
            else:   
                print(f"Iteration {i:5d} | Cost = {cost_history[-1]:.6f}")
                
    return w, b, cost_history


# ---------------- Main ----------------
def main():
    print("=" * 50, "\nUNIT TESTS\n", "=" * 50)

    # Test 1: perfect predictions -> cost = 0
    y = np.array([1.0, 2.0, 3.0])
    assert CostFunction(y, y) == 0.0
    print("Test 1 passed (cost = 0 for perfect predictions)")

    # Test 2: hand-computed cost
    y, y_hat = np.array([2.0, 4.0]), np.array([1.0, 3.0])
    assert np.isclose(CostFunction(y_hat, y), 0.5)   # (1+1)/(2*2) = 0.5
    print("Test 2 passed (hand-computed cost = 0.5)")

    # Test 3: gradient descent recovers y = 2x
    x_t, y_t = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
    w, b = 0.0, 0.0
    for _ in range(2000):
        w, b = GradientDescent(x_t, y_t, predict(x_t, w, b), w, b, 0.1)
    assert np.isclose(w, 2.0, atol=1e-2) and np.isclose(b, 0.0, atol=1e-2)
    print(f"Test 3 passed (recovered w={w:.3f}, b={b:.3f} for y=2x)")

    # Test 4: cost never increases during training
    w, b, costs = 0.0, 0.0, []
    for _ in range(100):
        y_hat = predict(x_t, w, b)
        costs.append(CostFunction(y_hat, y_t))
        w, b = GradientDescent(x_t, y_t, y_hat, w, b, 0.1)
    assert all(costs[i+1] <= costs[i] for i in range(len(costs) - 1))
    print("Test 4 passed (cost is non-increasing)")

    # ---------------- Train on real data ----------------
    print("\n" + "=" * 50, "\nTRAINING\n", "=" * 50)
    w, b, cost_history = train(X, Y, iterations=2000, learning_rate=0.05)

    # Undo normalization: y = w*(x - x_mean)/x_std * y_std + y_mean
    w_real = w * y_std / x_std
    b_real = y_mean - w_real * x_mean
    print(f"\nModel:  Weight = {w_real:.3f} * Height + {b_real:.3f}")

    # ---------------- Test predictions ----------------
    print("\n" + "=" * 50, "\nPREDICTIONS ON TEST HEIGHTS\n", "=" * 50)
    test_heights = np.array([55.0, 60.0, 65.0, 70.0, 75.0])
    test_preds = predict(test_heights, w_real, b_real)
    for h, p in zip(test_heights, test_preds):
        print(f"  Height {h:5.1f} in  ->  {p:6.1f} lbs")

    # Accuracy metrics
    preds = predict(x_train, w_real, b_real)
    mae = np.mean(np.abs(preds - y_train))
    r2 = 1 - np.sum((y_train - preds)**2) / np.sum((y_train - y_train.mean())**2)
    print(f"\nMAE = {mae:.2f} lbs | R^2 = {r2:.4f}")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Data + regression line
    axes[0].scatter(x_train, y_train, s=6, alpha=0.3, color='steelblue', label='Data')
    xs = np.linspace(x_train.min(), x_train.max(), 100)
    axes[0].plot(xs, predict(xs, w_real, b_real), 'r-', lw=2, label='Model')
    axes[0].scatter(test_heights, test_preds, color='green', marker='*',
                    s=120, zorder=5, label='Test predictions')
    axes[0].set_xlabel('Height (inches)')
    axes[0].set_ylabel('Weight (lbs)')
    axes[0].set_title('Linear Regression Fit')
    axes[0].legend()

    # 2) Cost curve
    axes[1].plot(cost_history, color='darkorange')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Cost')
    axes[1].set_title('Cost vs. Iteration')
    axes[1].grid(alpha=0.3)

    # 3) Residuals
    axes[2].scatter(x_train, y_train - preds, s=6, alpha=0.3, color='purple')
    axes[2].axhline(0, color='black', lw=1)
    axes[2].set_xlabel('Height (inches)')
    axes[2].set_ylabel('Residual (lbs)')
    axes[2].set_title('Residuals')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
