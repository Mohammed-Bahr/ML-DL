import numpy as np
from re import split

# ==========================================
# 1. DUMMY DATA & MEAN NORMALIZATION
# ==========================================
# 5 Movies (rows), 5 Users (cols)
# Users 0-3 have ratings. User 4 is a "New User" with NO ratings (Cold Start).
Y = np.array([
    [5, 5, 0, 0, 0],  # Movie 0 (Romance)
    [5, 0, 0, 0, 0],  # Movie 1 (Romance)
    [0, 4, 0, 0, 0],  # Movie 2 (Romance)
    [0, 0, 5, 4, 0],  # Movie 3 (Action)
    [0, 0, 5, 5, 0]   # Movie 4 (Action)
])

# R is our mask: 1 if rated, 0 if missing
R = (Y != 0).astype(int)

# Mean Normalization (per-item mean)
# Note: We only average over rated entries (using R)
item_counts = R.sum(axis=1) # it sums raw ratings so it returns the number of ratings per item in vector 5*1 shape
mu = (Y * R).sum(axis=1) / np.maximum(item_counts, 1) # Avoid div by zero

# Subtract mean ONLY where rated
Ynorm = np.where(R, Y - mu[:, None], 0).split(",")

print("--- Mean Normalization ---")
print("Item Means (mu):", mu)


# ==========================================
# 2. INITIALIZE PARAMETERS
# ==========================================
num_items, num_users = Y.shape
num_features = 2  # Let's learn 2 latent features (e.g., Romance vs Action)

# Random init for X and W
np.random.seed(42) # set seed for reproducibility it helps ensure reproducible results
X = np.random.randn(num_items, num_features) * 0.1
W = np.random.randn(num_users, num_features) * 0.1

# Initialize b to 0.
# Why? For the New User (col 4), R is all 0.
# Gradient for b[4] will be 0, so it stays 0.
# This forces the prediction to rely purely on the mean (mu).
b = np.zeros(num_users)

lambda_ = 0.1  # Regularization parameter
alpha = 0.05   # Learning rate
iterations = 1000


# ==========================================
# 3. TRAINING LOOP (Gradient Descent)
# ==========================================
print("\n--- Training ---")
for i in range(iterations):
    # Forward pass: Predictions (on normalized data)
    # X @ W.T gives (num_items, num_users). b broadcasts across rows.
    pred = X @ W.T + b

    # Error matrix: Only compute error where R == 1
    E = (pred - Y) * R

    # Compute Gradients (Derived from the combined cost function)
    grad_W = E.T @ X + lambda_ * W
    grad_b = E.T.sum(axis=0) # Sum over items for each user
    grad_X = E @ W + lambda_ * X

    # Update parameters
    W -= alpha * grad_W
    b -= alpha * grad_b
    X -= alpha * grad_X

    # Print cost every 200 iterations
    if i % 200 == 0:
        cost = 0.5 * np.sum(E**2) + (lambda_ / 2) * (np.sum(W**2) + np.sum(X**2))
        print(f"Iteration {i:04d} | Cost: {cost:.4f}")


# ==========================================
# 4. PREDICTION & COLD START CHECK
# ==========================================
# Final predictions on normalized scale
final_pred_norm = X @ W.T + b

# Add the mean back to get actual star ratings
final_pred = final_pred_norm + mu[:, None]

print("\n--- Final Predictions (Stars) ---")
print("Original Ratings (0 = unrated):")
print(Y)
print("\nPredicted Ratings:")
print(np.round(final_pred, 2))

print("\n--- Cold Start Check (New User - Column 4) ---")
print("New User Predictions:", np.round(final_pred[:, 4], 2))
print("Item Means (mu):    ", np.round(mu, 2))
print("Notice how the New User's predictions closely match the item means!")


# ==========================================
# 5. FINDING RELATED ITEMS
# ==========================================
# Use the learned features X to find similar items
# We use Squared Euclidean Distance as per the notes
def get_related_items(X, item_idx, k=2):
    # Calculate squared distance from item_idx to all other items
    dist = np.sum((X - X[item_idx]) ** 2, axis=1)
    dist[item_idx] = np.inf # Exclude the item itself

    # Get indices of smallest distances
    related = np.argsort(dist)[:k]
    return related

print("\n--- Finding Related Items ---")
movie_names = ["Romance 1", "Romance 2", "Romance 3", "Action 1", "Action 2"]
for i in range(num_items):
    related = get_related_items(X, i, k=2)
    print(f"{movie_names[i]:<10} is similar to: {[movie_names[r] for r in related]}")
