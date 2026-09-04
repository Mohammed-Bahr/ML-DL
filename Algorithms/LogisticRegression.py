# using sklearn's Logistic Regression model to predict binary outcomes based on input features

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# x_train = [[0, 1], [1, 3] , [12,2] , [4,3]]
# y_train = [0, 0, 1, 1]
# model.fit(x_train, y_train)
# x_test = [[1, 2], [3, 4], [12, 5]]
# predictions = model.predict(x_test)
# print(predictions)


# From scratch implementation of logistic regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('weight-height.csv')

x_train = data[['Height', 'Weight']].values
y_train = data['Gender'].values

m, n = x_train.shape

print("Number of training examples: ", m)
print("Number of training features: ", n)

# Scatter plot: Height vs Weight, colored by Gender
plt.figure(figsize=(8, 6))

male_mask = y_train == 'Male'
female_mask = y_train == 'Female'

plt.scatter(x_train[male_mask, 0], x_train[male_mask, 1],
            c='blue', label='Male', alpha=0.5, s=10)
plt.scatter(x_train[female_mask, 0], x_train[female_mask, 1],
            c='red', label='Female', alpha=0.5, s=10)

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight by Gender')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# -----------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def CostFunction(x, y, w, b):
    m = len(y)
    y_hat = sigmoid(np.dot(x, w) + b)
    cost = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return cost


def GradientDescent(x, y, w, b, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    for i in range(num_iterations):
        y_hat = sigmoid(np.dot(x, w) + b)
        dw = (1 / m) * np.dot(x.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = CostFunction(x, y, w, b)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    return w, b, cost_history

def predict(x, w, b):
    preds = np.zeros(m)

    z = np.dot(w, x.T) + b
    g = sigmoid(z)
    for i in range(m):
        preds[i] = 1 if g[i] >= 0.5 else 0

    return preds


def plot_decision_boundary(x, y, w, b, x_mean, x_std):
    plt.figure(figsize=(8, 6))

    male_mask = y == 1
    female_mask = y == 0

    plt.scatter(x[male_mask, 0], x[male_mask, 1],
                c='blue', label='Male', alpha=0.5, s=10)
    plt.scatter(x[female_mask, 0], x[female_mask, 1],
                c='red', label='Female', alpha=0.5, s=10)

    # Decision boundary: w0*((h-mu0)/s0) + w1*((wt-mu1)/s1) + b = 0
    # Solve for Weight given Height
    h_vals = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    h_scaled = (h_vals - x_mean[0]) / x_std[0]
    wt_scaled = -(w[0] * h_scaled + b) / w[1]
    wt_vals = wt_scaled * x_std[1] + x_mean[1]

    plt.plot(h_vals, wt_vals, 'k-', linewidth=2, label='Decision boundary')

    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    learning_rate = 0.1
    num_iterations = 1000

    y_train = (data['Gender'] == 'Male').astype(int).values

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train_scaled = (x_train - x_mean) / x_std

    w = np.zeros(n)
    b = 0

    cost = CostFunction(x_train_scaled, y_train, w, b)
    print("Initial cost: ", cost)

    w, b, cost_history = GradientDescent(x_train_scaled, y_train, w, b, learning_rate, num_iterations)

    plt.figure(figsize=(8, 5))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Final weights: ", w)
    print("Final bias: ", b)

    print("Final cost: ", CostFunction(x_train_scaled, y_train, w, b))

    preds = predict(x_train_scaled, w, b)
    accuracy = np.mean(preds == y_train) * 100
    print(f"Training accuracy: {accuracy:.2f}%")


    plot_decision_boundary(x_train, y_train, w, b, x_mean, x_std)

if __name__ == "__main__":
    main()
