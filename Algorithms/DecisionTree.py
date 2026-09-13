# import numpy as np
# import pandas as pd
# from collections import Counter

# from sklearn.model_selection import train_test_split

# data = pd.read_csv('./DataSets/Iris.csv')

# X = data.drop(columns=['Id', 'Species']).values
# y = data['Species'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# class Node:
#     def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):
#         # Decision Node
#         self.feature_idx = feature_idx
#         self.threshold = threshold
#         self.info_gain = info_gain
#         self.left = left
#         self.right = right

#         # Leaf Node
#         self.value = value

# class DecisionTree:
#     def __init__(self, min_split = 2 , max_depth=2):
#         self.max_depth = max_depth
#         self.min_split = min_split
#         self.root = None

#     def Tree(self, DataSet, depth=0):
#         X, y = DataSet[:, :-1], DataSet[:, -1]
#         n_samples, n_features = X.shape

#         if n_samples >= self.min_split and depth <= self.max_depth:
#             best_split = self.get_best_split(DataSet, n_samples, n_features)

#             if best_split


import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split


# =========================
# Load Dataset
# =========================

data = pd.read_csv('./DataSets/Iris.csv')

# Remove Id because it is not a feature
X = data.drop(columns=['Id', 'Species']).values
y = data['Species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Node
# =========================

class Node:

    def __init__(
        self,
        feature_idx=None,
        threshold=None,
        info_gain=None,
        left=None,
        right=None,
        value=None
    ):
        # Decision Node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # Leaf Node
        self.value = value


# =========================
# Decision Tree
# =========================

class DecisionTree:

    def __init__(self, min_split=2, max_depth=2):

        self.min_split = min_split
        self.max_depth = max_depth

        self.root = None

    # =========================
    # Build Tree
    # =========================

    def Tree(self, X, y, depth=0):

        n_samples, n_features = X.shape

        # ---------------------------------
        # Stopping Conditions
        # ---------------------------------

        # 1. Not enough samples
        # 2. Maximum depth reached
        # 3. All samples belong to one class
        if (
            n_samples < self.min_split
            or depth >= self.max_depth
            or len(np.unique(y)) == 1
        ):

            leaf_value = self.calculate_leaf_value(y)

            return Node(value=leaf_value)

        # ---------------------------------
        # Find Best Split
        # ---------------------------------

        best_split = self.get_best_split(
            X,
            y,
            n_samples,
            n_features
        )

        # ---------------------------------
        # No useful split
        # ---------------------------------

        if best_split["info_gain"] <= 0:

            leaf_value = self.calculate_leaf_value(y)

            return Node(value=leaf_value)

        # ---------------------------------
        # Recursively build left subtree
        # ---------------------------------

        left_subtree = self.Tree(
            best_split["X_left"],
            best_split["y_left"],
            depth + 1
        )

        # ---------------------------------
        # Recursively build right subtree
        # ---------------------------------

        right_subtree = self.Tree(
            best_split["X_right"],
            best_split["y_right"],
            depth + 1
        )

        # ---------------------------------
        # Return Decision Node
        # ---------------------------------

        return Node(
            feature_idx=best_split["feature_idx"],
            threshold=best_split["threshold"],
            info_gain=best_split["info_gain"],
            left=left_subtree,
            right=right_subtree
        )

    # =========================
    # Find Best Split
    # =========================

    def get_best_split(self, X, y, n_samples, n_features):

        best_split = {
            "feature_idx": None,
            "threshold": None,
            "X_left": None,
            "X_right": None,
            "y_left": None,
            "y_right": None,
            "info_gain": -float("inf")
        }

        # Try every feature
        for feature_idx in range(n_features):

            feature_values = X[:, feature_idx]

            # Possible thresholds
            thresholds = np.unique(feature_values)

            # Try every threshold
            for threshold in thresholds:

                X_left, X_right, y_left, y_right = self.split(
                    X,
                    y,
                    feature_idx,
                    threshold
                )

                # Ignore empty splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate information gain
                info_gain = self.information_gain(
                    y,
                    y_left,
                    y_right
                )

                # Keep best split
                if info_gain > best_split["info_gain"]:

                    best_split["feature_idx"] = feature_idx
                    best_split["threshold"] = threshold

                    best_split["X_left"] = X_left
                    best_split["X_right"] = X_right

                    best_split["y_left"] = y_left
                    best_split["y_right"] = y_right

                    best_split["info_gain"] = info_gain

        return best_split

    # =========================
    # Split Dataset
    # =========================

    def split(self, X, y, feature_idx, threshold):

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        X_left = X[left_mask]
        X_right = X[right_mask]

        y_left = y[left_mask]
        y_right = y[right_mask]

        return X_left, X_right, y_left, y_right

    # =========================
    # Gini Impurity
    # =========================

    def gini(self, y):

        classes = np.unique(y)

        gini = 1.0

        for cls in classes:

            probability = np.sum(y == cls) / len(y)

            gini -= probability ** 2

        return gini

    # =========================
    # Information Gain
    # =========================

    def information_gain(self, parent, left, right):

        parent_gini = self.gini(parent)

        left_gini = self.gini(left)
        right_gini = self.gini(right)

        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        weighted_gini = (
            left_weight * left_gini
            + right_weight * right_gini
        )

        return parent_gini - weighted_gini

    # =========================
    # Leaf Value
    # =========================

    def calculate_leaf_value(self, y):

        counter = Counter(y)

        return counter.most_common(1)[0][0]

    # =========================
    # Fit
    # =========================

    def fit(self, X, y):

        self.root = self.Tree(
            X,
            y,
            depth=0
        )

    # =========================
    # Predict
    # =========================

    def predict(self, X):

        predictions = []

        for row in X:

            prediction = self.make_prediction(
                row,
                self.root
            )

            predictions.append(prediction)

        return np.array(predictions)

    # =========================
    # Make Prediction
    # =========================

    def make_prediction(self, row, node):

        # Leaf
        if node.value is not None:

            return node.value

        # Go left
        if row[node.feature_idx] <= node.threshold:

            return self.make_prediction(
                row,
                node.left
            )

        # Go right
        return self.make_prediction(
            row,
            node.right
        )


# =========================
# Train
# =========================

tree = DecisionTree(
    min_split=2,
    max_depth=2
)

tree.fit(X_train, y_train)


# =========================
# Predict
# =========================

y_pred = tree.predict(X_test)


# =========================
# Accuracy
# =========================

accuracy = np.mean(y_pred == y_test)

print("Predictions:")
print(y_pred)

print("\nActual:")
print(y_test)

print("\nAccuracy:", accuracy)
