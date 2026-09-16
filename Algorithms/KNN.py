import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Source: https://youtu.be/xtaom__-drE

# ---------- Euclidean Distance ----------
def euclidean_distance(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sqrt(np.sum((p - q) ** 2))


# ---------- KNN Classifier ----------
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points): # as it is just lazy learner: just memorize the training set
        self.points = points

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result


# ---------- 2D Example ----------
points = {
    'blue': [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
    'red':  [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]]
}
while True:
    # new_point = [1, 2]
    raw = input("Enter new point as 'x,y': ").strip()
    if raw.lower() == 'exit':
        break

    new_point = [float(v) for v in raw.replace(',', ' ').split()]

    clf = KNearestNeighbors(k=3)
    clf.fit(points)
    print("2D prediction:", clf.predict(new_point))   # blue

    ax = plt.subplot()
    ax.grid(True, color='#323232')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for point in points['blue']:
        ax.scatter(point[0], point[1], color='#104DCA', s=60)
    for point in points['red']:
        ax.scatter(point[0], point[1], color='#FF0000', s=60)

    new_class = clf.predict(new_point)
    color = '#FF0000' if new_class == 'red' else '#104DCA'
    ax.scatter(new_point[0], new_point[1], color=color, marker='*', s=200, zorder=100)

    for point in points['blue']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]],
                color='#104DCA', linestyle='--', linewidth=1)
    for point in points['red']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]],
                color='#FF0000', linestyle='--', linewidth=1)

    plt.show()


# ---------- 3D Example ----------
points = {
    'blue': [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 2], [2, 1, 4]],
    'red':  [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6]]
}
while True:
    # new_point = [1, 2, 4]
    raw = input("Enter new point as 'x,y,z': ").strip()
    if raw.lower() == 'exit':
        break

    new_point = [float(v) for v in raw.replace(',', ' ').split()]

    clf = KNearestNeighbors(k=3)
    clf.fit(points)
    print("3D prediction:", clf.predict(new_point))

    figure = plt.figure(figsize=(15, 12))
    ax = figure.add_subplot(projection='3d')
    ax.grid(True, color='#323232')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for point in points['blue']:
        ax.scatter(point[0], point[1], point[2], color='#104DCA', s=60)
    for point in points['red']:
        ax.scatter(point[0], point[1], point[2], color='#FF0000', s=60)

    new_class = clf.predict(new_point)
    color = '#FF0000' if new_class == 'red' else '#104DCA'
    ax.scatter(new_point[0], new_point[1], new_point[2],
            color=color, marker='*', s=200, zorder=100)

    for point in points['blue']:
        ax.plot([new_point[0], point[0]],
                [new_point[1], point[1]],
                [new_point[2], point[2]],
                color='#104DCA', linestyle='--', linewidth=1)
    for point in points['red']:
        ax.plot([new_point[0], point[0]],
                [new_point[1], point[1]],
                [new_point[2], point[2]],
                color='#FF0000', linestyle='--', linewidth=1)

    plt.show()
