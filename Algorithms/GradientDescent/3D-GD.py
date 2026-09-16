import numpy as np
import matplotlib.pyplot as plt

def f(x , y):
    return np.sin(5 * x) * np.cos(5 * y) / 5

def df(x, y):
    df_dx = np.cos(5 * x) * np.cos(5 * y)
    df_dy = -np.sin(5 * x) * np.sin(5 * y)
    return np.array([df_dx, df_dy])


x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)


X , Y = np.meshgrid(x, y)
Z = f(X, Y)

# -------------------------------------------

current_pos = (0.7 , 0.4 , f(0.7 , 0.4))
learning_rate = 0.01
ax = plt.subplot(projection='3d' , computed_zorder=False)

for _ in range(1000):
    grad = df(current_pos[0], current_pos[1])
    new_x = current_pos[0] - learning_rate * grad[0]
    new_y = current_pos[1] - learning_rate * grad[1]
    new_z = f(new_x, new_y)
    current_pos = (new_x, new_y, new_z)
    
# -------------------------------------------

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7 , zorder=0)
    ax.set_title('3D Gradient Descent')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='r', s=50, zorder=1)
    plt.pause(0.01)
    ax.clear()
