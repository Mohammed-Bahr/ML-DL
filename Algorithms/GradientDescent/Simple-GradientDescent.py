import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + 0.5 * x

def df(x):
    return np.cos(x) + 0.10


# Focus on the region where the minima are
x_vals = np.arange(-20, 20, 0.05)
y_vals = f(x_vals)

# Start near the local minimum (x ≈ 0) so it converges safely
current_pos = (0.0, f(0.0))
learning_rate = 0.01

# Turn on interactive mode
plt.ion()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the function
line, = ax.plot(x_vals, y_vals, 'b-', label='f(x) = sin(x) + 0.5x')
# Initialize the point
point, = ax.plot([current_pos[0]], [current_pos[1]], 'ro', markersize=10, label='Current point')
ax.set_xlim(-20, 20)
ax.set_ylim(-10, 10)
ax.set_title('Gradient Descent - Single Updating Window')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True)

# Store trajectory for optional trail
traj_x = []
traj_y = []

for i in range(1000):
    grad = df(current_pos[0])
    new_x = current_pos[0] - learning_rate * grad
    new_y = f(new_x)
    current_pos = (new_x, new_y)
    
    traj_x.append(current_pos[0])
    traj_y.append(current_pos[1])
    
    # Update only the point
    point.set_data([current_pos[0]], [current_pos[1]])
    
    # Optional: redraw the trail of last 20 points
    # We'll skip for simplicity to avoid extra artists; just the dot moves.
    
    # Pause to let the GUI update
    plt.pause(0.01)

# Turn off interactive mode and keep the window open until closed
plt.ioff()
plt.show()
print("Finished. Close the window to exit.")