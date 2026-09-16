import numpy as np

class Adam:
    def __init__(self, params_shape, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = np.zeros(params_shape)   # first moment (momentum)
        self.v = np.zeros(params_shape)   # second moment (RMSprop)
        self.t = 0                        # timestep

    def update(self, w, grad):
        self.t += 1

        # 1. Momentum: m_t = β1*m_(t-1) + (1-β1)*grad
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # 2. RMSprop: v_t = β2*v_(t-1) + (1-β2)*grad^2
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # 3. Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 4. Final update: w_(t+1) = w_t - (lr/(sqrt(v_hat)+eps)) * m_hat
        w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return w


w = np.array([5.0])          # starting point
optimizer = Adam(params_shape=w.shape, lr=0.1)

for step in range(50):
    grad = 2 * w             # derivative of w^2
    w = optimizer.update(w, grad)

    if step % 10 == 0:
        print(f"step {step}: w = {w[0]:.4f}, loss = {w[0]**2:.4f}")
