"""
Content-Based Filtering (Two-Tower Neural Network) -- from scratch, NumPy only.

Pipeline (mirrors the notes):
    x_u --user tower--> v_u --L2 norm--\
                                        >-- dot --> y_hat   (trained with ONE MSE loss)
    x_m --item tower--> v_m --L2 norm--/

    + retrieval (cheap, broad) -> ranking (precomputed v_m, one v_u, dot products)
    + similar items via squared distance between v_m vectors
"""
import numpy as np


# ----------------------------------------------------------------------------
# 1. Building blocks: Dense layer, Tower (MLP), L2-normalize, Adam
# ----------------------------------------------------------------------------
class Dense:
    """y = activation(x @ W + b).  activation in {'relu', None}."""

    def __init__(self, n_in, n_out, activation=None, rng=None):
        rng = rng or np.random.default_rng()
        # He init for ReLU, Glorot-ish for linear
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation
        # Adam state
        self.m = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self.v = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        if self.activation == "relu":
            self.mask = z > 0
            return z * self.mask
        return z

    def backward(self, grad_out):
        if self.activation == "relu":
            grad_out = grad_out * self.mask
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def step(self, t, lr, l2, beta1=0.9, beta2=0.999, eps=1e-8):
        for name, param, grad in (("W", self.W, self.dW + l2 * self.W),  # weight decay on W only
                                  ("b", self.b, self.db)):
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            m_hat = self.m[name] / (1 - beta1 ** t)
            v_hat = self.v[name] / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)   # in-place update


class Tower:
    """Sequential MLP: hidden ReLU layers, final LINEAR layer of size `out_dim`."""

    def __init__(self, n_in, hidden=(256, 128), out_dim=32, rng=None):
        sizes = [n_in, *hidden]
        self.layers = [Dense(a, b, "relu", rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.layers.append(Dense(sizes[-1], out_dim, None, rng))  # no activation on v

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, t, lr, l2):
        for layer in self.layers:
            layer.step(t, lr, l2)


def l2_normalize(v, eps=1e-12):
    """v -> v / ||v||.  Returns (normalized, norms) so backward can reuse the norms."""
    norm = np.sqrt((v ** 2).sum(axis=1, keepdims=True)) + eps
    return v / norm, norm


def l2_normalize_backward(grad_n, n, norm):
    """Backprop through n = v / ||v||:  dL/dv = (g - n (n.g)) / ||v||."""
    return (grad_n - n * (n * grad_n).sum(axis=1, keepdims=True)) / norm


# ----------------------------------------------------------------------------
# 2. The model
# ----------------------------------------------------------------------------
class TwoTowerContentBased:
    def __init__(self, user_dim, item_dim, embed_dim=32,
                 user_hidden=(256, 128), item_hidden=(256, 128), seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.user_net = Tower(user_dim, user_hidden, embed_dim, rng)
        self.item_net = Tower(item_dim, item_hidden, embed_dim, rng)
        self.t = 0  # Adam step counter

    # ---- prediction: y_hat = v_u . v_m  (cosine similarity in [-1, 1]) ----
    def _forward(self, Xu, Xm):
        vu = self.user_net.forward(Xu)
        vm = self.item_net.forward(Xm)
        un, u_norm = l2_normalize(vu)
        mn, m_norm = l2_normalize(vm)
        pred = (un * mn).sum(axis=1)                       # Dot(axes=1)
        self._cache = (un, u_norm, mn, m_norm)
        return pred

    def _backward(self, dpred):
        un, u_norm, mn, m_norm = self._cache
        d_un = dpred[:, None] * mn                         # d(un.mn)/d un = mn
        d_mn = dpred[:, None] * un
        # gradients flow through the dot product into BOTH towers
        self.user_net.backward(l2_normalize_backward(d_un, un, u_norm))
        self.item_net.backward(l2_normalize_backward(d_mn, mn, m_norm))

    # ---- training: ONE cost J = mean (v_u.v_m - y)^2 + regularization ----
    def fit(self, Xu, Xm, user_idx, item_idx, y, epochs=30, batch_size=256,
            lr=1e-3, l2=1e-4, val=None, verbose=True):
        """
        Xu: (n_users, user_dim)   Xm: (n_items, item_dim)
        (user_idx[k], item_idx[k], y[k]) are the OBSERVED ratings, i.e. r(i,j)=1.
        """
        # Standardize features; scale ratings to [-1, 1] to match cosine range.
        self.u_mu, self.u_sd = Xu.mean(0), Xu.std(0) + 1e-8
        self.m_mu, self.m_sd = Xm.mean(0), Xm.std(0) + 1e-8
        self.y_min, self.y_max = y.min(), y.max()
        Xu_s, Xm_s = self._scale_u(Xu), self._scale_m(Xm)
        y_s = self._scale_y(y)

        n = len(y)
        history = []
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                pred = self._forward(Xu_s[user_idx[b]], Xm_s[item_idx[b]])
                err = pred - y_s[b]
                total += (err ** 2).sum()
                self._backward(2 * err / len(b))           # dMSE/dpred
                self.t += 1
                self.user_net.step(self.t, lr, l2)
                self.item_net.step(self.t, lr, l2)
            train_mse = total / n
            rec = {"epoch": epoch, "train_mse": train_mse}
            if val is not None:
                vu, vi, vy = val
                rec["val_rmse"] = self.rmse(Xu, Xm, vu, vi, vy)
            history.append(rec)
            if verbose and (epoch % 5 == 0 or epoch == 1):
                extra = f" | val RMSE (orig. scale): {rec['val_rmse']:.3f}" if val else ""
                print(f"epoch {epoch:3d} | train MSE (scaled): {train_mse:.4f}{extra}")
        return history

    # ---- scaling helpers ----
    def _scale_u(self, X): return (X - self.u_mu) / self.u_sd
    def _scale_m(self, X): return (X - self.m_mu) / self.m_sd
    def _scale_y(self, y): return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
    def _unscale_y(self, y): return (y + 1) / 2 * (self.y_max - self.y_min) + self.y_min

    # ---- inference ----
    def embed_users(self, Xu):
        return l2_normalize(self.user_net.forward(self._scale_u(Xu)))[0]

    def embed_items(self, Xm):
        return l2_normalize(self.item_net.forward(self._scale_m(Xm)))[0]

    def predict(self, Xu, Xm, user_idx, item_idx):
        Vu, Vm = self.embed_users(Xu), self.embed_items(Xm)
        cos = (Vu[user_idx] * Vm[item_idx]).sum(axis=1)
        return self._unscale_y(cos)

    def rmse(self, Xu, Xm, user_idx, item_idx, y):
        return np.sqrt(np.mean((self.predict(Xu, Xm, user_idx, item_idx) - y) ** 2))


# ----------------------------------------------------------------------------
# 3. Similar items + Retrieval -> Ranking
# ----------------------------------------------------------------------------
def precompute_similar_items(Vm, k=10):
    """Overnight job: for every item, the k items minimizing ||v_m^(k) - v_m^(i)||^2."""
    sq = (Vm ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * Vm @ Vm.T          # pairwise squared distances
    np.fill_diagonal(d2, np.inf)                             # exclude the item itself
    return np.argsort(d2, axis=1)[:, :k]                     # lookup table (n_items, k)


def retrieve(user_history, similar_table, item_popularity, n_popular=50, n_seen_sim=10):
    """
    Cheap, broad candidate generation (no neural net at request time):
      - items similar to the last few items the user consumed (lookup table)
      - globally popular items (stand-in for 'top genres / top in country')
    Then merge, dedupe, and drop items the user already consumed.
    """
    candidates = []
    for item in user_history[-n_seen_sim:]:
        candidates.extend(similar_table[item].tolist())
    candidates.extend(np.argsort(-item_popularity)[:n_popular].tolist())
    seen = set(user_history)
    return np.array(sorted({c for c in candidates if c not in seen}))


def recommend(model, x_user, Vm_precomputed, candidates, top_n=10):
    """
    Ranking: ONE forward pass of the user tower, then cheap dot products
    with the PRECOMPUTED v_m of each candidate. Sort and return top_n.
    """
    vu = model.embed_users(x_user[None, :])[0]               # single inference call
    scores = Vm_precomputed[candidates] @ vu                 # ~32 mult-adds per candidate
    order = np.argsort(-scores)[:top_n]
    return candidates[order], model._unscale_y(scores[order])


# ----------------------------------------------------------------------------
# 4. Demo on synthetic data
# ----------------------------------------------------------------------------
def make_synthetic_data(n_users=300, n_items=500, latent=6, user_dim=15, item_dim=10,
                        density=0.15, seed=1):
    """Hidden taste vectors generate BOTH features and ratings, so features are informative."""
    rng = np.random.default_rng(seed)
    Tu = rng.normal(size=(n_users, latent))
    Tm = rng.normal(size=(n_items, latent))
    Xu = Tu @ rng.normal(size=(latent, user_dim)) + 0.3 * rng.normal(size=(n_users, user_dim))
    Xm = Tm @ rng.normal(size=(latent, item_dim)) + 0.3 * rng.normal(size=(n_items, item_dim))

    cos = (Tu / np.linalg.norm(Tu, axis=1, keepdims=True)) @ \
          (Tm / np.linalg.norm(Tm, axis=1, keepdims=True)).T
    ratings = np.clip(np.round(3 + 2 * cos + 0.3 * rng.normal(size=cos.shape)), 1, 5)

    mask = rng.random((n_users, n_items)) < density          # r(i,j) = 1 where True
    u_idx, i_idx = np.nonzero(mask)
    return Xu, Xm, u_idx, i_idx, ratings[u_idx, i_idx]


if __name__ == "__main__":
    Xu, Xm, u_idx, i_idx, y = make_synthetic_data()
    n = len(y)
    perm = np.random.default_rng(0).permutation(n)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]

    model = TwoTowerContentBased(user_dim=Xu.shape[1], item_dim=Xm.shape[1])
    model.fit(Xu, Xm, u_idx[tr], i_idx[tr], y[tr], epochs=30, lr=1e-3,
              val=(u_idx[te], i_idx[te], y[te]))

    baseline = np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2))
    print(f"\nTest RMSE: {model.rmse(Xu, Xm, u_idx[te], i_idx[te], y[te]):.3f} "
          f"(predict-the-mean baseline: {baseline:.3f})")

    # Serving: precompute everything that doesn't depend on the request
    Vm = model.embed_items(Xm)
    similar = precompute_similar_items(Vm, k=10)
    popularity = np.bincount(i_idx[tr], minlength=len(Xm))

    user = 0
    history = i_idx[tr][u_idx[tr] == user].tolist()
    cands = retrieve(history, similar, popularity)
    items, scores = recommend(model, Xu[user], Vm, cands, top_n=5)
    print(f"\nUser {user}: retrieved {len(cands)} candidates -> top 5 ranked")
    for it, sc in zip(items, scores):
        print(f"  item {it:4d}  predicted rating {sc:.2f}")
    print(f"\nItems most similar to item 0: {similar[0][:5].tolist()}")
