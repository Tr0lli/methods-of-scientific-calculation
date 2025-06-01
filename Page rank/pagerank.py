import numpy as np

def pagerank(H, gamma, max_iter=500, tol=1e-13):
    n = H.shape[0]
    e = np.ones(n)
    v = e / n

    d = H @ e
    u = (d == 0).astype(float)
    dh = d + u * n
    dh = 1.0 / dh

    x = np.random.rand(n)
    x = x / np.sum(x)

    for it in range(1, max_iter + 1):
        y = x * dh
        y = H.T @ y + np.dot(u, y)
        y = gamma * y + (1 - gamma) * np.dot(v, y)
        err = np.max(np.abs(x - y))
        x = y
        if err < tol * np.max(x):
            break

    print("number of iterations:", it)
    return y


def pagerank_efficiente(H, gamma, max_iter=500, tol=1e-13):
    n = H.shape[0]
    e = np.ones(n)
    v = e / n

    d = (H @ e).T
    u = (d == 0).astype(float)
    dh = d + u * n
    dh = 1.0 / dh

    x = np.random.rand(n)
    x = x / np.sum(x)

    for it in range(1, max_iter + 1):
        y = x * dh
        y = y @ H + np.dot(u, y)
        y = gamma * y + (1 - gamma) * v
        err = np.max(np.abs(x - y))
        x = y
        if err < tol * np.max(x):
            break

    print("number of iterations:", it)
    return y
