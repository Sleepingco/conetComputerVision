import numpy as np
import scipy.special as sp

def self_attention(X, Wq, Wk, Wv, d_key):
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)

    QK = np.matmul(Q, np.transpose(K))  # Q·K^T
    A = sp.softmax(QK / np.sqrt(d_key), axis=1)  # scaled softmax attention weights
    C = np.matmul(A, V)  # context vectors (weighted sum of values)

    return Q, K, V, A, C

# 입력값 정의 (수정된 부분)
X = np.asarray([
    [0.0, 0.6, 0.3, 0.0],
    [0.1, 0.9, 0.0, 0.0],
    [0.0, 0.1, 0.8, 0.1],
    [0.3, 0.0, 0.6, 0.0],
    [0.0, 0.1, 0.0, 0.9]
])

Wq = np.asarray([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 3]
])
Wk = np.asarray([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 2]
])
Wv = np.asarray([
    [1, 2],
    [0, 1],
    [1, 0],
    [0, 0]
])

Q, K, V, A, C = self_attention(X, Wq, Wk, Wv, d_key=2)

# 결과 출력
np.set_printoptions(precision=3, suppress=True)
print(f"Query (Q):\n{Q}\n")
print(f"Key (K):\n{K}\n")
print(f"Value (V):\n{V}\n")
print(f"Attention Weights (A):\n{A}\n")
print(f"Context (C):\n{C}")
