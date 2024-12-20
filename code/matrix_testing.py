import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':
    # -- test matrix
    """a = torch.ones(170, 784)
    a_vector = torch.zeros(10, 170, 784)
    
    for i in range(10):
        a_vector[i] = a

    print(a_vector[5, 4, 60])

    b = torch.ones(2,10)
    print(a_vector.shape)
    print(b.shape)
    print(torch.einsum('ci,ijk->cjk',b, a_vector).shape)
    print(b @ a_vector)
    print(torch.einsum('ic,ijk->cjk',b, a_vector)[1,4, 60])"""

    min_tau = 1
    max_tau = 50
    base = max_tau / min_tau

    tau_vector = min_tau * (base ** torch.linspace(0, 1, 2))
    z_vector = 1 / tau_vector
    y_vector = 1 - z_vector
    y_vector[0] = 1

    v_vector = nn.Parameter(torch.nn.init.ones_(torch.empty(size=(1, 5), device="cpu")))
    test_matrix = nn.Parameter(torch.nn.init.ones_(torch.empty(size=(5, 2, 4), device="cpu")))

    product = torch.einsum('ci,ijk->cjk', v_vector, test_matrix).squeeze_(0)
    print(product.shape)
    print(product)

    print(tau_vector)
    print(z_vector)
    print(y_vector)

    # -- test matrix
    A = torch.randn((2, 3, 4))
    signed = torch.sign(A)
    pos = signed * A
    print(type(A))

    K_matrix = torch.nn.init.uniform_(torch.empty(size=(5, 5), device="cpu"), -0.01, 0.01).numpy()
    eigenvalues, eigenvectors = np.linalg.eig(K_matrix)
    print(np.max(np.abs(eigenvalues)))
    