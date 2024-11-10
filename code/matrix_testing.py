import torch

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
    max_tau = 30
    base = max_tau / min_tau

    tau_vector = min_tau * (base ** torch.linspace(0, 1, 5))
    z_vector = 1 / tau_vector
    y_vector = 1 - z_vector

    print(tau_vector)
    print(z_vector)
    print(y_vector)
    print(torch.sqrt(10))

    #print((a_vector * b[:, None, None]).shape)