import torch
import torch.nn as nn

if __name__ == "__main__":
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

    """min_tau = 2
    max_tau = 50
    base = max_tau / min_tau

    tau_vector = min_tau * (base ** torch.linspace(0, 1, 2))
    z_vector = 1 / tau_vector
    y_vector = 1 - z_vector
    y_vector[0] = 1

    v_vector = nn.Parameter(torch.nn.init.ones_(torch.empty(size=(1, 5), device="cpu")))
    test_matrix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(size=(2, 2, 2), device="cpu")))
    test_matrix_2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(size=(2, 2, 2), device="cpu")))

    print(test_matrix)
    print(test_matrix_2)
    product = torch.einsum("ijk,ijk->jk", test_matrix_2, test_matrix)
    print(product.shape)
    print(product)

    print(test_matrix)
    product = torch.nn.functional.softmax(test_matrix, dim=0)
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

    # -- test matrix
    A = torch.randn((3, 4))
    print(A[:, 1])
    norm_A = torch.nn.functional.normalize(A, p=2, dim=0)
    print(norm_A)
    print(A)"""

    min_tau = 10
    max_tau = 100
    base = max_tau / min_tau

    tau_vector = min_tau * (base ** torch.linspace(0, 1, 3))
    z_vector = 1 / tau_vector
    y_vector = 1 - z_vector

    """print(tau_vector)
    print(z_vector)
    print(y_vector)"""

    test_matrix_2 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(size=(3, 2, 2), device="cpu"))) * 0.1
    test_matrix = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(size=(2, 2), device="cpu")))
    test_matrix_norm = torch.norm(test_matrix, p=2)
    test_matrix_2_norm = torch.norm(test_matrix_2, p=2, dim=(1, 2))
    divesor = test_matrix_norm / test_matrix_2_norm
    print(test_matrix_norm)
    print(test_matrix_2_norm)
    print(divesor)
    fixed_matrix = test_matrix_2 * divesor[:, None, None]
    print(torch.norm(fixed_matrix, p=2, dim=(1, 2)))

    trainingDataPerClass = [
        0,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
    ]
    print(len(trainingDataPerClass))
