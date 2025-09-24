import matplotlib.pyplot as plt
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
    """divesor = test_matrix_norm / test_matrix_2_norm
    print(test_matrix_norm)
    print(test_matrix_2_norm)
    print(divesor)
    fixed_matrix = test_matrix_2 * divesor[:, None, None]
    print(torch.norm(fixed_matrix, p=2, dim=(1, 2)))"""

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
    """print(len(trainingDataPerClass))

    with torch.no_grad():
        LSTM_cell = nn.LSTMCell(input_size=2, hidden_size=3, bias=True)
        LSTM_cell.bias_ih[0] = 0

        LSTM_cell.weight_ih[:3, :] = 0"""

    """print(LSTM_cell.weight_ih)

    matrix = torch.randn((2, 3, 4))
    print(matrix)
    matrix = torch.reshape(matrix, (4, 6))
    print(matrix)
    matrix = torch.reshape(matrix, (2, 3, 4))
    print(matrix)"""

    min_tau = 2
    max_tau = 60
    base = max_tau / min_tau
    tau_vector = 2 * (base ** torch.linspace(0, 1, 256))
    z_vector = 1 / tau_vector
    y_vector = 1 - z_vector
    print(y_vector)
    print(z_vector)

    test_matrix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(size=(256, 256), device="cpu")))
    # nn.init.xavier_uniform_(test_matrix)
    diag_z = torch.diag(z_vector)
    test_matrix_z = diag_z @ test_matrix

    diag_y = torch.diag(y_vector)
    K_y = diag_y + test_matrix_z - torch.eye(test_matrix.shape[0])
    all_eigenvalues = torch.linalg.eigvals(K_y)
    print(torch.max(torch.real(all_eigenvalues)))

    all_eigenvalues_real = torch.real(all_eigenvalues)
    all_eigenvalues_imag = torch.imag(all_eigenvalues)

    plt.scatter(all_eigenvalues_real.detach().numpy(), all_eigenvalues_imag.detach().numpy())
    plt.show()

    test_vector = torch.randn((1, 256), device="cpu")
    result_vector = y_vector * test_vector + z_vector * (test_vector @ test_matrix)
    print(torch.norm(result_vector) / torch.norm(test_vector))
