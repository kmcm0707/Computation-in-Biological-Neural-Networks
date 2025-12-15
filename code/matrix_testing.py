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

    """min_tau = 2
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
    print(torch.norm(result_vector) / torch.norm(test_vector))"""

    linear = nn.Linear(4, 3, bias=False)
    x_vector = torch.randn((2, 4), requires_grad=True)
    z_vector = torch.randn((2, 4), requires_grad=True)
    y_vector = x_vector
    multiplied = x_vector * z_vector
    y_vector = multiplied
    x_vector = y_vector * z_vector
    y_vector = x_vector
    # activated_output = torch.nn.functional.softplus(output, beta=1)
    grad = torch.autograd.grad(
        outputs=y_vector,
        inputs=x_vector,
        grad_outputs=torch.ones_like(y_vector),
        retain_graph=True,
        create_graph=True,
    )
    # grad = grad[0].requires_grad_(True)
    print(grad)
    # print(z_vector)

    """matrix_1 = torch.randn((3, 4))
    matrix_2 = torch.randn((3, 4))
    matrix_3 = torch.randn((3, 4))
    matrix_4 = torch.randn((3, 4))
    matrix_5 = torch.randn((3, 4))

    normalized_matrix_1_mode_2 = torch.nn.functional.normalize(matrix_1, p=2, dim=0)

    normaliser = torch.norm(matrix_1, p=2, dim=0)
    print("Manual Normaliser:")
    print(normaliser)
    check_normalized_matrix_1_mode_2 = matrix_1 / normaliser
    print("Check Normaliser:")
    print(torch.norm(check_normalized_matrix_1_mode_2, p=2, dim=0))
    print(torch.equal(normalized_matrix_1_mode_2, check_normalized_matrix_1_mode_2))

    three_d_matrix_1 = torch.randn((5, 3, 4))
    three_d_matrix_1[:, :, :] = matrix_1
    normalized_three_d_matrix_1_mode_2 = three_d_matrix_1 / normaliser[None, :]
    print("3D Check Normaliser:")
    print(torch.norm(normalized_three_d_matrix_1_mode_2, p=2, dim=1))"""

    """print("Norms:")
    print(torch.norm(matrix_1, p=2))
    print(torch.norm(matrix_2, p=2))
    print(torch.norm(matrix_3, p=2))
    print(torch.norm(matrix_4, p=2))
    print(torch.norm(matrix_5, p=2))
    mean_matrix = (matrix_1 + matrix_2 + matrix_3 + matrix_4 + matrix_5) / 5
    print("Mean Norm:")
    print(torch.norm(mean_matrix, p=2))

    normalized_matrix_1 = matrix_1 * (torch.norm(mean_matrix, p=2) / torch.norm(matrix_1, p=2))
    normalized_matrix_2 = matrix_2 * (torch.norm(mean_matrix, p=2) / torch.norm(matrix_2, p=2))
    normalized_matrix_3 = matrix_3 * (torch.norm(mean_matrix, p=2) / torch.norm(matrix_3, p=2))
    normalized_matrix_4 = matrix_4 * (torch.norm(mean_matrix, p=2) / torch.norm(matrix_4, p=2))
    normalized_matrix_5 = matrix_5 * (torch.norm(mean_matrix, p=2) / torch.norm(matrix_5, p=2))
    print("Normalized Norms:")
    print(torch.norm(normalized_matrix_1, p=2))
    print(torch.norm(normalized_matrix_2, p=2))
    print(torch.norm(normalized_matrix_3, p=2))
    print(torch.norm(normalized_matrix_4, p=2))
    print(torch.norm(normalized_matrix_5, p=2))

    normalized_mean_matrix = (
        normalized_matrix_1 + normalized_matrix_2 + normalized_matrix_3 + normalized_matrix_4 + normalized_matrix_5
    ) / 5
    print("Normalized Mean Norm:")
    print(torch.norm(normalized_mean_matrix, p=2))

    normalized_matrix_1_mode_2 = torch.nn.functional.normalize(matrix_1, p=2, dim=0) / torch.sqrt(
        torch.tensor(matrix_1.shape[1], dtype=torch.float32)
    )
    normalized_matrix_2_mode_2 = torch.nn.functional.normalize(matrix_2, p=2, dim=0) / torch.sqrt(
        torch.tensor(matrix_2.shape[1], dtype=torch.float32)
    )
    normalized_matrix_3_mode_2 = torch.nn.functional.normalize(matrix_3, p=2, dim=0) / torch.sqrt(
        torch.tensor(matrix_3.shape[1], dtype=torch.float32)
    )
    normalized_matrix_4_mode_2 = torch.nn.functional.normalize(matrix_4, p=2, dim=0) / torch.sqrt(
        torch.tensor(matrix_4.shape[1], dtype=torch.float32)
    )
    normalized_matrix_5_mode_2 = torch.nn.functional.normalize(matrix_5, p=2, dim=0) / torch.sqrt(
        torch.tensor(matrix_5.shape[1], dtype=torch.float32)
    )
    print("Normalized Mode 2 Norms:")
    print(torch.norm(normalized_matrix_1_mode_2, p=2))
    print(torch.norm(normalized_matrix_2_mode_2, p=2))
    print(torch.norm(normalized_matrix_3_mode_2, p=2))
    print(torch.norm(normalized_matrix_4_mode_2, p=2))
    print(torch.norm(normalized_matrix_5_mode_2, p=2))

    normalized_mean_matrix_mode_2 = (
        normalized_matrix_1_mode_2
        + normalized_matrix_2_mode_2
        + normalized_matrix_3_mode_2
        + normalized_matrix_4_mode_2
        + normalized_matrix_5_mode_2
    ) / 5
    print("Normalized Mean Mode 2 Norm:")
    print(torch.norm(normalized_mean_matrix_mode_2, p=2))

    bigger_matrix_1 = torch.randn((30, 40))
    normalized_bigger_matrix_1_mode_2 = torch.nn.functional.normalize(bigger_matrix_1, p=2, dim=0) / torch.sqrt(
        torch.tensor(bigger_matrix_1.shape[1], dtype=torch.float32)
    )
    print("Bigger Matrix Mode 2 Norm:")
    print(torch.norm(normalized_bigger_matrix_1_mode_2, p=2))

    print(bigger_matrix_1.shape)

    linear = nn.Linear(4, 3, bias=False)
    print(linear.weight.shape)

    three_d_matrix = torch.randn((5, 4, 3))
    normalized_three_d_matrix_mode_2 = torch.nn.functional.normalize(three_d_matrix, p=2, dim=1) / torch.sqrt(
        torch.tensor(three_d_matrix.shape[2], dtype=torch.float32)
    )
    for i in range(5):
        print("3D Matrix Mode 2 Norm for slice ", i, ":")
        print(torch.norm(normalized_three_d_matrix_mode_2[i], p=2))"""

    """three_d_matrix_1 = torch.randn((4, 3, 2))
    for i in range(4):
        torch.nn.init.xavier_uniform_(three_d_matrix_1[i])

    v_vector = torch.tensor([0.25, 0.25, 0.25, 0.25])
    weighted_sum = torch.einsum("i,ijk->jk", v_vector, three_d_matrix_1)
    print("Weighted Sum:")
    print(weighted_sum)
    normaliser_weighted_sum = torch.norm(weighted_sum, p=2, dim=0)
    print(normaliser_weighted_sum.shape)
    print(normaliser_weighted_sum)
    normaizsed_weighted_sum = weighted_sum / normaliser_weighted_sum
    print("Weighted Sum Norm:")
    print(torch.norm(normaizsed_weighted_sum, p=2, dim=0))
    print("Weighted Sum Actual:")
    print(normaizsed_weighted_sum)
    normaizsed_three_d_matrix_1 = three_d_matrix_1 / normaliser_weighted_sum[None, :]
    for i in range(4):
        print("3D Matrix slice ", i, " Norm:")
        print(torch.norm(normaizsed_three_d_matrix_1[i], p=2, dim=0))
        print("total norm")
        print(torch.norm(normaizsed_three_d_matrix_1[i], p=2, dim=(0, 1)))
        print("actual slice")
        print(normaizsed_three_d_matrix_1[i])
        print("original slice")
        print(three_d_matrix_1[i])
        exit()"""

    """matrix = torch.nn.init.xavier_uniform_(torch.empty(size=(780, 200), device="cpu"))
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=0) * torch.sqrt(
        torch.tensor(matrix.shape[0], dtype=torch.float32)
    )
    print(matrix)

    dataset = SplitFashionMnistDataset(
        minTrainingDataPerClass=3,
        maxTrainingDataPerClass=3,
        queryDataPerClass=1,
        dimensionOfImage=28,
        class_indices=[1, 2],
        all_classes=True,
    )
    dimOut = 10

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=3 * 2)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=2, drop_last=True, num_workers=2)
    data_process = DataProcess(
        minTrainingDataPerClass=3,
        maxTrainingDataPerClass=3,
        queryDataPerClass=1,
        dimensionOfImage=28,
        device=torch.device("cpu"),
    )

    for eps, data in enumerate(metatrain_dataset):

        x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = data_process(data, 2)
        print(y_trn)
        print(y_qry)"""
