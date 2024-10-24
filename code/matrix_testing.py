import torch

if __name__ == '__main__':
    # -- test matrix
    a = torch.ones(170, 784)
    a_vector = torch.zeros(10, 170, 784)
    
    for i in range(10):
        a_vector[i] = a

    print(a_vector[5, 4, 60])

    b = torch.ones(2,10)
    print(a_vector.shape)
    print(b.shape)
    print(torch.einsum('ci,ijk->cjk',b, a_vector).shape)
    print(torch.einsum('ic,ijk->cjk',b, a_vector)[1,4, 60])

    #print((a_vector * b[:, None, None]).shape)