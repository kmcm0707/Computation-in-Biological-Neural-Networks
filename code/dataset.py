import os
import emnist
from emnist import list_datasets
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np

class EmnistDataset(Dataset):
    """
        EMNIST Dataset class.

    Constructs training and query sets for meta-training. Note that rather
    than a single image and the corresponding label, each data point
    represents samples from a class of images, containing training and query
    data from that category.

    Uses: EMNIST Balanced: 131,600 characters. 47 balanced classes. - Only test dataset is used for metalearning.

    Class Attributes:
        - trainingDataPerClass: (int) integer value representing the number of training data per class,
        - queryDataPerClass: (int) integer value representing the number of query data per class,
        - dimensionImages: (int) integer value representing the dimension size of the images.
        - transform: (torchvision.transforms.Compose) a composition of transformations to be applied to the images.
        - n_class: (int) integer value representing the number of classes in the dataset.
        - emnist_test_data: (emnist.EMNIST) the EMNIST test dataset.
    """
    def __init__(self, trainingDataPerClass, queryDataPerClass, dimensionImages):
        """
            Initialize the EmnistDataset class.

        The method first downloads and preprocesses the EMNIST dataset, creating
        directories and files necessary for later use. It then sets the values for the number of training data, the number of queries,
        and the dimensions of the images to be loaded, respectively. It also defines the transformation to be applied to the images.

        :param trainingDataPerClass: (int) integer value representing the number of training data per class,
        :param queryDataPerClassQ: (int) integer value representing the number of query data per class,
        :param dimensionImages: (int) integer value representing the dimension size of the images.
        """

        self.trainingDataPerClass = trainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.dimensionImages = dimensionImages
        self.n_class = 47

        print(emnist.get_cached_data_path()) # Printing the path to the cached data
        emnist.ensure_cached_data() # Ensuring the EMNIST dataset is cached

        images, labels = emnist.extract_test_samples("balanced"); # Extracting the EMNIST test dataset
        emnist_test_data = [[images[i] for i in range(len(labels)) if labels[i] == j] for j in range(47)] # Extracting the EMNIST test dataset
        self.emnist_test_data = np.array(emnist_test_data)
        self.emnist_test_data = self.emnist_test_data.reshape(self.emnist_test_data.shape[1], self.emnist_test_data.shape[0], 28, 28)

        self.transform = transforms.Compose([transforms.Resize((dimensionImages, dimensionImages)), transforms.ToTensor()])

    def __len__(self):
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of classes in the
            dataset
        """
        return self.n_class

    def __getitem__(self, index):
        """
            Return a tuple of tensors containing training and query images and
            corresponding labels for a given index.

        The images are loaded from the character folder at the given index. Each
        image is converted to grayscale and resized to the specified dimension
        using `torchvision.transforms.Resize` and `torchvision.transforms.ToTensor`.
        The returned tuples contain tensors of K and Q images, where K is the training
        data size per class and Q is the query data size per class. Both K and Q are
        specified at initialization. The indices corresponding to the images are
        also returned in tensors of size K and Q, respectively.

        :param index: (int) Index of the character folder from which images are to be retrieved.
        :return: tuple: A tuple (img_K, idx_vec_K, img_Q, idx_vec_Q) containing the following tensors:
            - img_K (torch.Tensor): A tensor of K images from class index
            - idx_vec_K (torch.Tensor): A tensor of K indices corresponding to class index
            - img_Q (torch.Tensor): A tensor of Q images from the class index
            - idx_vec_Q (torch.Tensor): A tensor of Q indices corresponding to class index.
        """
        images = []
        for image in self.emnist_test_data[:, index]:
            images.append(self.transform(Image.fromarray(image).convert('L')))

        images = torch.cat(images)
        idx_vec = index * torch.ones_like(torch.empty(400), dtype=int)

        return images[:self.trainingDataPerClass], idx_vec[:self.trainingDataPerClass], \
               images[self.trainingDataPerClass:self.trainingDataPerClass + self.queryDataPerClass], idx_vec[self.trainingDataPerClass:self.trainingDataPerClass + self.queryDataPerClass]


class DataProcess:
    """
        Meta-training data processor class.

    The function is designed to process meta-training data, specifically
    training and query data sets. The function performs several operations,
    including:
    1) Flattening images and merging image category and image index dimensions,
    2) Transferring the processed data to the specified processing device,
        which could either be 'cpu' or 'cuda',
    3) Shuffling the order of data points in the training set to avoid any
        potential biases during model training.
    """
    def __init__(self, K, Q, dim, device='cpu', iid=True):
        """
            Initialize the DataProcess object.

        :param K: (int) training data set size per class,
        :param Q: (int) query data set size per class,
        :param dim: (int) image dimension,
        :param device: (str) The processing device to use. Default is 'cpu',
        :param iid: (bool) shuffling flag. Default is True.
        """
        self.K = K
        self.Q = Q
        self.device = device
        self.iid = iid
        self.dim = dim

    def __call__(self, data, M):
        """
            Processing meta-training data.

        :param data: (tuple) A tuple of training and query data and the
            corresponding indices.
        :param M: (int) The number of classes.
        :return: tuple: A tuple of processed training and query data and
            the corresponding indices.
        """

        # -- load data
        x_trn, y_trn, x_qry, y_qry = data

        # -- reshape
        x_trn = torch.reshape(x_trn, (M * self.K, self.dim ** 2)).to(self.device)
        y_trn = torch.reshape(y_trn, (M * self.K, 1)).to(self.device)
        x_qry = torch.reshape(x_qry, (M * self.Q, self.dim ** 2)).to(self.device)
        y_qry = torch.reshape(y_qry, (M * self.Q, 1)).to(self.device)

        # -- shuffle
        if self.iid:
            perm = np.random.choice(range(M * self.K), M * self.K, False)

            x_trn = x_trn[perm]
            y_trn = y_trn[perm]

        return x_trn, y_trn, x_qry, y_qry