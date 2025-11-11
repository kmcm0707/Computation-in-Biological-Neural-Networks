import gzip
import os
import shutil
import zipfile
from typing import Literal

import numpy as np
import requests
import torch
import torch.utils
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EmnistDataset(Dataset):
    """
        EMNIST Dataset class.

    Constructs training and query sets for meta-training. Note that rather
    than a single image and the corresponding label, each data point
    represents samples from a class of images, containing training and query
    data from that category.
    """

    def __init__(
        self, minTrainingDataPerClass: int, maxTrainingDataPerClass: int, queryDataPerClass: int, dimensionOfImage: int
    ):
        """
            Initialize the EmnistDataset class.

        The method first downloads and preprocesses the EMNIST dataset, creating
        directories and files necessary for later use.

        :param trainingDataPerClass: (int) integer value representing the number of training data per class,
        :param queryDataPerClass: (int) integer value representing the number of query data per class,
        :param dimensionOfImage: (int) integer value representing the dimension size of the images.
        """
        try:
            # -- create directory
            s_dir = os.getcwd()
            self.emnist_dir = s_dir + "/data/emnist/"
            file_name = "gzip"
            os.makedirs(self.emnist_dir)

            # -- download
            emnist_url = "https://biometrics.nist.gov/cs_links/EMNIST/"
            self.download(emnist_url + file_name + ".zip", self.emnist_dir + file_name + ".zip")

            # -- unzip
            with zipfile.ZipFile(self.emnist_dir + file_name + ".zip", "r") as zip_file:
                zip_file.extractall(self.emnist_dir)
            os.remove(self.emnist_dir + file_name + ".zip")

            balanced_path = [f for f in [fs for _, _, fs in os.walk(self.emnist_dir + file_name)][0] if "balanced" in f]
            for file in balanced_path:
                with gzip.open(self.emnist_dir + "gzip/" + file, "rb") as f_in:
                    try:
                        f_in.read(1)
                        with open(self.emnist_dir + file[:-3], "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    except OSError:
                        pass
            shutil.rmtree(self.emnist_dir + file_name)

            # -- write images
            self.write_to_file()

            remove_path = [files for _, folders, files in os.walk(self.emnist_dir) if folders][0]
            for path in remove_path:
                os.unlink(self.emnist_dir + path)

        except FileExistsError:
            pass

        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass

        self.char_path = [folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders]
        self.transform = transforms.Compose(
            [
                transforms.Resize((dimensionOfImage, dimensionOfImage)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def download(url, filename):
        """
            A static method to download a file from a URL and save it to a local file.

        :param url: (str) A string representing the URL from which to download the file,
        :param filename: (str) A string representing the name of the local file to save
            the downloaded data to.
        :return: None
        """
        res = requests.get(url, stream=False)
        with open(filename, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def write_to_file(self):
        """
            Write EMNIST images to files.

        The function reads the EMNIST test images and labels from binary files and
        writes them to files. Each image is saved to a file under a directory
        corresponding to its label.

        :return: None.
        """
        n_class = 47

        # -- read images
        with open(self.emnist_dir + "emnist-balanced-test-images-idx3-ubyte", "rb") as f:
            f.read(3)
            image_count = int.from_bytes(f.read(4), "big")
            height = int.from_bytes(f.read(4), "big")
            width = int.from_bytes(f.read(4), "big")
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape((image_count, height, width))

        # -- read labels
        with open(self.emnist_dir + "emnist-balanced-test-labels-idx1-ubyte", "rb") as f:
            f.read(3)
            label_count = int.from_bytes(f.read(4), "big")
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        assert image_count == label_count

        # -- write images
        for i in range(n_class):
            os.mkdir(self.emnist_dir + f"character{i + 1:02d}")

        char_path = sorted([folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders])

        label_counter = np.ones(n_class, dtype=int)
        for i in range(label_count):
            im = Image.fromarray(images[i])
            im.save(char_path[labels[i]] + f"/{labels[i] + 1:02d}_{label_counter[labels[i]]:04d}.png")

            label_counter[labels[i]] += 1

    def __len__(self):
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of classes in the
            dataset
        """
        return len(self.char_path)

    def __getitem__(self, index: int):
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
        :return: tuple: A tuple of tensors containing training and query images and
            corresponding labels for a given index and current training data per class.
        """
        img = []
        for img_ in os.listdir(self.char_path[index]):
            ims = self.transform(Image.open(self.char_path[index] + "/" + img_, mode="r").convert("L"))
            img.append(ims)

        img = torch.cat(img)
        idx_vec = index * torch.ones(self.maxTrainingDataPerClass + self.queryDataPerClass, dtype=int)

        return (
            img[: self.maxTrainingDataPerClass],
            idx_vec[: self.maxTrainingDataPerClass],
            img[self.maxTrainingDataPerClass : self.maxTrainingDataPerClass + self.queryDataPerClass],
            idx_vec[self.maxTrainingDataPerClass : self.maxTrainingDataPerClass + self.queryDataPerClass],
        )


class FashionMnistDataset(Dataset):
    """
        Fashion MNIST Dataset class.

    Constructs training and query sets for meta-training. Note that rather
    than a single image and the corresponding label, each data point
    represents samples from a class of images, containing training and query
    data from that category.
    """

    def __init__(
        self,
        minTrainingDataPerClass: int,
        maxTrainingDataPerClass: int,
        queryDataPerClass: int,
        dimensionOfImage: int,
        all_classes: bool = False,
    ):
        """
            Initialize the FashionMnistDataset class.

        The method first downloads and preprocesses the Fashion MNIST dataset, creating
        directories and files necessary for later use.

        :param trainingDataPerClass: (int) integer value representing the number of training data per class,
        :param queryDataPerClass: (int) integer value representing the number of query data per class,
        :param dimensionOfImage: (int) integer value representing the dimension size of the images.
        """

        # -- create directory
        s_dir = os.getcwd()
        self.fashion_mnist_dir = s_dir + "/data/fashion_mnist/"
        os.makedirs(self.fashion_mnist_dir, exist_ok=True)

        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.all_classes = all_classes
        self.current_idx = 0

        # -- process data
        self.transform = transforms.Compose(
            [
                transforms.Resize((dimensionOfImage, dimensionOfImage)),
                # transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # -- download data
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=self.fashion_mnist_dir, download=True, train=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=self.fashion_mnist_dir, download=True, train=False, transform=self.transform
        )

    def __len__(self):
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of classes in the
            dataset
        """
        return 10

    def __getitem__(self, index: int):
        """
            Get a tuple of tensors containing training and query images and
            corresponding labels for a given index.

        The images are loaded from the Fashion MNIST dataset. Each image is
        converted to grayscale and resized to the specified dimension using
        `torchvision.transforms.Resize` and `torchvision.transforms.ToTensor`.
        The returned tuples contain tensors of K and Q images, where K is the
        training data size per class and Q is the query data size per class.
        Both K and Q are specified at initialization. The indices corresponding
        to the images are also returned in tensors of size K and Q, respectively.

        :param index: (int) Index of the character folder from which images are to be retrieved.
        :return: tuple: A tuple of tensors containing training and query images and
            corresponding labels for a given index and current training data per class.

        """
        if self.all_classes:
            index = self.current_idx
            self.current_idx += 1
            if self.current_idx == 10:
                self.current_idx = 0

        train_idx = self.train_dataset.targets == torch.tensor(index)
        train_idx = np.where(train_idx)[0]
        query_idx = self.test_dataset.targets == index
        query_idx = np.where(query_idx)[0]

        train_idx = np.random.choice(train_idx, self.maxTrainingDataPerClass, False)
        query_idx = np.random.choice(query_idx, self.queryDataPerClass, False)

        transformed_data = []
        for idx in train_idx:
            transformed_data.append(self.transform(self.train_dataset.data[idx].float().unsqueeze_(0) / 255))

        if self.maxTrainingDataPerClass > 0:
            transformed_data = torch.cat(transformed_data)
        else:
            transformed_data = torch.empty((0, 1, 28, 28))

        q_transformed_data = []
        for idx in query_idx:
            q_transformed_data.append(self.transform(self.test_dataset.data[idx].float().unsqueeze_(0) / 255))
        q_transformed_data = torch.cat(q_transformed_data)

        return (
            transformed_data,  # .float() / 255,
            torch.tensor([index] * self.maxTrainingDataPerClass),
            q_transformed_data,  # .float() / 255,
            torch.tensor([index] * self.queryDataPerClass),
        )


class CombinedDataset(Dataset):
    """
        Combined Dataset class.

    Combines multiple datasets into a single dataset for meta-training.
    """

    def __init__(
        self,
        EMNIST_before_FashionMNIST: bool,
        minEmnistTrainingDataPerClass: int,
        maxEmnistTrainingDataPerClass: int,
        emnistQueryDataPerClass: int,
        minFashionMnistTrainingDataPerClass: int,
        maxFashionMnistTrainingDataPerClass: int,
        fashionMnistQueryDataPerClass: int,
        dimensionOfImage: int,
        all_fashion_mnist_classes: bool = True,
    ):
        """
        Initialize the CombinedDataset class.
        """
        self.EMNIST_before_FashionMNIST = EMNIST_before_FashionMNIST
        self.minEmnistTrainingDataPerClass = minEmnistTrainingDataPerClass
        self.maxEmnistTrainingDataPerClass = maxEmnistTrainingDataPerClass
        self.emnistQueryDataPerClass = emnistQueryDataPerClass
        self.minFashionMnistTrainingDataPerClass = minFashionMnistTrainingDataPerClass
        self.maxFashionMnistTrainingDataPerClass = maxFashionMnistTrainingDataPerClass
        self.fashionMnistQueryDataPerClass = fashionMnistQueryDataPerClass
        self.dimensionOfImage = dimensionOfImage
        self.all_fashion_mnist_classes = all_fashion_mnist_classes

        s_dir = os.getcwd()
        self.emnist_dir = s_dir + "/data/emnist/"
        self.fashion_mnist_dir = s_dir + "/data/fashion_mnist/"

        self.Emnist_char_path = [folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders]
        self.Emnist_transform = transforms.Compose(
            [
                transforms.Resize((dimensionOfImage, dimensionOfImage)),
                transforms.ToTensor(),
            ]
        )

        # -- download data
        self.fashion_train_dataset = torchvision.datasets.FashionMNIST(
            root=self.fashion_mnist_dir, download=True, train=True, transform=self.transform
        )
        self.fashion_test_dataset = torchvision.datasets.FashionMNIST(
            root=self.fashion_mnist_dir, download=True, train=False, transform=self.transform
        )
        self.fashion_transform = transforms.Compose(
            [
                transforms.Resize((dimensionOfImage, dimensionOfImage)),
            ]
        )

    def __len__(self):
        """
            Get the length of the combined dataset.

        :return: int: the length of the combined dataset, i.e., the sum of lengths
            of all individual datasets.
        """
        return 47 + 10  # EMNIST has 47 classes, Fashion MNIST has 10 classes

    def __getitem__(self, index: int):
        """
            Return a data point from the combined dataset based on the given index.

        The method determines which individual dataset the index corresponds to
        and retrieves the data point from that dataset.

        :param index: (int) Index of the data point to be retrieved.
        :return: The data point from the appropriate individual dataset.
        """
        img = []
        for img_ in os.listdir(self.char_path[index]):
            ims = self.transform(Image.open(self.char_path[index] + "/" + img_, mode="r").convert("L"))
            img.append(ims)

        img = torch.cat(img)
        idx_vec = index * torch.ones(self.maxTrainingDataPerClass + self.queryDataPerClass, dtype=int)

        return (
            img[: self.maxTrainingDataPerClass],
            idx_vec[: self.maxTrainingDataPerClass],
            img[self.maxTrainingDataPerClass : self.maxTrainingDataPerClass + self.queryDataPerClass],
            idx_vec[self.maxTrainingDataPerClass : self.maxTrainingDataPerClass + self.queryDataPerClass],
        )


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

    def __init__(
        self,
        minTrainingDataPerClass: int,
        maxTrainingDataPerClass: int,
        queryDataPerClass: int,
        dimensionOfImage: int,
        device: Literal["cpu", "cuda"] = "cpu",
        iid: bool = True,
    ):
        """
            Initialize the DataProcess object.

        :param queryDataPerClass: (int) query data set size per class,
        :param dimensionOfImage: (int) image dimension,
        :param device: (str) The processing device to use. Default is 'cpu',
        :param iid: (bool) shuffling flag. Default is True.
        """
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.device = device
        self.iid = iid
        self.dimensionOfImage = dimensionOfImage

    def __call__(self, data, classes: int):
        """
            Processing meta-training data.

        :param data: (tuple) A tuple of training and query data and the
            corresponding indices.
        :param classes: (int) The number of classes.
        :return: tuple: A tuple of processed training and query data and
            the corresponding indices.
        """

        # -- load data
        current_training_data_per_class = 0
        if self.maxTrainingDataPerClass == self.minTrainingDataPerClass:
            current_training_data_per_class = self.minTrainingDataPerClass
        else:
            current_training_data_per_class = np.random.randint(
                self.minTrainingDataPerClass, self.maxTrainingDataPerClass
            )

        x_trn, y_trn, x_qry, y_qry = data
        x_trn = x_trn[:, :current_training_data_per_class, :, :]
        y_trn = y_trn[:, :current_training_data_per_class]

        # -- reshape
        x_trn = torch.reshape(x_trn, (classes * current_training_data_per_class, self.dimensionOfImage**2)).to(
            self.device
        )
        y_trn = torch.reshape(y_trn, (classes * current_training_data_per_class, 1)).to(self.device)
        x_qry = torch.reshape(x_qry, (classes * self.queryDataPerClass, self.dimensionOfImage**2)).to(self.device)
        y_qry = torch.reshape(y_qry, (classes * self.queryDataPerClass, 1)).to(self.device)

        # -- shuffle
        if self.iid:
            perm = np.random.choice(
                range(classes * current_training_data_per_class),
                classes * current_training_data_per_class,
                False,
            )

            x_trn = x_trn[perm]
            y_trn = y_trn[perm]

        return x_trn, y_trn, x_qry, y_qry, current_training_data_per_class
