import gzip
import os
import shutil
import zipfile
from typing import Literal

import numpy as np
import requests
import torch
import torchvision
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


class EmnistDataset(Dataset):
    """
        EMNIST Dataset class.

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
        use_jax: bool = False,
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
        self.use_jax = use_jax

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

        if not self.use_jax:
            img = torch.cat(img)
            idx_vec = index * torch.ones(self.maxTrainingDataPerClass + self.queryDataPerClass, dtype=int)
        else:
            img = np.stack([np.array(im) for im in img], axis=0)
            idx_vec = np.array(index * np.ones(self.maxTrainingDataPerClass + self.queryDataPerClass, dtype=int))

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


class SplitFashionMnistDataset(Dataset):
    """
        Split Fashion MNIST Dataset class.

    Constructs training and query sets for meta-training from a subset of
    Fashion MNIST classes. Note that rather than a single image and the
    corresponding label, each data point represents samples from a class of
    images, containing training and query data from that category.
    """

    def __init__(
        self,
        minTrainingDataPerClass: int,
        maxTrainingDataPerClass: int,
        queryDataPerClass: int,
        dimensionOfImage: int,
        class_indices: list[int],
        all_classes: bool = False,
    ):
        """
            Initialize the SplitFashionMnistDataset class.

        The method first downloads and preprocesses the Fashion MNIST dataset,
        creating directories and files necessary for later use.

        :param trainingDataPerClass: (int) integer value representing the number of training data per class,
        :param queryDataPerClass: (int) integer value representing the number of query data per class,
        :param dimensionOfImage: (int) integer value representing the dimension size of the images.
        :param class_indices: (list[int]) list of class indices to include in the dataset.
        """

        # -- create directory
        s_dir = os.getcwd()
        self.fashion_mnist_dir = s_dir + "/data/fashion_mnist/"
        os.makedirs(self.fashion_mnist_dir, exist_ok=True)

        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.class_indices = class_indices
        self.all_classes = all_classes
        self.current_idx = 0

        # -- process data
        self.transform = transforms.Compose(
            [
                transforms.Resize((dimensionOfImage, dimensionOfImage)),
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
        return len(self.class_indices)

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
            if self.current_idx == len(self.class_indices):
                self.current_idx = 0
        class_index = self.class_indices[index]

        train_idx = self.train_dataset.targets == torch.tensor(class_index)
        train_idx = np.where(train_idx)[0]
        query_idx = self.test_dataset.targets == class_index
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
            torch.tensor([class_index] * self.maxTrainingDataPerClass),
            q_transformed_data,  # .float() / 255,
            torch.tensor([class_index] * self.queryDataPerClass),
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


class AddingTaskDataset(Dataset):
    """
        Adding Task Dataset class.

    Constructs training and query sets for meta-training on the adding task.
    Each data point consists of two sequences of numbers and a target value
    representing the sum of two specific elements from the first sequence,
    as indicated by the second sequence.
    """

    def __init__(
        self,
        minSequenceLength: int,
        maxSequenceLength: int,
        numberOfSequences: int,
        device: Literal["cpu", "cuda"] = "cpu",
        use_jax: bool = False,
    ):
        """
            Initialize the AddingTaskDataset class.

        :param sequenceLength: (int) Length of each sequence,
        :param numberOfSequences: (int) Number of sequences in the dataset,
        :param device: (str) The processing device to use. Default is 'cpu',
        """
        self.minSequenceLength = minSequenceLength
        self.maxSequenceLength = maxSequenceLength
        self.numberOfSequences = numberOfSequences
        self.device = device
        self.use_jax = use_jax

    def __len__(self):
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of sequences
            in the dataset
        """
        return self.numberOfSequences

    def __getitem__(self, index: int):
        """
            Return a data point from the dataset based on the given index.

        Each data point consists of two sequences of numbers and a target value
        representing the sum of two specific elements from the first sequence,
        as indicated by the second sequence.

        :param index: (int) Index of the data point to be retrieved.
        :return: tuple: A tuple containing two sequences and a target value.
        """
        sequenceLength = np.random.randint(self.minSequenceLength, self.maxSequenceLength + 1)
        seq1 = np.random.rand(sequenceLength).astype(np.float32)
        idx1 = np.random.randint(0, sequenceLength // 2)
        idx2 = np.random.randint(sequenceLength // 2, sequenceLength)
        seq2 = np.zeros(sequenceLength, dtype=np.float32)
        seq2[idx1] = 1.0
        seq2[idx2] = 1.0
        target = seq1[idx1] + seq1[idx2]

        if not self.use_jax:
            seq1 = torch.tensor(seq1).to(self.device)
            seq2 = torch.tensor(seq2).to(self.device)
            target = torch.tensor(target).to(self.device)
        return seq1, seq2, target


class AddBernoulliTaskDataset(Dataset):
    """
        Add Bernoulli Task Dataset class.

    Constructs training and query sets for meta-training on the adding
    Bernoulli task. Each data point consists of 1 sequence of Bernoulli
    numbers and a target value representing the sum of two specific
    numbers from the sequence, with a lag
    """

    def __init__(
        self,
        minSequenceLength: int,
        maxSequenceLength: int,
        querySequenceLength: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
        use_jax: bool = False,
    ):
        """
            Initialize the AddBernoulliTaskDataset class.

        :param sequenceLength: (int) Length of each sequence,
        :param numberOfSequences: (int) Number of sequences in the dataset,
        :param device: (str) The processing device to use. Default is 'cpu',
        """
        self.minSequenceLength = minSequenceLength
        self.maxSequenceLength = maxSequenceLength
        self.querySequenceLength = querySequenceLength
        self.device = device
        self.use_jax = use_jax

    def __len__(self):
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of sequences
            in the dataset
        """
        return 1000000  # Arbitrary large number for infinite sampling

    def __getitem__(self, index: int):

        sequenceLength = np.random.randint(self.minSequenceLength, self.maxSequenceLength + 1)
        seq1 = np.random.binomial(1, 0.5, sequenceLength).astype(np.float32)
        seq1 = torch.tensor(seq1)
        seq2 = np.random.binomial(1, 0.5, self.querySequenceLength).astype(np.float32)
        seq2 = torch.tensor(seq2)
        return seq1, seq2


class AddBernoulliTaskDataProcess:
    """
        Add Bernoulli Task data processor class.

    The function is designed to process adding Bernoulli task data, specifically
    sequences and their corresponding targets.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        min_lag_1: int = 1,
        max_lag_1: int = 10,
        min_lag_2: int = 1,
        max_lag_2: int = 10,
        use_jax: bool = False,
    ):
        """
            Initialize the AddBernoulliTaskDataProcess object.

        :param device: (str) The processing device to use. Default is 'cpu',
        """
        self.device = device
        self.use_jax = use_jax
        self.min_lag_1 = min_lag_1
        self.max_lag_1 = max_lag_1
        self.min_lag_2 = min_lag_2
        self.max_lag_2 = max_lag_2

    def __call__(self, data):
        """
            Processing adding Bernoulli task data.

        :param data: (tuple) A tuple of sequences and their corresponding targets.
        :return: tuple: A tuple of processed sequences and their corresponding targets.
        f(x) = 0.5 + 0.5*(x_{t-lag_1}) - 0.25*(x_{t-lag_2})
        """

        seq1, seq2 = data
        roll_1 = np.random.randint(self.min_lag_1, self.max_lag_1 + 1)
        roll_2 = np.random.randint(self.min_lag_2, self.max_lag_2 + 1)

        rolled_seq_11 = torch.roll(seq1, shifts=roll_1, dims=1)
        rolled_seq_12 = torch.roll(seq1, shifts=roll_2, dims=1)

        rolled_seq_21 = torch.roll(seq2, shifts=roll_1, dims=1)
        rolled_seq_22 = torch.roll(seq2, shifts=roll_2, dims=1)

        rolled_seq_11 = rolled_seq_11[:, roll_1:]
        rolled_seq_12 = rolled_seq_12[:, roll_2:]
        rolled_seq_21 = rolled_seq_21[:, roll_1:]
        rolled_seq_22 = rolled_seq_22[:, roll_2:]

        rolled_seq_11 = torch.cat((torch.zeros((seq1.shape[0], roll_1), device=seq1.device), rolled_seq_11), dim=1)
        rolled_seq_12 = torch.cat((torch.zeros((seq1.shape[0], roll_2), device=seq1.device), rolled_seq_12), dim=1)

        rolled_seq_21 = torch.cat((torch.zeros((seq2.shape[0], roll_1), device=seq2.device), rolled_seq_21), dim=1)
        rolled_seq_21[:, 0:roll_1] = seq1[:, -roll_1:]  # add dependency
        rolled_seq_22 = torch.cat((torch.zeros((seq2.shape[0], roll_2), device=seq2.device), rolled_seq_22), dim=1)
        rolled_seq_22[:, 0:roll_2] = seq1[:, -roll_2:]  # add dependency

        y_05 = torch.ones_like(seq1) * 0.5
        y_052 = torch.ones_like(seq2) * 0.5

        target = y_05 + rolled_seq_11 * 0.5 - rolled_seq_12 * 0.25
        target_2 = y_052 + rolled_seq_21 * 0.5 - rolled_seq_22 * 0.25

        seq1 = torch.cat((seq1, 1 - seq1), dim=0).T
        seq2 = torch.cat((seq2, 1 - seq2), dim=0).T
        target = torch.cat((target, 1 - target), dim=0).T
        target_2 = torch.cat((target_2, 1 - target_2), dim=0).T

        if not self.use_jax:
            seq1 = seq1.to(self.device)
            seq2 = seq2.to(self.device)
            target = target.to(self.device)
            target_2 = target_2.to(self.device)
        else:
            seq1 = np.array(seq1)
            seq2 = np.array(seq2)
            target = np.array(target)
            target_2 = np.array(target_2)

        return seq1, target, seq2, target_2, roll_1, roll_2


class IMDBMetaDataset(Dataset):
    """
    IMDB Dataset for Meta-Learning without torchtext.
    Uses Hugging Face 'datasets' for stability on Windows.
    """

    def __init__(self, minNumberOfSequences, maxNumberOfSequences, query_q, max_seq_len=200):
        self.minNumberOfSequences = minNumberOfSequences
        self.maxNumberOfSequences = maxNumberOfSequences
        self.query_q = query_q
        self.max_seq_len = max_seq_len
        self.current_idx = 0

        # 1. Load dataset (Automatically downloads if not present)
        # IMDB has 'train' and 'test' splits
        raw_datasets = load_dataset("imdb")

        # 2. Setup Tokenizer (Fast, modern alternative to torchtext)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 3. Organize by class: 0 (Neg), 1 (Pos)
        # We combine train/test to have a larger pool for meta-learning tasks
        self.class_data = {0: [], 1: []}

        cache_dir = os.path.join(os.getcwd(), "data", "imdb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"imdb_meta_{max_seq_len}.pt")

        if os.path.exists(cache_file):
            self.class_data = torch.load(cache_file, weights_only=True)
        else:
            print("No cache found. Processing raw data...")
            raw_datasets = load_dataset("imdb")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            self.class_data = {0: [], 1: []}
            for split in ["train", "test"]:
                for item in raw_datasets[split]:
                    tokens = self.tokenizer(
                        item["text"], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt"
                    )
                    data_entry = {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }
                    self.class_data[item["label"]].append(data_entry)

            # Save the processed dictionary
            torch.save(self.class_data, cache_file)

    def __len__(self):
        # 2 classes: Positive and Negative
        return 2

    def __getitem__(self, index):
        # index 0 = Negative, index 1 = Positive
        index = self.current_idx
        self.current_idx += 1
        if self.current_idx == 2:
            self.current_idx = 0
        all_samples = self.class_data[index]

        # Determine K-shot
        k_shot = self.maxNumberOfSequences
        total_needed = k_shot + self.query_q

        # Sample unique indices
        indices = np.random.choice(len(all_samples), total_needed, replace=False)

        def gather_data(idxs):
            texts = torch.stack([all_samples[i]["input_ids"] for i in idxs])
            masks = torch.stack([all_samples[i]["attention_mask"] for i in idxs])
            return texts, masks

        support_texts, support_masks = gather_data(indices[:k_shot])
        query_texts, query_masks = gather_data(indices[k_shot:])

        # Create labels
        support_labels = torch.full((k_shot,), index, dtype=torch.long)
        query_labels = torch.full((self.query_q,), index, dtype=torch.long)

        return support_texts, support_masks, support_labels, query_texts, query_masks, query_labels


class IMDBDataProcess:
    """
        IMDB data processor class.

    The function is designed to process IMDB data, specifically sequences and
    their corresponding targets. The function performs several operations,
    including:
    1) Transferring the processed data to the specified processing device,
        which could either be 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        use_jax: bool = False,
        minNumberOfSequencesPerClass: int = 20,
        maxNumberOfSequencesPerClass: int = 40,
    ):
        """
            Initialize the IMDBDataProcess object.

        :param device: (str) The processing device to use. Default is 'cpu',
        """
        self.device = device
        self.use_jax = use_jax
        self.minNumberOfSequencesPerClass = minNumberOfSequencesPerClass
        self.maxNumberOfSequencesPerClass = maxNumberOfSequencesPerClass

        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    def __call__(self, data):
        """
            Processing IMDB data.

        :param data: (tuple) A tuple of sequences and their corresponding targets.
        :return: tuple: A tuple of processed sequences and their corresponding targets.
        """
        current_sequences_per_class = 0
        if self.maxNumberOfSequencesPerClass == self.minNumberOfSequencesPerClass:
            current_sequences_per_class = self.minNumberOfSequencesPerClass
        else:
            current_sequences_per_class = np.random.randint(
                self.minNumberOfSequencesPerClass, self.maxNumberOfSequencesPerClass
            )
        support_texts, support_masks, support_labels, query_texts, query_masks, query_labels = data
        support_texts = support_texts[:, :current_sequences_per_class, :]
        support_masks = support_masks[:, :current_sequences_per_class, :]
        support_labels = support_labels[:, :current_sequences_per_class]

        support_texts = torch.reshape(
            support_texts, (current_sequences_per_class * support_texts.shape[0], support_texts.shape[2])
        )  # reshape to (K, seq_len)
        support_masks = torch.reshape(
            support_masks, (current_sequences_per_class * support_masks.shape[0], support_masks.shape[2])
        )  # reshape to (K, seq_len)
        support_texts = self.bert(support_texts, attention_mask=support_masks).last_hidden_state
        support_labels = torch.reshape(
            support_labels, (current_sequences_per_class * support_labels.shape[0], 1)
        )  # reshape to (K,)

        query_texts = torch.reshape(
            query_texts, (query_texts.shape[0] * query_texts.shape[1], query_texts.shape[2])
        )  # reshape to (Q, seq_len)
        query_masks = torch.reshape(
            query_masks, (query_masks.shape[0] * query_masks.shape[1], query_masks.shape[2])
        )  # reshape to (Q, seq_len)
        query_texts = self.bert(query_texts, attention_mask=query_masks).last_hidden_state
        query_labels = torch.reshape(
            query_labels, (query_labels.shape[0] * query_labels.shape[1], 1)
        )  # reshape to (Q,)

        if not self.use_jax:
            support_texts = support_texts.to(self.device)
            support_labels = support_labels.to(self.device)
            query_texts = query_texts.to(self.device)
            query_labels = query_labels.to(self.device)
        else:
            support_texts = np.array(support_texts)
            support_labels = np.array(support_labels)
            query_texts = np.array(query_texts)
            query_labels = np.array(query_labels)

        return support_texts, support_labels, query_texts, query_labels, current_sequences_per_class


class RnnDataProcess:
    """
        RNN data processor class.

    The function is designed to process RNN data, specifically sequences and
    their corresponding targets. The function performs several operations,
    including:
    1) Transferring the processed data to the specified processing device,
        which could either be 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        minNumberOfSequences: int = 20,
        maxNumberOfSequences: int = 40,
        use_jax: bool = False,
    ):
        """
            Initialize the RnnDataProcess object.

        :param device: (str) The processing device to use. Default is 'cpu',
        """
        self.device = device
        self.use_jax = use_jax
        self.minNumberOfSequences = minNumberOfSequences
        self.maxNumberOfSequences = maxNumberOfSequences

    def __call__(self, data):
        """
            Processing RNN data.

        :param data: (tuple) A tuple of sequences and their corresponding targets.
        :return: tuple: A tuple of processed sequences and their corresponding targets.
        """

        seq, target = data

        if not self.use_jax:
            seq = seq.to(self.device)
            target = target.to(self.device)

        return seq, target


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
        use_jax: bool = False,
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
        self.use_jax = use_jax

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
        if not self.use_jax:
            x_trn = torch.reshape(x_trn, (classes * current_training_data_per_class, self.dimensionOfImage**2)).to(
                self.device
            )
            y_trn = torch.reshape(y_trn, (classes * current_training_data_per_class, 1)).to(self.device)
            x_qry = torch.reshape(x_qry, (classes * self.queryDataPerClass, self.dimensionOfImage**2)).to(self.device)
            y_qry = torch.reshape(y_qry, (classes * self.queryDataPerClass, 1)).to(self.device)
        else:
            x_trn = np.reshape(x_trn, (classes * current_training_data_per_class, self.dimensionOfImage**2))
            y_trn = np.reshape(y_trn, (classes * current_training_data_per_class, 1))
            x_qry = np.reshape(x_qry, (classes * self.queryDataPerClass, self.dimensionOfImage**2))
            y_qry = np.reshape(y_qry, (classes * self.queryDataPerClass, 1))

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
