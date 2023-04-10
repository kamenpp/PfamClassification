import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse


class OneHotDataset(Dataset):
    """
    A PyTorch dataset class for one-hot encoded features and labels.
    Args:
    features (List[sparse.csr.csr_matrix]): A list of sparse matrices of shape (padded_sequence_length, n_features)
        representing the one-hot encoded features of the dataset (n_features is the one-hot encoding length).
    labels (sparse.csr.csr_matrix): A sparse matrix of shape (n_samples, n_classes)
        representing the one-hot encoded labels of the dataset.

    Returns:
        A PyTorch dataset instance that can be used with DataLoader for efficient mini-batch processing.
    """

    def __init__(self,
                 features: np.ndarray[sparse.csr.csr_matrix],
                 labels: sparse.csr.csr_matrix):
        super(OneHotDataset, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset at the specified index."""
        return torch.from_numpy(self.features[idx].toarray()), torch.from_numpy(self.labels[idx].toarray())
