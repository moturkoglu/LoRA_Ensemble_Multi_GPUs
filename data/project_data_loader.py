#!/usr/bin/env python

"""
Implements the data loaders for this project
"""

### IMPORTS ###
# Built-in imports
import pickle

# Lib imports
import torch 
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from torchvision import transforms
import json
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.datasets.utils import download_url, verify_str_arg


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class HAM10000(Dataset):
    """
    A dataloader for the HAM10000 dataset.
    """
     
    def __init__(self, labels, dataset_path, transform=None):
        """
        A dataloader for the HAM10000 dataset.

        Parameters
        ----------
        labels : pd.DataFrame
            The dataframe containing the labels
        dataset_path : Path
            The path to the dataset
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label

        Labels in the dataset:
        ----------------------
        - Benign keratosis-like lesions (bkl): 0 (∼11.0%)
        - Melanocytic nevi (nv): 1 (∼66.9%)
        - Dermatofibroma (df): 2 (∼1.1%)
        - Melanoma (mel):	3 (∼11.1%)
        - Vascular lesions (vasc): 4 (∼1.4%)
        - Basal cell carcinoma (bcc): 5 (∼5.1%)
        - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec): 6 (∼3.2%)
        """

        self.labels = labels
        self.path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # get image name
        img_name = self.labels.iloc[idx, 0]

        # get image path
        img_path = self.path.joinpath('HA_10000_images').joinpath(img_name + '.jpg')

        # get image
        image = plt.imread(img_path)

        # get label
        label = self.labels.iloc[idx, 2]

        # convert image to tensor and permute the dimensions
        image = torch.tensor(image).permute(2, 0, 1).float()

        # apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
    

def unpickle(file):
    """
    Function to unpickle a file

    Parameters
    ----------
    file : str
        Path to the file to unpickle

    Returns
    -------
    dict
        The unpickled dictionary
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10(Dataset):
    """
    A dataloader for the CIFAR10 dataset.
    """

    def __init__(self, batch_files, transform=None):
        """
        A dataloader for the CIFAR10 dataset.

        Parameters
        ----------
        batch_files : List
            List of paths to the batch files
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        # Create a list to store the data and labels
        self.data = []
        self.labels = []

        # Load the data from the batch files
        for file in batch_files:
            # Unpickle the file
            batch_data = unpickle(file)
            # Append the data and labels to the lists
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])
        # Create a tensor from the data
        self.data = np.concatenate(self.data, axis=0)  
        self.data = torch.tensor(self.data)
        # Create a tensor from the labels
        self.labels = torch.tensor(self.labels)
        # Store the transformation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters
        ----------
        idx : int
            The index of the image to return

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        img = self.data[idx]
        img = img.reshape(3, 32, 32)  
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img.float())
        return img, label


class CIFAR100(Dataset):
    """
    A dataloader for the CIFAR100 dataset.
    """

    def __init__(self, batch_files, transform=None):
        """
        A dataloader for the CIFAR100 dataset.

        Parameters
        ----------
        batch_files : list
            List of paths to the batch files
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        # Create a list to store the data and labels
        self.data = []
        self.labels = []

        # Load the data from the batch files
        for file in batch_files:
            # Unpickle the file
            batch_data = unpickle(file)
            # Append the data and labels to the lists
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'fine_labels'])
        # Create a tensor from the data
        self.data = np.concatenate(self.data, axis=0)
        self.data = torch.tensor(self.data)
        # Create a tensor from the labels
        self.labels = torch.tensor(self.labels)
        # Store the transformation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters
        ----------
        idx : int
            The index of the image to return

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        img = self.data[idx]
        img = img.reshape(3, 32, 32)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img.float())
        return img, label 
    


class INat2017(VisionDataset):
    """`iNaturalist 2017 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(INat2017, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            if not (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['annos'][1]))):
                print('Downloading...')
                self._download()
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.file_list['imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['annos'][1]))
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']
        image =plt.imread(path)

        # check if numpy array has dimension (x, y, 3)$
        if len(image.shape) == 2:
            # Convert grayscale to RGB by duplicating the channel
            image = image[:, :, None]  # Add a new channel dimension
            image = image.repeat(3, axis=2)  # Repeat the channel 3 times
        elif image.shape[2] == 4:
            image = image[:, :, :3]  

        elif image.shape[2] != 3:
            # return zero image
            image = np.zeros((224, 224, 3))
       
        image = torch.tensor(image).permute(2, 0, 1)
        #if self.split == 'train':
        #    # AugMix: Perform data augmentation
        #    image = transforms.AugMix()(image)
        
        #self.plot_image(image)
        image = image.float()
 
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def plot_image(self, image):
        """Utility function to plot an image."""
        if torch.is_tensor(image):
            # Convert CHW to HWC for plotting
            image = image.permute(1, 2, 0).numpy()

            # max and min pixel values
            min_val = image.min()
            max_val = image.max()
            print(f'Min pixel value: {min_val}')
            print(f'Max pixel value: {max_val}')

        # denormalize
        #image = (image) * 255
        #min_val = image.min()
        #max_val = image.max()
        #print(f'Min pixel value: {min_val}')
        # Convert to uint8
        #image = image.astype(np.uint8)
        if isinstance(image, transforms.ToTensor):
            image = transforms.ToPILImage()(image)  # Convert back to PIL for display

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _download(self):
        for url, filename, md5 in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
            if not check_integrity(os.path.join(self.root, filename), md5):
                raise RuntimeError("File not found or corrupted.")
            
