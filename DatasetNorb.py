#Imporitng Libraries
import struct
from itertools import groupby
from os import makedirs
from os.path import exists
from os.path import join

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
plt.ion()


class NORBExample:

    def __init__(self):
        self.image_lt = None
        self.image_rt = None
        self.category = None
        self.instance = None
        self.elevation = None
        self.azimuth = None
        self.lighting = None

    def __lt__(self, other):
        return self.category < other.category or \
               (self.category == other.category and self.instance < other.instance)

    def show(self, subplots):
        fig, axes = subplots
        fig.suptitle(
            'Category: {:02d} - Instance: {:02d} - Elevation: {:02d} - Azimuth: {:02d} - Lighting: {:02d}'.format(
                self.category, self.instance, self.elevation, self.azimuth, self.lighting))
        axes[0].axis('off')
        axes[0].imshow(self.image_lt, cmap='gray')
        axes[1].axis('off')
        axes[1].imshow(self.image_rt, cmap='gray')

    @property
    def pose(self):
        return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)


class NORBDataset:
    # Number of examples in both train and test set
    n_examples = 29160

    # Categories present in NORB dataset
    categories = ['animal', 'human', 'airplane', 'truck', 'car', 'blank']

    def __init__(self, dataset_root):
        """
        Initialize NORB dataset wrapper

        Parameters
        ----------
        dataset_root: str
            Path to directory where NORB archives have been extracted.
        """

        self.dataset_root = dataset_root
        self.initialized = False

        # Store path for each file in NORB dataset (for compatibility the original filename is kept)
        self.dataset_files = {
            'train': {
                'cat_1': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-01-cat.mat'),
                'cat_2': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-02-cat.mat'),
                'cat_3': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-03-cat.mat'),
                'cat_4': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-04-cat.mat'),
                'cat_5': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-05-cat.mat'),
                'cat_6': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-06-cat.mat'),
                'cat_7': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-07-cat.mat'),
                'cat_8': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-08-cat.mat'),
                'cat_9': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-09-cat.mat'),
                'cat_10': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Cat\\norb-5x46789x9x18x6x2x108x108-training-10-cat.mat'),
                'info_1': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-01-info.mat'),
                'info_2': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-02-info.mat'),
                'info_3': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-03-info.mat'),
                'info_4': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-04-info.mat'),
                'info_5': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-05-info.mat'),
                'info_6': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-06-info.mat'),
                'info_7': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-07-info.mat'),
                'info_8': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-08-info.mat'),
                'info_9': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-09-info.mat'),
                'info_10': join(self.dataset_root,
                                'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Info\\norb-5x46789x9x18x6x2x108x108-training-10-info.mat'),
                'dat_1': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-01-dat.mat'),
                'dat_2': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-02-dat.mat'),
                'dat_3': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-03-dat.mat'),
                'dat_4': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-04-dat.mat'),
                'dat_5': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-05-dat.mat'),
                'dat_6': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-06-dat.mat'),
                'dat_7': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-07-dat.mat'),
                'dat_8': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-08-dat.mat'),
                'dat_9': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-09-dat.mat'),
                'dat_10': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Training Dat\\norb-5x46789x9x18x6x2x108x108-training-10-dat.mat')
            },
            'test': {
                'cat_1': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Cat\\norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat'),
                'cat_2': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Cat\\norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat'),
                'info_1': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Info\\norb-5x01235x9x18x6x2x108x108-testing-01-info.mat'),
                'info_2': join(self.dataset_root,
                               'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Info\\norb-5x01235x9x18x6x2x108x108-testing-02-info.mat'),
                'dat_1': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Dat\\norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat'),
                'dat_2': join(self.dataset_root,
                              'C:\\Users\\user\\Desktop\\Norb Dataset\\Testing Dat\\norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat')
            }
        }

        # Initialize both train and test data structures
        self.data = {
            'train': [NORBExample() for _ in range(NORBDataset.n_examples)],
            'test': [NORBExample() for _ in range(NORBDataset.n_examples)]
        }

        # Fill data structures parsing dataset binary files
        for data_split in ['train']:
            self._fill_data_structures(data_split)

        self.initialized = True

    def explore_random_examples(self, dataset_split):
        """
        Visualize random examples for dataset exploration purposes

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None
        """
        if self.initialized:
            subplots = plt.subplots(nrows=1, ncols=2)
            plt.axis('off')
            for i in np.random.permutation(NORBDataset.n_examples):
                plt.axis('off')
                self.data[dataset_split][i].show(subplots)
                plt.waitforbuttonpress()
                plt.cla()

    def export_to_jpg(self, export_dir):
        """
        Export all dataset images to `export_dir` directory

        Parameters
        ----------
        export_dir: str
            Path to export directory (which is created if nonexistent)

        Returns
        -------
        None
        """
        if self.initialized:
            for split_name in ['train', 'test']:

                split_dir = join(export_dir, split_name)
                if not exists(split_dir):
                    makedirs(split_dir)

                for i, norb_example in tqdm(iterable=enumerate(self.data[split_name]),
                                            total=len(self.data[split_name]),
                                            desc='Exporting {} images to {}'.format(split_name, export_dir)):
                    category = NORBDataset.categories[norb_example.category]
                    instance = norb_example.instance

                    image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.jpg'.format(i, category, instance))
                    image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance))

                    imageio.imwrite(image_lt_path, norb_example.image_lt)
                    imageio.imwrite(image_rt_path, norb_example.image_rt)

    def group_dataset_by_category_and_instance(self, dataset_split):
        """
        Group NORB dataset for (category, instance) key

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        groups: list
            List of 25 groups of 972 elements each. All examples of each group are
            from the same category and instance
        """
        if dataset_split not in ['train', 'test']:
            raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))

        groups = []
        for key, group in groupby(iterable=sorted(self.data[dataset_split]),
                                  key=lambda x: (x.category, x.instance)):
            groups.append(list(group))

        return groups

    def _fill_data_structures(self, dataset_split):
        """
        Fill NORBDataset data structures for a certain `dataset_split`.

        This means all images, category and additional information are loaded from binary
        files of the current split.

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None

        """
        dat_data = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat_1'])
        cat_data = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat_1'])
        info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info_1'])
        for i, norb_example in enumerate(self.data[dataset_split]):
            norb_example.image_lt = dat_data[2 * i]
            norb_example.image_rt = dat_data[2 * i + 1]
            norb_example.category = cat_data[i]
            norb_example.instance = info_data[i][0]
            norb_example.elevation = info_data[i][1]
            norb_example.azimuth = info_data[i][2]
            norb_example.lighting = info_data[i][3]

    @staticmethod
    def matrix_type_from_magic(magic_number):
        """
        Get matrix data type from magic number

        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from NORB files

        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]

    @staticmethod
    def _parse_NORB_header(file_pointer):
        """
        Parse header of NORB binary file

        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a NORB binary file

        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': NORBDataset.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_cat_file(file_path):
        """
        Parse NORB category file

        Parameters
        ----------
        file_path: str
            Path of the NORB `*-cat.mat` file

        Returns
        -------
        """
        with open(file_path, mode='rb') as f:
            header = NORBDataset._parse_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(range(num_examples), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def _parse_NORB_dat_file(file_path):
        """
        Parse NORB data file

        Parameters
        ----------
        file_path: str
            Path of the NORB `*-dat.mat` file

        Returns
        -------
            Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
        with open(file_path, mode='rb') as f:
            header = NORBDataset._parse_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):
                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def _parse_NORB_info_file(file_path):
        """
        Parse NORB information file

        Parameters
        ----------
        file_path: str
            Path of the NORB `*-info.mat` file

        Returns
        -------

             - column 1: the instance in the category (0 to 9)
             - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70
               degrees from the horizontal respectively)
             - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
             - column 4: the lighting condition (0 to 5)
        """
        with open(file_path, mode='rb') as f:

            header = NORBDataset._parse_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(range(num_examples), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples
  
#Creating a dataset object
dataset = NORBDataset(dataset_root='./norb/')
#dataset.export_to_jpg(export_dir='C://Users//user//Image Classification of Norb Dataset//Norb_10')
for i in range(10):
    dataset.explore_random_examples(dataset_split='train')

img, label = dataset[0]
print(img.shape,label)
