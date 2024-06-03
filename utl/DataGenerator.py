import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import euclidean_distances

from utl.data_aug_op import random_flip_img, random_rotate_img
from utl.dataset import get_coordinates

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, k, data_set, siamese_model=None, mode="euclidean", shuffle=True, batch_size=32):
        self.data_set = data_set
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.siamese_model = siamese_model
        self.k = k
        self.mode = mode
        self.on_epoch_end()
        self.mode=mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_set)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_set))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        "returns one element from the data_set"
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.data_set[k] for k in indexes]

        X, f, y = self.__data_generation(list_IDs_temp)

        return [X[0], f[0]], y

    def __Get_exploration_order(self,data, shuffle):
        "shuffles the elements of the trainset"
        indexes = np.arange(len(data))

        if shuffle:
            random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_train):
        """

        Parameters
        ----------
        batch_train:  a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches

        Returns
        -------
        bag_batch: a list of np.ndarrays of size (numnber of patches,h,w,d) , each of which contains the patches of an image
        neighbors: a list  of the adjacency matrices of size (numnber of patches,number of patches) of every image
        bag_label: an np.ndarray of size (number of patches,1) reffering to the label of the image

        """
        bag_batch = []
        bag_label = []
        neighbors = []

        for ibatch, batch in enumerate(batch_train):
            aug_batch = []
            img_data = batch[0]
            for i in range(img_data.shape[0]):
                ori_img = img_data[i, :, :, :]

                if self.shuffle:
                    img = random_flip_img(ori_img, horizontal_chance=0.5, vertical_chance=0.5)
                    img = random_rotate_img(img)

                else:
                    img = ori_img
                exp_img = np.expand_dims(img, 0)

                aug_batch.append(exp_img)

            input_batch = np.concatenate(aug_batch)
            bag_batch.append((input_batch))
            bag_label.append(batch[1])

            Idx = self._get_indices(batch[2], neighbors=self.k)
            if self.mode=="siamese":
                siamese_pairs = self.generate_siamese_pairs(batch[0], Idx)

                adjacency_matrix = self.get_siamese_affinity(Idx, siamese_pairs)
            else:
                adjacency_matrix = self.get_knn_affinity(Idx)
            neighbors.append(adjacency_matrix)

        return bag_batch, neighbors, bag_label


    def generate_siamese_pairs(self, images, Idx):
        """

        Parameters
        ----------
        images :  np.ndarray of size (numnber of patches,h,w,d) contatining the pathes of an image
        Idx    : indices of the closest neighbors of every image

        Returns
        -------
        a list of np.ndarrays, pairing every patch of an image with its closest neighbors
        """

        image_pairs = []

        columns = (np.concatenate(np.asarray(Idx)).ravel())

        rows = [[enum] * len(item) if isinstance(item, np.ndarray) else np.asarray([enum]) for enum, item in
                enumerate(Idx)]
        rows = np.concatenate(np.asarray(rows)).ravel()

        for row, column in zip(rows, columns):
            image_pairs.append([images[int(row)], images[int(column)]])
        image_pairs = np.asarray(image_pairs)

        return [image_pairs[:, 0], image_pairs[:, 1]]

    def get_knn_affinity(self, Idx):
        """
        Create the adjacency matrix of each bag based on the euclidean distances between the patches
        Parameters
        ----------
        Idx:   a list of indices of the closest neighbors of every image

        Returns
        -------
        affinity:  an nxn np.ndarray that contains the neighborhood information for every patch.
        """

        affinity = np.zeros((Idx.shape[0], Idx.shape[0]), float)

        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()

        columns = Idx.ravel()

        affinity[rows, columns] = 1

        return affinity

    def get_siamese_affinity(self, Idx, train_set):
        """
        Create   :    the adjacency matrix of each bag based on the distance scores produced by the siamese network

        Parameters
        ----------
        Idx       :  nxk np.ndarray containing the indices of the closest spatial neigbors of every patch
        train_set :  list of (patches, label, filenames) describing a bag

        Returns
        -------
        affinity : an nxn np.ndarray that contains the neighborhood information for every patch.

        """

        predictions = self.siamese_model.predict(train_set)
        affinity = np.zeros((Idx.shape[0], Idx.shape[0]), float)

        columns = (np.concatenate(np.asarray(Idx)).ravel())

        rows = [[enum] * len(item) if isinstance(item, np.ndarray) else np.asarray([enum]) for enum, item in
                enumerate(Idx)]
        rows = np.concatenate(np.asarray(rows)).ravel().astype(int)

        affinity[rows, columns] = tf.squeeze(predictions)

        affinity = np.where(affinity > 0, np.exp(-affinity), 0)

        affinity = affinity.astype("float32")
        return affinity

    def _get_indices(self, filenames, neighbors):
        """
        Computes the indices that correspond to the closest neighbors of each patch in a bag
        Parameters
        ----------
        filenames: list of filenames of all the patches in a bag. Each filename has the following format imgx-xposd-yposd-class,
        which enables the extraction of its spatial coordinates (xpos,ypos)
        neighbors: number of neigbors to be taken into account

        Returns
        -------
        neigbor_indices: nxk np.ndarray containing the indices of the closest spatial neigbors of every patch
        """

        coordinates = []

        for enum, each_path in enumerate(filenames):
            coords = get_coordinates(each_path)
            coordinates.append(coords)

        patch_distances = euclidean_distances(coordinates)

        neighbor_indices = np.argsort(patch_distances, axis=1)[:, :neighbors + 1]

        return np.asarray(neighbor_indices)
