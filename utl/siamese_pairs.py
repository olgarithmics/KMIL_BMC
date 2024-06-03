from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf

import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import euclidean_distances

from utl.dataset import get_coordinates, exclude_self
from .data_aug_op import random_flip_img, random_rotate_img


def get_choices( arr, num_choices, valid_range=[-1, np.inf], not_arr=None, replace=False):
    '''
    Select n=num_choices choices from arr, with the following constraints for
    each choice:
        choice > valid_range[0],
        choice < valid_range[1],
        choice not in not_arr
    if replace == True, draw choices with replacement
    if arr is an integer, the pool of choices is interpreted as [0, arr]
    (inclusive)
        * in the implementation, we use an identity function to create the
        identity map arr[i] = i
    '''
    if not_arr is None:
        not_arr = []
    if isinstance(valid_range, int):
        valid_range = [0, valid_range]

    if isinstance(arr, tuple):
        if min(arr[1], valid_range[1]) - max(arr[0], valid_range[0]) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = arr[1]
        arr0 = arr[0]
        arr = defaultdict(lambda: -1)
        get_arr = lambda x: x
        replace = True
    else:

        greater_than = np.array(arr) > valid_range[0]
        less_than = np.array(arr) < valid_range[1]

        if np.sum(np.logical_and(greater_than, less_than)) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")

        n_arr = len(arr)
        arr0 = 0
        arr = np.array(arr, copy=True)
        get_arr = lambda x: arr[x]
    not_arr_set = set(not_arr)

    def get_choice():
        arr_idx = random.randint(arr0, n_arr - 1)
        while get_arr(arr_idx) in not_arr_set:
            arr_idx = random.randint(arr0, n_arr - 1)
        return arr_idx

    if isinstance(not_arr, int):
        not_arr = list(not_arr)
    choices = []
    for _ in range(num_choices):
        arr_idx = get_choice()
        while get_arr(arr_idx) <= valid_range[0] or get_arr(arr_idx) >= valid_range[1]:
            arr_idx = get_choice()
        choices.append(int(get_arr(arr_idx)))
        if not replace:
            arr[arr_idx], arr[n_arr - 1] = arr[n_arr - 1], arr[arr_idx]
            n_arr -= 1
    return choices


def data_aug(img):
    img = random_flip_img(img, horizontal_chance=0.5, vertical_chance=0.5)
    img = random_rotate_img(img)
    return img


def get_siamese_pairs( image_bags,pixel_distance, k,total_pop=1):
    """
    Constuct siamese pairs
    Parameters
    ----------
    image_bags:  a list of lists, each of which contains an np.ndarray of the patches of each image,
    the label of each image and a list of filenames of the patches

    total_pop: int, reffering to the total population of training pairs to be created

    Returns
    -------
    pairs: list of lists, each of which contains pairs of either positve or negative training instances
    labels list of integers, each of which corresponds to the inferred labels of the training pairs
    """
    pairs = []
    labels = []

    pos_indices = [enum for enum, data in enumerate(image_bags) if np.mean(data[1]) == 1]

    neg_indices = [enum for enum, data in enumerate(image_bags) if np.mean(data[1]) == 0]

    for ibag, bag in enumerate(image_bags):


        node_dictionary = []

        filenames = bag[2]

        for ipath, path in enumerate(filenames):
            coords = get_coordinates(path)

            node_dictionary.append((path, coords))

        patch_distances = euclidean_distances([coords for paths, coords in node_dictionary])

        non_zero_elements = np.argwhere(np.sum(patch_distances < pixel_distance, axis=1) > k).flatten()

        if non_zero_elements.shape[0] > total_pop:

            n = np.random.choice(non_zero_elements, len(non_zero_elements), replace=False)

            pos_Idx = np.argsort(patch_distances[n, :], axis=1)[:, :k + 1]

            pos_Idx = exclude_self(pos_Idx)

            k_max = min(pos_Idx.shape[1], k)

            for i, self_id in enumerate(n):
                choices = get_choices(pos_Idx[i, :k_max], k, replace=False)

                new_pos = [[data_aug(bag[0][self_id]), data_aug(bag[0][id])] for id in choices]

                if np.mean(bag[1]) == 1:
                    bag_choices = random.choices(neg_indices, k=k)

                else:
                    bag_choices = random.choices(pos_indices, k=k)

                image_choices = [random.choice(np.arange(image_bags[id][0].shape[0]) - 1) for id in bag_choices]

                new_neg = [[data_aug(bag[0][self_id]), data_aug(image_bags[bag_id][0][image_id])] for
                           bag_id, image_id in zip(bag_choices, image_choices)]

                labels += [1] * len(new_pos) + [0] * len(new_neg)

                pairs += new_pos + new_neg

    return np.array(pairs), np.asarray(labels, dtype=np.float32)



class SiameseGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs,labels,batch_size ,dim,shuffle):
        self.pairs = pairs
        self.labels=labels
        self.batch_size = batch_size
        self.dim=dim
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return   int(np.floor(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        pairs_temp = [self.pairs[k] for k in indexes]
        labels_temp =[self.labels[k] for k in indexes]

        # Generate data
        [x1,x2],y = self.__data_generation(pairs_temp,labels_temp)

        return  [x1,x2],y

    def __data_generation(self, pairs_temp,labels_temp):

        x1= np.empty((self.batch_size, *self.dim))
        x2 = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=np.float32)

        for i,(image_pair, label_pair) in enumerate(zip(pairs_temp,labels_temp)):

            x1[i]= image_pair[0]
            x2[i] = image_pair[1]

            y[i] =label_pair

        return [x1,x2], y

