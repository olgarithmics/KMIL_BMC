import glob
import itertools
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from args import parse_args
from utl.DataGenerator import DataGenerator
from utl.siamese_pairs import get_siamese_pairs,SiameseGenerator
from utl.custom_layers import NeighborAggregator, Graph_Attention, Last_Sigmoid, DistanceLayer, multiply, Mil_Attention
from utl.dataset import Get_train_valid_Path
from utl.metrics import bag_accuracy, bag_loss
from utl.metrics import get_contrastive_loss, siamese_accuracy
from utl.stack_layers import stack_layers, make_layer_list

args = parse_args()

class SiameseNet:
    def __init__(self, arch, input_shape):
        """
        Build the architecture of the siamese net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """

        self.input_shape = input_shape

        self.inputs = {
            'left_input': Input(self.input_shape),
            'right_input': Input(self.input_shape),
        }

        self.layers = []
        self.layers += make_layer_list(arch, 'siamese', args.weight_decay)

        self.outputs = stack_layers(self.inputs, self.layers)

        self.distance = DistanceLayer(output_dim=1)([self.outputs["left_input"], self.outputs["right_input"]])

        self.net = Model(inputs=[self.inputs["left_input"], self.inputs["right_input"]], outputs=[self.distance])

        self.net.compile(optimizer=Adam(lr=args.siam_init_lr, beta_1=0.9, beta_2=0.999),
                             loss=get_contrastive_loss(m_neg=1, m_pos=0.05), metrics=[siamese_accuracy])

    def train(self, pairs_train, check_dir, irun, ifold):
        """
        Train the siamese net

        Parameters
        ----------
        pairs_train : a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        check_dir   : str, specifying the directory where weights of the siamese net are going to be stored
        irun        : int reffering to the id of the experiment
        ifold       : fold reffering to the fold of the k-cross fold validation

        Returns
        -------
        A History object containing a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values

        """


        model_train_set, model_val_set = Get_train_valid_Path(pairs_train, ifold, train_percentage=0.9)

        train_pairs, train_labels = get_siamese_pairs(model_train_set, k=args.k, pixel_distance=args.siam_pixel_dist)
        train_gen = SiameseGenerator(train_pairs,train_labels, batch_size=args.siam_batch_size, dim=args.input_shape, shuffle=True)
        train_pairs, train_labels = get_siamese_pairs(model_val_set, k=args.k, pixel_distance=args.siam_pixel_dist)
        val_gen=  SiameseGenerator(train_pairs,train_labels, batch_size=args.siam_batch_size, dim=args.input_shape, shuffle=False)


        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        filepath = os.path.join(check_dir,
                                "weights-irun:{}-ifold:{}".format(irun, ifold) + ".hdf5")

        checkpoint_fixed_name = ModelCheckpoint(filepath,
                                                monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='auto', save_freq='epoch')

        EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

        callbacks = [checkpoint_fixed_name, EarlyStop]


        self.net.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen) ,
                               epochs=args.siam_epochs, validation_data=val_gen,
                               validation_steps=len(val_gen), callbacks=callbacks)

    def predict(self, x):
        return self.net.predict_on_batch(x)


class GraphAttnet:
    def __init__(self, arch, mode,input_shape):
        """
        Build the architercure of the Graph Att net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        mode            :str, specifying the version of the model (siamese, euclidean)
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """
        self.mode=mode
        self.input_shape = input_shape

        self.inputs = {
            'bag': Input(self.input_shape),
            'adjacency_matrix': Input(shape=(None,), dtype='float32', name='adjacency_matrix'),
        }

        self.layers = []
        self.layers += make_layer_list(arch, 'graph', args.weight_decay)

        self.outputs = stack_layers(self.inputs, self.layers)

        neigh = Graph_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(args.weight_decay), name='neigh',
                                use_gated=args.useGated)(self.outputs["bag"])

        alpha = NeighborAggregator(output_dim=1, name="alpha")([neigh, self.inputs["adjacency_matrix"]])

        # alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(args.weight_decay), name='neigh',
        #                         use_gated=args.useGated)(self.outputs["bag"])

        x_mul = multiply([alpha, self.outputs["bag"]], name="mul")

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)

        self.net = Model(inputs=[self.inputs["bag"], self.inputs["adjacency_matrix"]], outputs=[out, alpha])

        self.net.compile(optimizer=Adam(learning_rate=args.init_lr, beta_1=0.9, beta_2=0.999), loss=bag_loss,
                             metrics=[bag_accuracy])

    @property
    def model(self):
        return self.net

    def load_siamese(self, check_dir, irun, ifold):
        """
        Loads the appropriate siamese model using the information of the fold of k-cross
        fold validation and the id of experiment
        Parameters
        ----------
        check_dir  : directory where the weights of the pretrained siamese network. Weight files are stored in the format:
        weights-irun:d-ifold:d.hdf5
        irun       : int referring to the id of the experiment
        ifold      : int referring to the fold from the k-cross fold validation

        Returns
        -------
        returns  a Keras model instance of the pre-trained siamese net
        """

        def extract_number(f):
            s = re.findall("\d+\.\d+", f)
            return ((s[0]) if s else -1, f)

        file_paths = glob.glob(os.path.join(check_dir, "weights-irun:{}-ifold:{}*.hdf5".format(irun, ifold)))
        file_paths.reverse()
        file_path = (min(file_paths, key=extract_number))

        self.siamese_net = load_model(file_path, custom_objects={'DistanceLayer': DistanceLayer,
                                                                 "contrastive_loss": get_contrastive_loss(),
                                                                 "siamese_accuracy": siamese_accuracy})
        return self.siamese_net

    def train(self, train_set, check_dir, irun, ifold,weight_file=True):
        """
        Train the Graph Att net
        Parameters
        ----------
        train_set       : a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        check_dir       :str, specifying directory where the weights of the siamese net are stored
        irun            :int, id of the experiment
        ifold           :int, fold of the k-corss fold validation
        weight_file     :boolen, specifying whether there is a weightflie or not

        Returns
        -------
        A History object containing  a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.

        """
        model_train_set, model_val_set = Get_train_valid_Path(train_set, ifold, train_percentage=0.9)
        if self.mode=="siamese":
            if weight_file:
                self.siamese_net = self.load_siamese(check_dir, irun, ifold)
            else:

                self.siamese_net = SiameseNet(args.arch, self.input_shape)
                self.siamese_net.train(train_set, check_dir=check_dir, irun=irun, ifold=ifold)

            train_gen = DataGenerator(batch_size=1, data_set=model_train_set, k=args.k, shuffle=True, mode=self.mode,
                                      siamese_model=self.siamese_net)

            val_gen = DataGenerator(batch_size=1, data_set=model_val_set, k=args.k, shuffle=False, mode=self.mode,
                                    siamese_model=self.siamese_net)
        else:
            train_gen = DataGenerator(batch_size=1, data_set=model_train_set, k=args.k, shuffle=True, mode=self.mode)

            val_gen = DataGenerator(batch_size=1, data_set=model_val_set, k=args.k, shuffle=False, mode=self.mode)


        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_name = os.path.join(args.save_dir, args.experiment_name + "_fold_{}.hdf5".format(ifold))


        checkpoint_fixed_name = ModelCheckpoint(model_name,
                                                monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=True, mode='auto', save_freq='epoch')

        EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

        callbacks = [checkpoint_fixed_name, EarlyStop]

        history = self.net.fit(train_gen, steps_per_epoch=len(model_train_set),
                               epochs=args.max_epoch, validation_data=val_gen,
                               validation_steps=len(model_val_set), callbacks=callbacks)

        return history

    def predict(self, test_set,check_dir, irun, ifold,weight_file=True):

        """
        Evaluate the test set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches

        Returns
        -------

        test_loss : float reffering to the test loss
        acc       : float reffering to the test accuracy
        precision : float reffering to the test precision
        recall    : float referring to the test recall
        auc       : float reffering to the test auc
        """

        if self.mode == "siamese":
            if weight_file:
                self.siamese_net = self.load_siamese(check_dir, irun, ifold)
                test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode,
                                         siamese_model=self.siamese_net)
            else:
                test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode)

        test_gen_1, test_gen_2 = itertools.tee(test_gen, 2)

        num_test_batch = len(test_set)

        y_true = [np.mean(label) for image_data, label, filenames in test_set]

        test_loss, acc = self.net.evaluate(test_gen_1, steps=num_test_batch, workers=0, use_multiprocessing=False)

        y_pred= self.net.predict(test_gen_2, steps=num_test_batch, workers=0, use_multiprocessing=False)

        auc = roc_auc_score(y_true, y_pred)
        print("AUC {}".format(auc))

        precision = precision_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("precision {}".format(precision))

        recall = recall_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("recall {}".format(recall))

        return test_loss, acc, auc, precision, recall

    def get_score(self, test_set, check_dir, irun, ifold,weight_file=True):

        """
        Evaluate the test set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches

        Returns
        -------

        test_loss : float reffering to the test loss
        acc       : float reffering to the test accuracy
        precision : float reffering to the test precision
        recall    : float referring to the test recall
        auc       : float reffering to the test auc
        """

        if self.mode == "siamese":
            if weight_file:
                self.siamese_net = self.load_siamese(check_dir, irun, ifold)
                test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode,
                                         siamese_model=self.siamese_net)
            else:
                test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode)
        scores=[]
        for test_idx, label_idx in test_gen:
                y_pred, alpha = self.net.predict_on_batch(test_idx)
                scores.append([alpha])

        return scores

