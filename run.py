import time
import numpy as np
from args import parse_args, set_seed
from utl.dataset import load_dataset, generate_batch
from utl.networks import GraphAttnet
import os
def model_training(input_dim, dataset, irun, ifold,mode, data_name):
    train_bags = dataset['train']
    test_bags = dataset['test']

    train_set = generate_batch(train_bags,data_name)
    test_set = generate_batch(test_bags,data_name)

    net = GraphAttnet(args.arch, mode,input_dim)

    t1 = time.time()

    #net.train(train_set, check_dir=args.siamese_path, weight_file=args.weight_file, irun=irun, ifold=ifold)

    model_name = os.path.join(args.save_dir, args.experiment_name + "_fold_{}.hdf5".format(ifold))
    net.model.load_weights(model_name)

    test_loss, test_acc, auc, precision, recall = net.predict(test_set,check_dir=args.siamese_path, weight_file=args.weight_file, irun=irun, ifold=ifold)

    t2 = time.time()
    print('run time:', (t2 - t1) / 60.0, 'min')
    print('test_acc={:.3f}'.format(test_acc))

    return test_loss, test_acc, recall, precision, auc


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(args.seed_value)

    acc = np.zeros((args.run, args.n_folds), dtype=float)
    precision = np.zeros((args.run, args.n_folds), dtype=float)
    recall = np.zeros((args.run, args.n_folds), dtype=float)
    auc = np.zeros((args.run, args.n_folds), dtype=float)
    test_loss = np.zeros((args.run, args.n_folds), dtype=float)


    for irun in range(args.run):
        datasets= load_dataset(dataset_path=args.data_path, n_folds=args.n_folds, rand_state=irun)

        for ifold in range(args.n_folds):
            print('run=', irun, '  fold=', ifold)

            test_loss[irun][ifold],acc[irun][ifold], recall[irun][ifold], precision[irun][ifold], auc[irun][ifold] = \
                        model_training(tuple(args.input_shape), datasets[ifold], irun=irun, ifold=ifold, mode=args.mode, data_name=args.data)

    test_loss_mean = np.mean(test_loss)
    test_loss_std = np.std(test_loss)

    np.save('ucsb_errors/test_acc_mean_k_{}.npy'.format(args.k), np.mean(acc))
    np.save('ucsb_errors/test_acc_std_k_{}.npy'.format(args.k), np.std(acc))

    np.save('ucsb_errors/test_aυc_mean_k_{}.npy'.format(args.k), np.mean(auc))
    np.save('ucsb_errors/test_aυc_std_k_{}.npy'.format(args.k), np.std(auc))

    recall = np.mean(recall)
    precision = np.mean(precision)
    fscore=2 * (precision * recall) / (precision + recall)

    np.save('ucsb_errors/test_fscore_mean_k_{}.npy'.format(args.k), np.mean(fscore))
    np.save('ucsb_errors/test_fscore_std_k_{}.npy'.format(args.k), np.std(fscore))

    print("number of neighbors used {}:".format(args.k))
    print('mean loss = ', np.mean(test_loss))
    print('std = ', np.std(test_loss))
    print('mean accuracy = ', np.mean(acc))
    print('std = ', np.std(acc))
    print('mean precision = ', np.mean(precision))
    print('std = ', np.std(precision))
    print('mean recall = ', np.mean(recall))
    print('std = ', np.std(recall))
    print('mean auc = ', np.mean(auc))
    print('std = ', np.std(auc))

