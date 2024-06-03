
## Not seeing the trees for the forest. The impact of neighbours on graph-based configurations in histopathology
The code can be used to run experiments on the the COLON cancer and UCSB datasets. 
To test the siamese version of our model the weights are stored 
in the root directory under the name colon_weights and cell_weights, respectively. 
One can select between two different running modes: euclidean and siamese one. 
Using the argument k, the number of neighbors can be specified. 
Because the running time grows proportionally to the number of K, it is advised to tested it for K<10.

## Installation
It is advised to use a conda installation, as it is more straightforward.
 - python=3.7
 - tensorflow-gpu=2.1
 - scikit-learn 
 - Pillow
 - opencv
 

## How to use



The script uses a number of command line arguments to work the most important of which are the following :

* k- the number of neighbors taken into account
* data_path - the directory where the images are stored
* siamese_weights_path - the directory where the weights of the pre-trained siamese network are stored
* extention - the file extention of the image patches
* mode - (euclidean or siamese) version of the model 
* weight_file - define whether there is, or not a file of weights
* experiment_name - the name of the experiment to run
* input_shape - (w,h,d) of the image patches
* folds - the number of folds to be used in the k-cross fold validation: 10 for the COLON cancer dataset, 4 for the UCSB


For the COLON cancer dataset the script can be executed from the command line typing the command:
```sh
cd <root_directory>
$ python run.py --experiment_name test --mode siamese --k 1 --weight_file  --data_path colon_cancer_patches --input_shape 27 27 3 --extention bmp --siamese_weights_path weights --data colon
```
or without weights:
```sh
cd <root_directory>
$ python run.py --experiment_name test --mode siamese --k 1 --data_path colon_cancer_patches --input_shape 27 27 3 --extention bmp --siamese_weights_path test_weights --siam_pixel_distance 20 --data colon
```
or the following command in case the euclidean mode is selected:
```sh
cd <root_directory>
$ python run.py --experiment_name test --mode euclidean -k 1 --data_path colon_cancer_patches --input_shape 27 27 3 --extention bmp --data colon
```


For the UCSB cancer dataset the script can be executed from the command line typing the command:
```sh
cd <root_directory>
$  python run.py --experiment_name test --mode siamese --k 1 --weight_file --data_path cells --extention png --input_shape 32 32 3 --siamese_weights_path cell_weights --folds 4 --data ucsb
```
or without weights:
```sh
cd <root_directory>
$ python run.py --experiment_name test --mode siamese --k 1 --data_path cells --input_shape 32 32 3 --extention png --siamese_weights_path test_weights --folds 4 --siam_pixel_distance 25 --data ucsb
```
or the following command in case the euclidean mode is selected:
```sh
cd <root_directory>
$  python run.py --experiment_name test --mode euclidean --k 1 --data_path cells --extention png --input_shape 32 32 3 --folds 4 --data ucsb
```

