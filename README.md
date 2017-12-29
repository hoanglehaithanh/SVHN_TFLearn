# SVHN Classifier with TFLearn

### 1. Description
This project is a classifier system for **The Street View House Numbers** (SVHN) Dataset

*"SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images."*

More information about this dataset can be found [here](http://ufldl.stanford.edu/housenumbers/)

The dataset used here is Format 2.

The aim of the project is a simple classifier using deep learning techniques with good evaluation result (>90%).

### 2. Prerequisites:
- Python 3.5
- [TFLearn](http://www.tflearn.org) 0.3
- [TensorFlow](http://www.tensorflow.org) 1.0+
- NumPy
- SciPy

### 3. System structure
##### a. There are 3 python files as 3 modules:
- [main.py](main.py) : The start point of the program.
- [data_helper.py](data_helper.py) : Data module that does preprocessing actions: Load SVHN dataset; Seperate dataset to training set, validation set, test set; Encode labels to one-hot format,..
- [network.py](network.py) : Model module. Define deeplearning networks' architecture.

Other files:
- [RAW dataset](assignment_httm_data) : Dataset in MATLAB .mat format
- [README.md](README.md) : This file, obiviously :)

##### b. Models
1. Basic Convolutional Neural Network
2. Highway Convolutional Neural Network
3. Deep neural network (Multilayer Perceptron)

### 4. Installation
#### There are two ways of running the program:
Use default arguments:
```sh
$python3 main.py
```
Use custom arguments: 
```sh
$python3 main.py [--mode MODE] 
                 [--model_name MODEL_NAME]
                 [--learning_rate LEARNING_RATE] 
                 [--num_epochs NUM_EPOCHS]
                 [--model MODEL] 
                 [--batch_size BATCH_SIZE]
                 [--train_path TRAIN_PATH] 
                 [--test_path TEST_PATH]
                 [--extra_path EXTRA_PATH] 
                 [--load_extra LOAD_EXTRA]
                 [--validation_percentage VALIDATIONL_PERCENTAGE]
                 [--data_shuffle DATA_SHUFFLE] 
                 [--name NAME]
                 [--tensorboard_dir TENSORBOARD_DIR]
```
Usage:
```--mode```Mode to run (Train/Eval) (default = train).
```--model_name```: Model to be evaluation in Eval mode (default = None).
```--learning_rate```: Initial learning rate (default = 0.001).
```--num_epochs``` : Number of training epochs (default = 25).
```--model```Model to use (CNN1 - Simple/ CNN2 - Complex, DNN) (default = CNN1).
``` --batch_size``` : Batch size (default = 128).
```--train_path``` : Path to training dataset (default = ```./assignment_httm_data/train_32x32.mat```).
```--test_path``` : Path to test dataset (default = ```./assignment_httm_data/test_32x32.mat```).
```--extra_path``` : Path to extra test dataset (default = ```./assignment_httm_data/extra_32x32.mat```).
```--load_extra``` : Use the extra dataset (default = False).
```--validation_percentage``` : Validation percentage from original training set (default = 0.1).
```--data_shuffle``` : Shuffle the data when training (default = True).
```--name``` : Run name (default = current datetime)
```--tensorboard_dir``` :Tensorboard log directory (default = ```~/tensorboard_runs```).

### 5. Result
##### Database:
##### Result of the assignment (default parameters):
Updating...
### 6. Future development
- More models (RNN, SVM,...)
- Cleaner code
- More specific structure
- PyTorch?
### 7. License
MIT License

© 2017 Hoàng Lê Hải Thanh (Thanh Hoang Le Hai) aka GhostBB



