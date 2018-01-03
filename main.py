#Python 2 compability 
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#Library
import numpy as np
import tflearn
import argparse
import time
from pathlib import Path
import tensorflow as tf

#Import data-handler
import data_helper

#Import models
from network import CNN_1, CNN_2, DNN

#default parameters for argparse
default_params = {
    "learning_rate": 0.001,
    "num_epochs": 25,
    "batch_size": 128,
    "train_data_file": "./assignment_httm_data/train_32x32.mat",
    "test_data_file": "./assignment_httm_data/test_32x32.mat",
    "extra_data_file": "./assignment_httm_data/extra_32x32.mat",
    "load_extra": False,
    "model": "CNN1",
    "validation_percentage": 0.1,
    "data_shuffle": True,
    "preprocess": False,
    "mode": 'train',
    "runs_name": None,
    "tensorboard_dir": '~/tensorboard_runs'
}

def get_network(network_name ,args):
    if (network_name == "CNN1"):
        return CNN_1(args.learning_rate).get_model()
    elif (network_name == "DNN"):
        return DNN(args.learning_rate).get_model()
    elif (network_name == "CNN2"):
         return CNN_2(args.learning_rate).get_model()
    else:
        return None

#Main entry point
def main(args):
    """
    Main entry point for the program
    """
    
    train_X, train_Y, eval_X, eval_Y, test_X, test_Y = data_helper.load_svhn_data(train_path = args.train_path,
                                                                      test_path = args.test_path,
                                                                      extra_path = args.extra_path,
                                                                      load_extra = args.load_extra,
                                                                      eval_percentage = args.validation_percentage
                                                                     )

    run_id = 'svhn_runs_{}'.format(args.name)
    
    home = str(Path.home())
    tensorboard_dir = args.tensorboard_dir.replace('~',home)

    
    if (args.mode == 'train'):
        network = get_network(args.model, args)
        model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir=tensorboard_dir)
        #Training
        model.fit(train_X, train_Y, 
                  n_epoch= args.num_epochs, 
                  shuffle=args.data_shuffle, 
                  validation_set=(eval_X, eval_Y),
                  show_metric=True, 
                  batch_size=args.batch_size, 
                  run_id= run_id)
        print('Training complete')
        model_save_path = "./Model/{}.tfl".format(run_id)
        model.save(model_save_path)
        print("Model saved at {}".format(model_save_path))
        file = open("./Model/lastest_run", 'w')
        file.write(run_id+"\n")
        file.write(args.model)
        file.close()
    else:
        #Evaluation(Test)
        runs_name = args.runs_name
        if (not runs_name):
            try:
                file = open("./Model/lastest_run", 'r')
            except e:
                print('Lastest run not found!')
                return
            parameters = file.readlines()
            runs_name = parameters[0].replace('\n','')
            model_name = parameters[1]
            print(runs_name, model_name)
            network = get_network(model_name, args)
            file.close()
        print('Testing...')
        
        model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir=tensorboard_dir)        
        runs_load_path = "./Model/{}.tfl".format(runs_name)
        print('Model loaded at {}'.format(runs_load_path))
        model.load(runs_load_path)
        score = model.evaluate(test_X, test_Y)
        print('Test accuracy: {:0.4f}'.format(score[0]))
    
#Add Arguments parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SVHN Data Classifier Command Line")
    
    parser.add_argument(
      '--mode',
      choices = ['train','eval'],
      default=default_params['mode'],
      help='Mode to run (Train/Eval) (default = {}).'.format(default_params['mode'])
      )
              
    parser.add_argument(
      '--runs_name',
      default=default_params['runs_name'],
      help='Run to be evaluation in Eval mode (default: Lastest run)'
      )
    
    parser.add_argument(
      '--learning_rate',
      type=float,
      default=default_params['learning_rate'],
      help='Initial learning rate (default = {}).'.format(default_params['learning_rate'])
      )
    
    parser.add_argument(
      '--num_epochs',
      type= int,
      default =default_params['num_epochs'],
      help='Number of training epochs (default = {}).'.format(default_params['num_epochs'])
      )
    
    parser.add_argument(
      '--model',
        default=default_params['model'],
        choices= ['CNN1','CNN2','DNN'],   
        help='Model to use (CNN1 - Simple/ CNN2 - Complex, DNN) (default = {}).'.format(default_params['model'])
      )
    
    parser.add_argument(
      '--batch_size',
      type= int,
      default=default_params['batch_size'],
      help='Batch size (default = {}).'.format(default_params['batch_size'])
      )
    
    parser.add_argument(
      '--train_path',
      default =default_params['train_data_file'],
      help='Path to training dataset (default = {}).'.format(default_params['train_data_file'])
      )
    
    parser.add_argument(
      '--test_path',
        default=default_params['test_data_file'],
        help='Path to test dataset (default = {}).'.format(default_params['test_data_file'])
      )
    
    parser.add_argument(
      '--extra_path',
      default=default_params['extra_data_file'],
      help='Path to extra test dataset (default = {}).'.format(default_params['extra_data_file'])
      )
    
    parser.add_argument(
      '--load_extra',
      type= bool,
      default =default_params['load_extra'],
      help='Use the extra dataset (default = {}).'.format(default_params['load_extra'])
      )
    
    parser.add_argument(
      '--validation_percentage',
        default=default_params['validation_percentage'],
        type = float,
        help='Validation percentage from original training set (default = {}).'.format(default_params['validation_percentage'])
      )
    
    parser.add_argument(
      '--data_shuffle',
      type= bool,
      default=default_params['data_shuffle'],
      help='Shuffle the data when training (default = {}).'.format(default_params['data_shuffle'])
      )
    
    parser.add_argument(
      '--name',
      default= time.strftime("%Y%m%d%H%M%S", time.localtime()),
      help='Run name (default = current datetime)'
      )
    
    parser.add_argument(
      '--tensorboard_dir',
      default=default_params['tensorboard_dir'],
      help='Tensorboard log directory (default = {}).'.format(default_params['tensorboard_dir'])
      )
    
    
    args = parser.parse_args()
    main(args)