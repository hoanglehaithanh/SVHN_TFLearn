import numpy as np
import scipy.io as sio


def load_raw_data(train_data_file, test_data_file, load_extra_data, extra_data_file):
    """
    Load RAW Google SVHN Digit Localization from .mat files
    """
    loading_information =  "with Extra" if load_extra_data else "without Extra"
    print("Loading SVHN dataset {}...".format(loading_information))
    raw_train_data = sio.loadmat(train_data_file)
    raw_test_data = sio.loadmat(test_data_file)
    if load_extra_data:
        raw_extra_data = sio.loadmat(extra_data_file)
        print("Train size: {}, Test size: {}, Extra size: {}".format(raw_train_data['X'].shape[3],
                                                                     raw_test_data['X'].shape[3],
                                                                     raw_extra_data['X'].shape[3]))
        return [raw_train_data, raw_test_data, raw_extra_data]
    else:
        print("Train size: {}, Test size: {}".format(raw_train_data['X'].shape[3],
                                                     raw_test_data['X'].shape[3]))
        return [raw_train_data, raw_test_data]

    
def format_data(raw_data, number_of_examples):
    """
    Reshape RAW data to regular shape
    """
    old_shape = raw_data.shape
    new_data = []
    for i in range(number_of_examples):
        new_data.append(raw_data[:,:,:,i])
    new_data = np.asarray(new_data)
    print("Data has been reshaped from {} to {}".format(raw_data.shape, new_data.shape))
    return new_data/255.

def one_hot_encoder(data, number_of_labels):
    """
    One-hot encoder for labels
    """
    data_size = len(data)
    one_hot_matrix = np.zeros(shape=(data_size, number_of_labels))
    for i in range(data_size):
        current_row = np.zeros(shape=(number_of_labels))
        current_number = data[i][0]
        if current_number == 10:
            current_row[0] = 1
        else:
            current_row[current_number] = 1
        one_hot_matrix[i] = current_row
    return one_hot_matrix

def load_svhn_data(train_path ,test_path, extra_path, load_extra, eval_percentage):
    """
    Load SVHN Dataset
    """
    print("Loading SVHN dataset for classification...")
    #Load raw dataset
    if load_extra:
        train,test,extra = load_raw_data(train_path, test_path, load_extra, extra_path)
        train['X'] = np.concatenate((train['X'], extra['X']), axis=3)
        train['y'] = np.concatenate((train['y'], extra['y']), axis=0)
    else:
        train, test = load_raw_data(train_path, test_path, load_extra, extra_path)
    
    #get values and labels
    train_all_values = format_data(train['X'], train['X'].shape[3])
    train_all_labels = one_hot_encoder(train['y'], 10)
    test_values = format_data(test['X'], test['X'].shape[3])
    test_labels = one_hot_encoder(test['y'], 10)

    np.random.seed(41)
    shuffle_indices = np.random.permutation(np.arange(len(train_all_values)))
    train_values_shuffled = train_all_values[shuffle_indices]
    train_labels_shuffled = train_all_labels[shuffle_indices]

    #Seperate into training and eval set
    train_index = -1 * int(eval_percentage * float(len(train_values_shuffled)))
    train_values, eval_values = train_values_shuffled[:train_index], train_values_shuffled[train_index:]
    train_labels, eval_labels = train_labels_shuffled[:train_index], train_labels_shuffled[train_index:]
    print("Train/Eval split: {:d}/{:d}".format(len(train_labels), len(eval_labels)))
    print("Loading data completed")
    return [train_values, train_labels, eval_values, eval_labels, test_values, test_labels]
    
    
    
    
    
    
    
