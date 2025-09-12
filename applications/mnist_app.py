from neural_network.model import NeuralNetwork
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns


def encode_y(Y):
    enc_y = np.zeros((Y.max() + 1, Y.size))
    # Placing 1 according to the label (Y value, index of size m)
    enc_y[Y, np.arange(Y.size)] = 1
    return enc_y


def main():
    df_train = pd.read_csv('./data/raw/mnist/mnist_train.csv')
    df_test = pd.read_csv('./data/raw/mnist/mnist_test.csv')
    train_data = np.array(df_train)
    test_data = np.array(df_test)

    n_h = [32,16,10]
    activations = ['relu', 'softmax']
    # m training examples and n features
    train_data.shape
    m, n = train_data.shape
    X_train = (train_data[:, 1:].T)/255
    y_train = train_data[:, 0]
    # X_train.shape  #((784, 60000)
    # y_train.shape  (60000,)
    
    X_test = (test_data[:, 1:].T)/255
    y_test = test_data[:, 0]

 
    y_train_enc = encode_y(y_train)

    jod = NeuralNetwork(n_h, activations, iterations = 800, learning_rate = 0.1, print_cost = True)
    params = jod.train_NN(X_train, y_train_enc)

    train_predictions = jod.predict_y(X_train, params, activations=activations)
    train_accuracy = jod.accuracy(y_train, train_predictions)
    print(f"The accuracy of the NN in Training : {train_accuracy:.4f}\n")

    
    test_predictions = jod.predict_y(X_test, params, activations=activations)
    test_accuracy = jod.accuracy(y_test, test_predictions)
    print(f"The accuracy of the NN in Test : {test_accuracy:.4f}")
    

if __name__ == '__main__':
    main()




    

    