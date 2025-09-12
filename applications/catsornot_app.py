from neural_network.model import NeuralNetwork
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import dagshub
import os
import joblib
import h5py


dagshub.init(repo_owner='Hemanthanne411', repo_name='Neural-Network-Scratch', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Hemanthanne411/Neural-Network-Scratch.mlflow")
def load_data():
    train_dataset = h5py.File("./data/raw/cats/train_catvsnoncat.h5", "r")
    test_dataset = h5py.File("./data/raw/cats/test_catvsnoncat.h5", "r")
    
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # (m_train, 64, 64, 3)
    train_set_y = np.array(train_dataset["train_set_y"][:])      # (m_train,)
    
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])    # (m_test, 64, 64, 3)
    test_set_y = np.array(test_dataset["test_set_y"][:])         # (m_test,)

    classes = np.array(test_dataset["list_classes"][:])          # [b'non-cat' b'cat']
    
    train_set_y = train_set_y.reshape(1, -1)  # shape (1, m_train)
    test_set_y = test_set_y.reshape(1, -1)    # shape (1, m_test)



    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

def preprocess_data(train_x_orig, test_x_orig):
    # Flatten each image
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # shape (12288, m_train)
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T     # shape (12288, m_test)

    # Normalize
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    return train_x, test_x

# ========== Main Entry Point ==========
def main():
    np.random.seed(1)
    #  Load the data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    np.random.seed(1)
    #  Preprocess the data
    train_x, test_x = preprocess_data(train_x_orig, test_x_orig)

    #  Define your NN architecture
    n_h = [20, 7, 5, 1]  # example 4-layer model for binary classification
    activations =['relu', 'sigmoid']
    iterations = 2900
    learning_rate = 0.0075
    

    mlflow.set_experiment("NN_Application-2, Cats or Not")
    artifacts_dir = "artifacts/catsornot"
    os.makedirs(artifacts_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("Hidden_layers", n_h)
        mlflow.log_param("Activations", activations)
        mlflow.log_param("Iterations", iterations)
        mlflow.log_param("Learning_rate", learning_rate)


        jod_cat = NeuralNetwork(n_h, activations, iterations = iterations, learning_rate = learning_rate, print_cost = True)
        jod_cat.train_NN(train_x, train_y)

        #  Predict
        train_predictions = jod_cat.predict_y(train_x, jod_cat.parameters, activations=activations)
        train_accuracy = jod_cat.accuracy(train_y, train_predictions)
        # print(f"The accuracy of the NN in Training : {train_accuracy:.4f}\n")

        
        test_predictions = jod_cat.predict_y(test_x, jod_cat.parameters, activations=activations)
        test_accuracy = jod_cat.accuracy(test_y, test_predictions)
        # print(f"The accuracy of the NN in Test : {test_accuracy:.4f}")
        
        mlflow.log_metric("Training Accuracy", train_accuracy)
        mlflow.log_metric("Test Accuracy", test_accuracy) 
        artifact_dir = "artifacts/images"
        os.makedirs(artifact_dir, exist_ok=True)

        # Picking a random test example
        idx = np.random.randint(0, test_x.shape[1])
        sample_img = test_x[:, idx].reshape(64, 64, 3)  # adjust shape
        sample_img = (sample_img * 255).astype(np.uint8)  # rescale if normalized
        pred_label = test_predictions[0, idx]
        true_label = test_y[0, idx]

        # Saving the image in the artifact folder
        image_path = os.path.join(artifact_dir, "sample_prediction.png")
        plt.imshow(sample_img)
        plt.title(f"Pred: {pred_label}, True: {true_label}")
        plt.axis("off")
        plt.savefig(image_path)
        plt.close()

        # Log artifact to MLflow
        mlflow.log_artifact(image_path, artifact_path="images") 
        model_path = os.path.join(artifacts_dir, "catsornot_model.pkl")
        joblib.dump(jod_cat, model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
   



