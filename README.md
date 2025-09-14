# Neural Network from Scratch
![Python](https://img.shields.io/badge/python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.27-orange)
![License](https://img.shields.io/badge/license-MIT-green)
## Project Overview
This project implements a **Neural Network from scratch in Python** using **NumPy**, without relying on external deep learning frameworks. It demonstrates the key principles of neural networks, including forward propagation, backpropagation, and gradient descent. The project also integrates **MLflow** and **DagsHub** for experiment tracking, model versioning, and reproducibility.

---

## Features
- Customizable fully connected network architecture with any number of hidden layers  
- Supports **binary and categorical classification**  
- Multiple activation functions: `ReLU`,`Tanh`, `Sigmoid`, `Softmax`  
- Computes **accuracy** for both binary and categorical outputs  

---

## Neural Network Implementation Details

This project implements a **fully connected neural network from scratch** using **NumPy**, covering the full workflow of training and inference. Key aspects include:

1. **Parameter Initialization**  
   - Weights are initialized with small random values to break symmetry.  
   - Biases are initialized to zeros.  
   - This ensures stable forward and backward propagation.

2. **Forward Propagation**  
   - Computes the linear combination `Z = W·A_prev + b` for each layer.  
   - Applies the chosen **activation function** (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`) to produce activations for the next layer.  
   - Stores caches (`A_prev`, `W`, `b`, `Z`) for backpropagation.

3. **Cost Function**  
   - For **binary classification**, binary cross-entropy is used.  
   - For **multi-class classification**, categorical cross-entropy with softmax output is applied.  
   - Cost is computed over the entire dataset to guide parameter updates.

4. **Backpropagation**  
   - Computes gradients of the cost with respect to weights and biases for all layers.  
   - Uses the chain rule to propagate the error backward through the network.  
   - Special handling for output layer:  
     - Binary: derivative from sigmoid + BCE  
     - Categorical: derivative from softmax + CCE  

5. **Gradient Descent**  
   - Updates parameters using `W = W - learning_rate*dW` and `b = b - learning_rate*db`.  
   - Learning rate controls the step size during optimization.

6. **Prediction**  
   - Forward pass is performed with trained parameters.  
   - Output layer activation decides the class:  
     - Binary → `AL > 0.5` → 0 or 1  
     - Multi-class → `argmax(AL)` → predicted class index  

7. **Accuracy**  
   - Compares predicted labels against true labels for both binary and categorical outputs.  


---

## Applications
The modular Neural Network implemented in this project is designed to handle **two classification tasks** and has been applied to:
1. **Binary Classification**: Cat vs. Not Cat image classification  
    - Test Accuracy: 82%
2. **Digit Classification**: MNIST handwritten digit recognition  
    - Test Accuracy: 93%

---

## Experiment Tracking
**Experiment Tracking**: Tracked model performance, artifacts, and trained models using MLflow and DagsHub
- **MLflow**: Logs metrics (accuracy), trained models, and sample prediction images  
  - [View MLflow experiments](https://dagshub.com/Hemanthanne411/Neural-Network-Scratch.mlflow)  
- **DagsHub**: Version control for code, models, and datasets  
  - [Project on DagsHub](https://dagshub.com/Hemanthanne411/Neural-Network-Scratch)  

*Note*: Artifacts such as sample predictions and trained models are stored in MLflow; DagsHub primarily tracks code, data, and model files via Git/DVC.

---

## Using the Neural Network


- Clone the [repository](https://github.com/Hemanthanne411/Neural-Network-Scratch.git) and install the requirements inside your venv.
 

### Functions
Import the NeuralNetwork class from `model.py`:

- Initialize the neural network with architecture parameters
- nn = `NeuralNetwork(n_h, activations, iterations=800, learning_rate=0.01, print_cost=False, seed=1)` 
- nn.`train_NN(X, Y)` - Train the network on input data and labels
- nn.`predict_y(X, nn.parameters, activations)` - Make predictions using trained parameters
- nn.`accuracy(true_y, predicted_y)` - Compute accuracy of predictions against true labels

### Architecture Parameters
- **`n_h` / `hidden_layers`**: List of neurons in each hidden layer (input layer is excluded)  
  **Example**: `[20, 7, 5, 1]` → four layers after input with 20, 7, 5, 1 neurons

- **`activations`**: List with two elements
  - `activations[0]` → activation function for all hidden layers
  - `activations[1]` → activation function for the output layer


---



## References & Acknowledgements

This project was inspired by foundational concepts in deep learning and neural networks. Key references include:

- **Andrew Ng’s Deep Learning Specialization** (Coursera) – for understanding neural network theory, forward/backward propagation, and optimization techniques  
  - [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)  

- **NumPy Documentation** – for numerical computations and matrix operations  
  - [NumPy](https://numpy.org/doc/)  

- **MLflow & DagsHub Documentation** – for experiment tracking, model versioning, and reproducibility  
  - [MLflow](https://mlflow.org/docs/latest/index.html)  
  - [DagsHub](https://dagshub.com/)  

This implementation demonstrates a **hands-on understanding of neural networks from scratch**, combining theory with practical experiment tracking and model versioning to ensure reproducibility.





