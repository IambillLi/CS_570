# 
#
#

# Use verbose to control the output of the perceptron and test functions
# verbose = "perceptron"  outputs the perceptron calculations
# verbose = "test"  # outputs the test results
verbose = "" # verbose disabled 

import numpy as np
import random

test_function_training_data = [
        ([1, 1, 1], 1),
        ([1, 0, 1], 1),
        ([0, 1, 1], 0),
        ([0, 0, 1], 0),
        ([1, 1, 1], 1),
        ([1, 0, 1], 1),
        ([0, 1, 1], 0),
        ([0, 0, 1], 0)
    ]

xor_training_data = [
        ([0, 0, 1], 0),
        ([0, 1, 1], 1),
        ([1, 0, 1], 1),
        ([1, 1, 1], 0)
    ]


#------------------------------------------------------------------------
# TODO: Code a linear activation function by completing the TODOs
# Using a linear activation function, the output of the perceptron will not
# be limited to binary values (0 or 1), but will instead be a continuous value.
# The perceptron will get close to 1 and close to 0, but not exactly 1 or 0.
#------------------------------------------------------------------------
def activation_function(weighted_sum):
     return weighted_sum


#------------------------------------------------------------------------
# TODO: Code a perceptron by completing the TODOs
#------------------------------------------------------------------------
def perceptron(weights, inputs): 
    """
    Computes the output of a perceptron given weights and inputs.
    
    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :return: Output of the perceptron (1 or 0).
    """
    if len(weights) != len(inputs):
        raise ValueError("Weights and inputs must have the same length.")
    
    weighted_sum = sum(weight * value for weight, value in zip(weights, inputs))
    activate = activation_function(weighted_sum)   
    if "perceptron" in verbose:
        print(f"Perceptron: Inputs: {inputs}, Weights: {weights}, Weighted sum: {weighted_sum}, Activate: {activate}")    
    return activate

#------------------------------------------------------------------------
# TODO: Calculate the Squared Error part of the Mean Squared Error (MSE) by completing the TODOs
#------------------------------------------------------------------------
def squared_error(expected, actual):
    se = (expected - actual) ** 2
    return se


#------------------------------------------------------------------------
# TODO: Test the weights on a single sample of the training set by completing the TODOs
#------------------------------------------------------------------------
def test_perceptron(weights, inputs, expected):
    """
    Tests the perceptron with given weights and inputs, returning the Mean Squared Error (MSE).

    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :param expected: Expected output for the given inputs.
    :return: Squared Error (SE) between expected and actual output.
    """
    actual = perceptron(weights, inputs)
 
    se = squared_error(expected, actual)
    if "test" in verbose:
        print(f"Test: Expected: {expected}, Actual: {actual}, SE: {se}")
    return se


#------------------------------------------------------------------------
# TODO: Test the weights across the entire training set by completing the TODOs
#------------------------------------------------------------------------
def batch_test_perceptron(weights, training_data = test_function_training_data):
    results = []
    for inputs, expected in training_data:
        squared_error = test_perceptron(weights, inputs, expected)
        results.append((inputs, expected, squared_error))

    batch_mse = sum(error for _, _, error in results) / len(results)
    return batch_mse


#------------------------------------------------------------------------
# Use this function to experiment with brute force training
#------------------------------------------------------------------------
def find_weights_using_brute_force(weight_range, learning_rate, training_data = test_function_training_data):
    epochs = 0
    for w1 in np.arange(weight_range[0], weight_range[1], learning_rate):
        for w2 in np.arange(weight_range[0], weight_range[1], learning_rate):
            for w3 in np.arange(weight_range[0], weight_range[1], learning_rate):
                epochs += 1
                weights = [w1, w2, w3]
                mse = batch_test_perceptron(weights, training_data=training_data)
                if "brute_force" in verbose:
                    print(f"Epoch {epochs}: Weights: {weights}, MSE: {mse}")
                if mse < 0.005:
                    print(f"Found weights: {weights} after {epochs} epochs with MSE: {mse}")
                    return weights, epochs
    print(f"No suitable weights found after {epochs} epochs.")
    return None, epochs


#------------------------------------------------------------------------
# Use this function to experiment with random search for weights
#------------------------------------------------------------------------
def find_weights_using_random_search(max_epochs=10000):
    epochs = 0
    num_inputs = 3  # Number of inputs to the perceptron
    while True:
        epochs += 1
        weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        mse = batch_test_perceptron(weights)
        if "random_search" in verbose:
            print(f"Epoch {epochs}: Weights: {weights}, MSE: {mse}")
        if mse < 0.005:  # If MSE is low enough, we consider it a success
            print(f"Found weights after {epochs} epochs: {weights}")
            return weights
        if epochs >= max_epochs:
            print(f"Reached maximum epochs ({max_epochs}) without finding suitable weights.")
            return None   
    

#------------------------------------------------------------------------
# no derivative batch gradient estimation
# are we walking uphill or downhill
# adjust the offset slightly up and down, compute the loss, 
# and see which direction (up or down) reduces the loss.
#------------------------------------------------------------------------
def batch_gradient_estimation(weights, training_rate):
    """
    Estimates the gradient for each weight by adjusting the weights slightly up and down.
    
    :param weights: List of weights for the perceptron.
    :param training_data: Training data to test the perceptron.
    :param training_rate: Step size for adjusting weights.
    :return: List of gradients for each weight.
    """
    gradients = []
    if "gradient1" in verbose:
        print(f"Calculating gradients for weights: {weights} with training rate: {training_rate}")

    for i in range(len(weights)):
        # Adjust the weight slightly up and down
        weights_up = weights[:]
        weights_down = weights[:]
        weights_up[i] += training_rate
        weights_down[i] -= training_rate
        
        mse_up = batch_test_perceptron(weights_up)
        mse_down = batch_test_perceptron(weights_down)

        gradient = (mse_up - mse_down) / (2 * training_rate)  # Estimate the gradient    
        gradients.append(gradient)       

        if "gradient2" in verbose:
            print(f"Weight {i}: Up MSE: {mse_up}, Down MSE: {mse_down}, Gradient: {gradient}")
    
    return gradients




#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# Test the perceptron and acivation function with some know good weights
if 0: # Set to one to test your perceptron and activation function
    verbose = "test"  # Enable verbose output for perceptron and test functions
    weights = [0.8999999999999995, -2.220446049250313e-16, 0.09999999999999964]
    batch_test_perceptron(weights)
    print()
    weights = [0.9498386580831919, -0.04052028630978577, 0.004611405977543281]
    batch_test_perceptron(weights)
    print()
    weights = [0.92758834748299, 0.052563750418611166, -0.002519118520321495]
    batch_test_perceptron(weights)
    print()

#------------------------------------------------------------------------
# TODO: Experiment with step size and brute force search for weights 
# What is the impact of training size when searching for weights?
#------------------------------------------------------------------------
print()
print("Experiment with step size and brute force search for weights")
find_weights_using_brute_force((-1.0, 1.1), 0.5)
find_weights_using_brute_force((-1.0, 1.1), 0.25)
find_weights_using_brute_force((-1.0, 1.1), 0.1)
print("\nBrute-force search on XOR")
find_weights_using_brute_force((-1.0, 1.1), 0.1, xor_training_data)
#------------------------------------------------------------------------
# TODO: Can you get the perceptron to train if we just try random weights?
#------------------------------------------------------------------------
print()
print("Experiment with random weights")
random.seed(570)

for run in range(1, 11):
    print(f"\nRandom search run {run}")
    weights = find_weights_using_random_search()

    if weights is not None:
        mse = batch_test_perceptron(weights)
        print(f"MSE: {mse:.8f}")


#------------------------------------------------------------------------
# TODO: Calculate the weights for the perceptron using Batch Gradient Descent
# Batch Gradient Descent: Updates weights after computing the gradient over the entire training set.
#------------------------------------------------------------------------
print()
print("Calculate the weights for the perceptron using Batch Gradient Descent")

verbose = ""  # Enable verbose output for perceptron, test, and gradient calculations

initialization = random.Random(42)
weights = [initialization.uniform(-1, 1) for _ in range(3)]

epochs = 0
training_rate = 0.5  # Learning rate controls the size of weight updates
while True:
    epochs += 1

    # Compute the Loss: Use your model with current weights to predict outputs for your data.
    # Compare predictions to actual results using a loss function 
    # (e.g., mean squared error, which measures the average squared difference between predictions and actual values). 
    # With each iteration, mse_of_all_samples_using_current_weights should decrease
    mse_of_all_samples_using_current_weights = batch_test_perceptron(weights)
    print(f"Epoch {epochs}: Weights: {weights}, MSE: {mse_of_all_samples_using_current_weights}")

    if mse_of_all_samples_using_current_weights < 0.005 or epochs >= 1000:
        break

    gradients = batch_gradient_estimation(weights, 1e-5)

    # Update weights: weight = weight - learning_rate * gradient 
    for i in range(len(weights)):
        weights[i] -= training_rate * gradients[i]

# The final parameters are your optimized values.
print(f"Current Weights: {weights}, MSE: {mse_of_all_samples_using_current_weights}, training epochs: {epochs}")





