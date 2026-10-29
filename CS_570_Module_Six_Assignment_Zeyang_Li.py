# Module6.py
import sys # required for command line arguments
from sklearn.neural_network import MLPClassifier  # required for MLPClassifier

def perceptron(weights, inputs):
    if len(weights) != len(inputs):
        raise ValueError("Weights and inputs must have the same length.")

    weighted_sum = sum(w * x for w, x in zip(weights, inputs))

   # print("y =", weighted_sum)

    return 1 if weighted_sum > 0 else 0


def train_perceptron(training_data, learning_rate=0.1, epochs=100):
    if not training_data:
        raise ValueError("Training data cannot be empty.")

    num_inputs = len(training_data[0][0])
    weights = [0.0] * num_inputs

    for epoch in range(epochs):
        mistakes = 0

        for inputs, expected in training_data:
            output = perceptron(weights, inputs)
            error = expected - output

            if error != 0:
                mistakes += 1

                for i in range(num_inputs):
                    weights[i] += learning_rate * error * inputs[i]

        if mistakes == 0:
            break

    return weights


#------------------------------------------------------------------------
# Training Data
#------------------------------------------------------------------------
function1_training_data = [
        ([1, 0, 1], 0), 
        ([0, 1, 1], 0),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function2_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1), 
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function3_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 0)   
    ]


#------------------------------------------------------------------------
# Create a MLP and train to solve function 3 
#------------------------------------------------------------------------
def test_function3_on_mlp(p_verbose, p_solver, p_hidden_layer_nodes, p_activation):
    inputs = [data[0] for data in function3_training_data]
    outputs = [data[1] for data in function3_training_data]
    
    learning_rate = 0.001
    tolerance = 0.0001
    patience = 10

    if p_activation == "logistic" and p_solver in ("sgd", "adam"):
        learning_rate = 0.1 if p_solver == "sgd" else 0.01
        tolerance = 0.00000001
        patience = 100

    mlp = MLPClassifier(
        verbose=False,
        solver=p_solver,
        hidden_layer_sizes=(p_hidden_layer_nodes,),
        activation=p_activation,
        max_iter=10000,
        random_state=42,
        learning_rate_init=learning_rate,
        tol=tolerance,
        n_iter_no_change=patience
    )
    mlp.fit(inputs, outputs) 

    # Get the structure of the MLP model
    if p_verbose:
        print("Number of layers (including input and output):", mlp.n_layers_)
        print("Number of outputs:", mlp.n_outputs_)
        print("Hidden layer sizes:", mlp.hidden_layer_sizes)

        # Access weights and biases
        print("Input-to-hidden weights:\n", mlp.coefs_[0])
        print("Hidden biases:\n", mlp.intercepts_[0])
        print("Hidden-to-output weights:\n", mlp.coefs_[1])
        print("Output bias:\n", mlp.intercepts_[1])

    # Test the trained MLP
    if p_verbose:
        print()
        print("Test the trained MLP for Function 3")
    passed = True
    for inputs, expected in function3_training_data: 
        output = mlp.predict([inputs])[0]
        if output != expected:
            passed = False
        if p_verbose:  
            print(f"Function 3 (MLP): Input {inputs} classified as: {output} (Expected: {expected})")  
    return passed


#------------------------------------------------------------------------
# TODO: Train the perceptron manually from the command line
#------------------------------------------------------------------------
weights = list(map(float, sys.argv[1:]))
if 0: # set to 1 to test weights from command line inputs for function 1
    for inputs, expected in function1_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 2
    for inputs, expected in function2_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 3
    for inputs, expected in function3_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")


#------------------------------------------------------------------------
# TODO: Test your algorithm to train the perceptron
#------------------------------------------------------------------------
if 1: 
    function1_weights = train_perceptron(function1_training_data)
    function2_weights = train_perceptron(function2_training_data)
    function3_weights = train_perceptron(function3_training_data)

    print(f"Trained weights for Function 1: {function1_weights}")
    print(f"Trained weights for Function 2: {function2_weights}")
    print(f"Trained weights for Function 3: {function3_weights}")
    
    print()
    print("Test the trained perceptron")
    for inputs, expected in function1_training_data:
        output = perceptron(function1_weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

    print()
    for inputs, expected in function2_training_data:
        output = perceptron(function2_weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

    print()
    for inputs, expected in function3_training_data:
        output = perceptron(function3_weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")

 
 #------------------------------------------------------------------------
 # Test the MLPClassifier on function 3 by calling test_function3_on_mlp()
 #------------------------------------------------------------------------
# TODO - call test_function3_on_mlp() with different parameters
weights = [-1, -1, 1.5]

perceptron(weights, [0, 0, 1])
perceptron(weights, [1, 1, 1])
perceptron(weights, [1, 0, 1])
perceptron(weights, [0, 1, 1])

solvers = ["lbfgs", "sgd", "adam"]
activations = ["relu", "identity", "logistic", "tanh"]
results = []

for solver in solvers:
    for activation in activations:
        successful_nodes = None

        for nodes in range(1, 11):
            passed = test_function3_on_mlp(
                False, solver, nodes, activation
            )

            if passed:
                successful_nodes = nodes

                print(
                    f"\nPASS: {solver}, {activation}, "
                    f"hidden_layer_sizes=({nodes},)"
                )

                test_function3_on_mlp(
                    True, solver, nodes, activation
                )
                break

        if successful_nodes is None:
            print(
                f"\nFAILED: {solver}, {activation}, "
                "with 1 through 10 hidden nodes"
            )

        results.append((solver, activation, successful_nodes))

print("\nFINAL SUMMARY")
for solver, activation, nodes in results:
    outcome = (
        f"hidden_layer_sizes=({nodes},)"
        if nodes is not None
        else "Failed with 1 through 10 nodes"
    )
    print(f"{solver:<6} {activation:<9} {outcome}")