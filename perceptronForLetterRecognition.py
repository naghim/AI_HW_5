import random
import numpy as np


# Helper function. Activation function.
def step_function(value):
    return 1 if value > 0 else 0


# Trains the network.
def train(data, results):
    N, n = data.shape
    learning_rate = 0.01
    weights_1 = np.random.randn(n, 1)
    weights_2 = np.random.randn(n, 1)
    weights_3 = np.random.randn(n, 1)
    weights_4 = np.random.randn(n, 1)

    error = 1
    error_rate = 0

    while error > error_rate:
        error = 0

        for i in range(N):
            first_neuron = step_function(np.dot(data[i], weights_1))
            second_neuron = step_function(np.dot(data[i], weights_2))
            third_neuron = step_function(np.dot(data[i], weights_3))
            fourth_neuron = step_function(np.dot(data[i], weights_4))

            iter_error1 = results[0][i] - first_neuron
            iter_error2 = results[1][i] - second_neuron
            iter_error3 = results[2][i] - third_neuron
            iter_error4 = results[3][i] - fourth_neuron

            weights_1 += learning_rate * iter_error1 * data[i].reshape(n, 1)
            weights_2 += learning_rate * iter_error2 * data[i].reshape(n, 1)
            weights_3 += learning_rate * iter_error3 * data[i].reshape(n, 1)
            weights_4 += learning_rate * iter_error4 * data[i].reshape(n, 1)

            error += iter_error1 ** 2
            error += iter_error2 ** 2
            error += iter_error3 ** 2
            error += iter_error4 ** 2

    print("Weights:")
    print(weights_1)
    print(weights_2)
    print(weights_3)
    print(weights_4)

    return np.array([weights_1, weights_2, weights_3, weights_4])


# Evaluates the network.
def evaluate(data, weights, results):
    N, n = data.shape
    letter_to_idx = {
        0: 'H',
        1: 'I',
        2: 'O',
        3: 'T',
    }

    for i in range(N):
        first_neuron = step_function(np.dot(data[i], weights[0]))
        second_neuron = step_function(np.dot(data[i], weights[1]))
        third_neuron = step_function(np.dot(data[i], weights[2]))
        fourth_neuron = step_function(np.dot(data[i], weights[3]))

        nn_result = np.array([first_neuron, second_neuron, third_neuron, fourth_neuron])
        expected_output = np.array([results[0][i], results[1][i], results[2][i], results[3][i]])
        if np.array_equal(expected_output, nn_result):
            print("Letter recognised as", letter_to_idx[i])


# Main
def main():
    letter_H = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    letter_I = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    letter_O = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    letter_T = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]

    data = np.array([letter_H, letter_I, letter_O, letter_T])
    results = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Training our network
    weights = train(data, results)

    # Recognising test dataset
    evaluate(data, weights, results)

    print("Testing for modified letters...")
    letter_H_modif = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    letter_I_modif = [1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    letter_O_modif = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    letter_T_modif = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]

    # Recognising incomplete letters
    data = np.array([letter_H_modif, letter_I_modif, letter_O_modif, letter_T_modif])
    evaluate(data, weights, results)


if __name__ == '__main__':
    main()
