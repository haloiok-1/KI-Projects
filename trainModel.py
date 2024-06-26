import numpy as np
import pickle


# Aktivierungsfunktion und deren Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Initialisierung der Netzwerkparameter
input_layer_neurons = 2  # Anzahl der Eingabe-Neuronen
hidden_layer_neurons = 2  # Anzahl der Neuronen im versteckten Layer
output_neurons = 2  # Anzahl der Ausgabe-Neuronen (zwei Ausgabeneuronen)

# Gewichte und Biases initialisieren
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))


# Vorwärtspropagation
def forward_propagation(X):
    # Berechnung der Eingaben zur versteckten Schicht
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    # Aktivierung der versteckten Schicht
    hidden_layer_activation = sigmoid(hidden_layer_input)

    # Berechnung der Eingaben zur Ausgabeschicht
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    # Aktivierung der Ausgabeschicht
    output = sigmoid(output_layer_input)

    return hidden_layer_activation, output


# Rückwärtspropagation
def backward_propagation(X, y, hidden_layer_activation, output):
    # Fehler in der Ausgabeschicht
    output_error = y - output
    # Berechnung des Delta-Fehlers für die Ausgabeschicht
    output_delta = output_error * sigmoid_derivative(output)

    # Fehler in der versteckten Schicht
    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    # Berechnung des Delta-Fehlers für die versteckte Schicht
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_activation)

    return hidden_layer_delta, output_delta


# Gewichte und Biases aktualisieren
def update_weights(X, hidden_layer_activation, hidden_layer_delta, output_delta, learning_rate=0.1):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    # Aktualisierung der Gewichte und Biases für die Eingabe-zu-versteckte-Schicht-Verbindungen
    weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    # Aktualisierung der Gewichte und Biases für die versteckte-zu-Ausgabe-Schicht-Verbindungen
    weights_hidden_output += hidden_layer_activation.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate


# Training des Netzwerks
def train(X, y, epochs=100000, learning_rate=0.5):
    for epoch in range(epochs):
        # Vorwärtspropagation
        hidden_layer_activation, output = forward_propagation(X)

        # Rückwärtspropagation
        hidden_layer_delta, output_delta = backward_propagation(X, y, hidden_layer_activation, output)

        # Gewichte und Biases aktualisieren
        update_weights(X, hidden_layer_activation, hidden_layer_delta, output_delta, learning_rate)


def safe_model():
    model_parameters = {
        "weights_input_hidden": weights_input_hidden,
        "bias_hidden": bias_hidden,
        "weights_hidden_output": weights_hidden_output,
        "bias_output": bias_output,
        "input_layer_neurons": input_layer_neurons,
        "hidden_layer_neurons": hidden_layer_neurons,
        "output_neurons": output_neurons
    }
    with open("neural_network_model.pkl", "wb") as file:
        pickle.dump(model_parameters, file)
        print("Model saved successfully!")



def load_model():
    with open("neural_network_model.pkl", "rb") as file:
        model_parameters = pickle.load(file)

        global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
        weights_input_hidden = model_parameters["weights_input_hidden"]
        bias_hidden = model_parameters["bias_hidden"]
        weights_hidden_output = model_parameters["weights_hidden_output"]
        bias_output = model_parameters["bias_output"]

        global input_layer_neurons, hidden_layer_neurons, output_neurons
        input_layer_neurons = model_parameters["input_layer_neurons"]
        hidden_layer_neurons = model_parameters["hidden_layer_neurons"]
        output_neurons = model_parameters["output_neurons"]

        print("Model loaded successfully!")




def predict(X):
    hidden_layer_activation, output = forward_propagation(X)
    return output

if __name__ == "__main__":
    # Beispiel-Daten
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Eingabewerte für XOR-Funktion
    # Zielwerte für XOR-Funktion in One-Hot-Encoding
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    # Training
    train(X, y)

    # show the progress of the training in the console
    print("Final hidden weights: \n", weights_input_hidden)
    print("Final hidden bias: \n", bias_hidden)
    print("Final output weights: \n", weights_hidden_output)
    print("Final output bias: \n", bias_output)


    # Ausgabe der Vorhersagen
    hidden_layer_activation, output = forward_propagation(X)
    print("Predicted Output: \n", output)
    safe_model()


