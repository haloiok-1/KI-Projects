import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# Beispielhafte Netzwerkparameter
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 2

weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))


def load_model():
    global input_layer_neurons, hidden_layer_neurons, output_neurons
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    # Laden der Netzwerkparameter
    with open('neural_network_model.pkl', 'rb') as file:
        model = pickle.load(file)
        input_layer_neurons = model['input_layer_neurons']
        hidden_layer_neurons = model['hidden_layer_neurons']
        output_neurons = model['output_neurons']

        weights_input_hidden = model['weights_input_hidden']
        bias_hidden = model['bias_hidden']
        weights_hidden_output = model['weights_hidden_output']
        bias_output = model['bias_output']


# Funktion zur Visualisierung des Netzwerks
def plot_neural_network(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Knotenpositionen definieren
    print("bias_hidden", bias_hidden)
    print("bias_output", bias_output)

    # Erstellen des Graphen
    G = nx.DiGraph()

    # add input nodes
    G.add_nodes_from([f'input_{i + 1}' for i in range(input_layer_neurons)])
    G.add_nodes_from([f'hidden_{i + 1}' for i in range(hidden_layer_neurons)])
    G.add_nodes_from([f'output_{i + 1}' for i in range(output_neurons)])

    G.add_nodes_from([f'bias_hidden_{i + 1}' for i in range(hidden_layer_neurons)])
    G.add_nodes_from([f'bias_output_{i + 1}' for i in range(output_neurons)])

    general_node_size = 1000
    bias_node_size = 400

    pos = {}

    for i in range(input_layer_neurons):
        pos[f'input_{i + 1}'] = (0, (input_layer_neurons / 2) - i - 0.5)

    for i in range(hidden_layer_neurons):
        print(hidden_layer_neurons)
        print((hidden_layer_neurons / 2) - (0.5 - i))
        print(hidden_layer_neurons / 2)
        pos[f'hidden_{i + 1}'] = (1, (hidden_layer_neurons / 2) - 0.5 - i)
        pos[f'bias_hidden_{i + 1}'] = (1, (hidden_layer_neurons / 2) - i)

    for i in range(output_neurons):
        pos[f'output_{i + 1}'] = (2, (output_neurons / 2) - 0.5 - i)
        pos[f'bias_output_{i + 1}'] = (2, (output_neurons / 2) - i)

    print("pos", pos)

    nx.draw_networkx_nodes(G, pos, nodelist=[f'input_{i + 1}' for i in range(input_layer_neurons)],
                           node_size=general_node_size, node_color='gray')
    nx.draw_networkx_labels(G, pos, labels={f'input_{i + 1}': f'input_{i + 1}' for i in
                                            range(input_layer_neurons)})  # input labels

    nx.draw_networkx_nodes(G, pos, nodelist=[f'hidden_{i + 1}' for i in range(hidden_layer_neurons)],
                           node_size=general_node_size, node_color='darkgray')
    nx.draw_networkx_labels(G, pos, labels={f'hidden_{i + 1}': f'hidden_{i + 1}' for i in
                                            range(hidden_layer_neurons)})  # hidden labels

    nx.draw_networkx_nodes(G, pos, nodelist=[f'output_{i + 1}' for i in range(output_neurons)],
                           node_size=general_node_size, node_color='gray')
    nx.draw_networkx_labels(G, pos, labels={f'output_{i + 1}': f'output_{i + 1}' for i in
                                            range(output_neurons)})  # output labels

    nx.draw_networkx_nodes(G, pos, nodelist=[f'bias_hidden_{i + 1}' for i in range(hidden_layer_neurons)],
                           node_size=bias_node_size, node_color='red')

    nx.draw_networkx_nodes(G, pos, nodelist=[f'bias_output_{i + 1}' for i in range(output_neurons)], )

    nx.draw_networkx_nodes(G, pos, nodelist=[f'bias_output_{i + 1}' for i in range(output_neurons)],
                           node_size=bias_node_size, node_color='red')

    nx.draw_networkx_labels(G, pos, labels={f'bias_hidden_{i + 1}': f'bias_hidden_{i + 1}' for i in
                                            range(hidden_layer_neurons)}, font_size=10)  # hidden bias labels
    nx.draw_networkx_labels(G, pos, labels={f'bias_output_{i + 1}': f'bias_output_{i + 1}' for i in
                                            range(output_neurons)}, font_size=10)  # output bias labels

    # Hinzufügen der Kanten mit Gewichten (Eingabeschicht -> Versteckte Schicht)
    for i in range(input_layer_neurons):
        for j in range(hidden_layer_neurons):
            G.add_edge(f'input_{i + 1}', f'hidden_{j + 1}', weight=weights_input_hidden[i, j])

    # Hinzufügen der Kanten mit Gewichten (Versteckte Schicht -> Ausgabeschicht)
    for i in range(hidden_layer_neurons):
        for j in range(output_neurons):
            G.add_edge(f'hidden_{i + 1}', f'output_{j + 1}', weight=weights_hidden_output[i, j])

    # Hinzufügen der Kanten mit Gewichten (Bias -> Versteckte Schicht)
    for i in range(hidden_layer_neurons):
        G.add_edge(f'bias_hidden_{i + 1}', f'hidden_{i + 1}', weight=bias_hidden[0, i])

    # Hinzufügen der Kanten mit Gewichten (Bias -> Ausgabeschicht)
    for i in range(output_neurons):
        G.add_edge(f'bias_output_{i + 1}', f'output_{i + 1}', weight=bias_output[0, i])

    # Zeichnen des Graphen
    edges = G.edges(data=True)
    weights = [data['weight'] for u, v, data in edges]

    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Greens)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in edges},
                                 font_color='red')

    plt.title('Neuronales Netzwerk')
    plt.show()


# Plotten des Netzwerks
load_model()
plot_neural_network(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
