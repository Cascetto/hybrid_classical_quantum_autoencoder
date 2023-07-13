import pennylane as qml
import torch.nn as nn
import torch

qubits = 2
dev = qml.device('default.qubit', qubits)


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(qubits))
    qml.BasicEntanglerLayers(weights, wires=range(qubits))  # shapes [NQubits]
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(qubits)]


# n_layers = 6
# weight_shapes = {"weights": (n_layers, qubits)}

# qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# layer1 = nn.Linear(2, 2)
# layer2 = nn.Linear(2, 2)
# softmax = nn.Softmax(1)
# layers = [layer1, qlayer, layer2, softmax]

# model = nn.Sequential(*layers)

# opt = torch.optim.SGD(model.parameters(), lr=0.001)
# loss = nn.L1Loss()

# x = torch.rand((10, 2))
# y = model(x)
# print(y)

nqubits = 2
qdevice = qml.device('default.qubit', nqubits)


@qml.qnode(qdevice, interface='torch')
def circuit(inputs, theta):
    qml.RX(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(theta, wires=0)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))


@qml.qnode(qdevice, interface='torch')
def hae_block(inputs: torch.Tensor, weights: torch.Tensor):
    # weight [REPS Q 2]
    R, Q, _ = weights.shape
    for i in range(R - 1):
        for j in range(Q):
            qml.PauliRot(weights[i, j, 0], "Y", wires=j)
            qml.PauliRot(weights[i, j, 1], "X", wires=j)
        for j in range(Q - 1, Q - 1 + Q):
            qml.CNOT(wires=[j % nqubits, (j + 1) % nqubits])
        for j in range(Q):
            qml.PauliRot(inputs[j], "X", wires=j)
    print("First loop done")
    for j in range(Q):
        qml.PauliRot(weights[i, j, 0], "Y", wires=j)
        qml.PauliRot(weights[i, j, 1], "X", wires=j)
    print("second loop done")
    for j in range(Q - 1, Q - 1 + Q):
        qml.CNOT(wires=[j % nqubits, (j + 1) % nqubits])
    print("third loop done")
    # return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]
    return qml.expval(qml.PauliZ(0))


class PennyHAE(nn.Module):

    def __init__(self, input_size: int, n_qubits: int, reps: int) -> None:

        super(PennyHAE, self).__init__()

        self.encoder = nn.Sequential(
            *[
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, input_size // 4),
                nn.ReLU(),
                nn.Linear(input_size // 4, n_qubits)
            ]
        )

        weight_shapes = {"weights": (reps, qubits, 2)}
        weight_shapes = {"theta": (1)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.decoder = nn.Sequential(
            *[
                nn.Linear(n_qubits, input_size // 4),
                nn.ReLU(),
                nn.Linear(input_size // 4, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, input_size)
            ]
        )

    def forward(self, X):
        X = self.encoder(X)
        print(X.shape)
        X = self.qlayer(X)
        print(X.shape)
        return self.decoder(X)


n_qubits = 2
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)
weight_shapes = {"weights": (n_layers, n_qubits)}


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(2, 4)
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, 2, dim=1)
        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_2(x)
        return self.softmax(x)


if __name__ == '__main__':

    model = HybridModel()
    x = torch.rand((4, 2))
    y = model(x)
    print(y)
