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


class PennyHAE(nn.Module):

    def __init__(self, input_size: int, n_qubits: int, reps: int) -> None:

        super(PennyHAE, self).__init__()
        self.qdevice = qml.device('default.qubit', n_qubits)

        @qml.qnode(self.qdevice, interface='torch')
        def hae_block(inputs: torch.Tensor, weights: torch.Tensor):
            # weight [REPS Q 2]
            R, Q, _ = weights.shape
            for i in range(R - 1):
                for j in range(Q):
                    qml.PauliRot(weights[i, j, 0], "Y", wires=j)
                    qml.PauliRot(weights[i, j, 1], "X", wires=j)
                for j in range(Q - 1, Q - 1 + Q):
                    qml.CNOT(wires=[j % n_qubits, (j + 1) % n_qubits])
                for j in range(Q):
                    qml.PauliRot(inputs[j], "X", wires=j)
            for j in range(Q):
                qml.PauliRot(weights[i, j, 0], "Y", wires=j)
                qml.PauliRot(weights[i, j, 1], "X", wires=j)
            for j in range(Q - 1, Q - 1 + Q):
                qml.CNOT(wires=[j % n_qubits, (j + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(qubits)]

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
        self.qlayer = qml.qnn.TorchLayer(hae_block, weight_shapes)

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
        return self.decoder(X)


if __name__ == '__main__':

    phae = PennyHAE(10, 2, 3)
    x = torch.rand((4, 10))
    y = phae(x)
    print(y)
