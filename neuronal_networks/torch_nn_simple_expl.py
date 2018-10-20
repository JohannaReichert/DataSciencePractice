from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


# N sind die Anzahl der Datenpunkte
# D_in ist die Input-Dimension
# D_out ist die Output-Dimension
N, D_in, N_classes = 4, 2, 2
H = 8
# Input sind die alle zweier-Tupel aus TRUE und FALSE, Target ist der zugehörige XOR wert
x = torch.Tensor(np.array([[0,0], [0,1], [1,0], [1,1]]))
y = torch.Tensor(np.array([0,1,1,0])).long()

# Logistic regression model
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, N_classes)
# )


## 2-Layer-Network
model = torch.nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, N_classes)
)

# Hyper-parameters
learning_rate = 0.001

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_hist = []

# Train
for t in range(50000):
    # Berechne die Vorhersage (foward step)
    outputs = model(x)

    # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
    loss = criterion(outputs, y)

    # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
    if t % 500 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())


plt.plot(loss_hist);
plt.show()

torch.max(model(x), 1)