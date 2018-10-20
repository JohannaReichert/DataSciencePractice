import pandas as pd
import numpy as np
import features.prep
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import json
from torch.autograd import Variable
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('data/colorsurvey/df_nospam_levenst.csv')

####### df with same number of rows per colorcluster #######
df = df.groupby('colorcluster', group_keys=False).apply(lambda x: x.sample(min(len(x), 300)))
df.shape

########### Enter Modelname ############
modelname = "Strati_2layer_H8"
save_model=False


"""
#Taking only the names used more than 2000 times
counts = df['colorname'].value_counts()
common_names = [list(counts.index)[i] for i in range(len(counts)) if counts[i]>500]  # bei 300 000 immer noch 314145
df = df[df['colorname'].isin(common_names)]
"""

y_column = 'colorcluster'

###################################################
df = df.sample(50000)
print(df.shape)
###################################################

# checking if there are any nans in the dataset
nans = df[df[y_column].isnull()]
print(f"NaNs: {nans}")

y = df[y_column]
y_nrs, y_names_dict = pd.factorize(y)
y_nrs = np.array(y_nrs)
y_nr_col_name = y_column + "nr"
df[y_nr_col_name] = y_nrs
n_classes = df[y_column].nunique()
x_all = df[['r','g','b']]


yel = df[y.isin(["yellow","light yellow","dark yellow","mustard"])]
print(y.isin(["yellow","light yellow","dark yellow","mustard"]).shape)
print(yel.shape)
print(f"nr of unique colornames: {df['colorname'].nunique()}")
print(f"nr of unique colorclusters: {df['colorcluster'].nunique()}")

cname_nrs, cname_names_dict = pd.factorize(df['colorname'])
cname_nrs = np.array(cname_nrs)


data = df[['r','g','b',y_nr_col_name]]

x_train, x_val, x_test, y_train, y_val, y_test = features.prep.split_train_val_test(data,y_nr_col_name,[0.6,0.2,0.2],stratify = True, random_state = 42)

# Neuronal Network

y_train = np.asarray(y_train).astype('float32')
y_val = np.asarray(y_val).astype('float32')

dtype = torch.float
device = torch.device("cpu")

# N sind die Anzahl der Datenpunkte
# D_in ist die Input-Dimension
# D_out ist die Output-Dimension
N, D_in, N_classes = x_train.shape[0], x_train.shape[1], n_classes
H = 8


x = torch.tensor(np.array(x_train), dtype = dtype, device = device)
y = torch.tensor(y_train, dtype=torch.long, device = device)


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
#criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_hist = []
batch_size = 32
#print(torch.max(y,1))
# Train

epochs = range(1000)
#idx = 0
for t in epochs:
    for batch in range(0, int(N / batch_size)):
        # Berechne den Batch

        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        # Berechne die Vorhersage (foward step)
        outputs = model(batch_x)

        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        loss = criterion(outputs, batch_y)

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
    if t % 50 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())
        if save_model:
            torch.save(model, 'trained_models/colors/'+modelname)

# output propabilities
print('output: ', torch.nn.functional.softmax(model(x)))

# output predicted classes
torch.max(model(x), 1)

plt.plot(loss_hist);
plt.show()

if save_model:
    torch.save(model,'trained_models/colors/'+modelname)

#2layer_H8: 2.248 crossentropy bei batchsize 32, epochs 1000, optimizer sgd. N = 50000
#2layer_H8_Adam: 2.429 crossentropy bei batchsize 32, epochs 1000, optimizer adam. N = 50000
#Strati_2layer_H8: 5.388 crossentropy bei batchsize 32, epochs 1000, optimizer sdg. N = 50000
