import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def create_pred_color_cube(modelname):
    # sample used for evaluation
    color_cube = torch.zeros((256, 256, 256, 3))

    print('creating cube')
    for i in range(0, 256):
        color_cube[i, :, :, 0] = i
        color_cube[:, i, :, 1] = i
        color_cube[:, :, i, 2] = i

    print('predicting cube')
    dtype = torch.float
    device = torch.device("cpu")

    model = torch.load('trained_models/colors/'+modelname)

    input_data = color_cube.reshape(256 * 256 * 256, 3)
    print(input_data.shape)

    outputs = model(input_data)
    y_pred = torch.max(outputs.data, 1)[1]

    pd_data = pd.DataFrame(input_data.data.numpy())
    pd_data['pred'] = pd.DataFrame(y_pred.data.numpy())

    pd_data.to_csv('data/colorsurvey/pred_color_cube_'+modelname+'.csv')


def show_2d_rgb_comp(modelname):
    r = np.linspace(0, 1, 256).reshape(-1, 1).repeat(256, axis=1)
    g = np.linspace(1, 0, 256).reshape(-1, 1).repeat(256, axis=1)
    b = np.linspace(0, 1, 256).reshape(-1, 1).T.repeat(256, axis=0)
    image = np.stack([r, g, b], axis=2)
    image_scaled = image / image.max(axis=2)[:, :, None]
    image_scaled *= 1
    plt.figure(figsize=(6, 6))
    plt.imshow(image_scaled, origin='lower', extent=(0, 1, 0, 1))
    print(image_scaled.shape)
    data = pd.read_csv("data/colorsurvey/pred_color_cube_"+modelname+".csv", index_col=0)
    mean_colors = data.groupby('pred').median()
    mean_colors = mean_colors.astype('int32')
    data = data.astype('int32')
    rgb_image = (image * 255).astype('int32')
    np_data = data.values.reshape(256, 256, 256, 4)
    rgb_image_new = np.full_like(rgb_image, 0)
    for i in range(0, 256):
        for j in range(0, 256):
            color = rgb_image[i, j]
            pred = np_data[color[0], color[1], color[2]][3]
            new_color = mean_colors.loc[pred]
            rgb_image_new[i, j] = new_color
    image_new_scaled = rgb_image_new / rgb_image_new.max(axis=2)[:, :, None]
    image_new_scaled = rgb_image_new / 256
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_scaled, origin='lower', extent=(0, 1, 0, 1))
    plt.subplot(1, 2, 2)
    plt.imshow(image_new_scaled, origin='lower', extent=(0, 1, 0, 1))
    plt.show()