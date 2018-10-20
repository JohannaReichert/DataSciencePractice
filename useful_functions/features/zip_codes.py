import numpy as np


def multires(data):
    num_samples = data.shape[0]
    reshaped_input = data.reshape(num_samples, 16, 16)
    new_feature_collection = data.mean(1).reshape((-1, 1))

    for res in [2, 4, 8]:
        pixel = res
        pixel_size = int(16 / pixel)
        new_features = np.zeros((num_samples, pixel * pixel))
        for k in range(reshaped_input.shape[0]):
            for i in range(0, pixel):
                for j in range(0, pixel):
                    new_features[k, i * pixel + j] = reshaped_input[k, (i * pixel_size):(i + 1) * pixel_size,
                                                     (j * pixel_size):(j + 1) * pixel_size].mean()
        new_feature_collection = np.append(new_feature_collection, new_features, axis=1)

    return new_feature_collection
