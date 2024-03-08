from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, fft
import tensorflow as tf

def compute_doppler_maps(data:np.ndarray) -> np.ndarray:
    """
    Compute the doppler maps from the input data.
    """
    # Compute the fft along the chirps axis
    fft_data = rfft(data, axis=2)
    # Compute the fft along the samples axis
    fft_data = fft_data[:,:,1:]
    fft_data = fft(fft_data, axis=3)

    return np.abs(fft_data)

def compute_doppler_maps_old(data:np.ndarray) -> np.ndarray:
    fft_data = fft(data, axis=2)        # fft on chirps
    fft_data = fft_data[:,:,:fft_data.shape[2]//2]  # drop second symmetric half
    fft_data = fft(fft_data, axis=3)        # cfft on samples
    fft_data = fft_data[:,:,1:]
    return np.abs(fft_data)

def get_people_from_path(path:Path) -> list:
    l = set([f.parts[-3] for f in path.glob('**/*.npy')])
    return list(l)

def get_gestures_from_path(path:Path) -> list:
    l = set([f.parts[-2] for f in path.glob('**/*.npy')])
    return list(l)

def load_data(path:Path, people:list, gestures:list, antennas:int=3) -> pd.DataFrame:
    assert len(people) > 0, "No people provided"
    assert len(gestures) > 0, "No gestures provided"
    assert antennas in [1, 2, 3], "Antennas must be 1 to 3"

    files = path.glob('**/*.npy')
    files = [f for f in files if (f.parts[-3] in people) and (f.parts[-2] in gestures)]

    dataset = list()
    for f in files:
        user, gesture = f.parts[-3:-1]
        data = np.load(f)[...,:antennas]
        # explode the data on samples dimension
        d = [{'user':user, 'label':gesture, 'data':sample} for sample in data]
        dataset.extend(d)

    dataset = pd.DataFrame(dataset)
    dataset = pd.get_dummies(dataset, columns=['label'], prefix='ges')

    return dataset

# TODO Add a precompute, add a data_shape attribute
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, raw_data, batch_size, sequence_labels:bool=False) -> None:
        self.raw_data = raw_data
        self.batch_size = batch_size
        self.sequence_labels = sequence_labels

        self.rng = np.random.default_rng()
        self.clip_value = 1
        self.chirps_slice_size = 32

        self.idxes = self.rng.permutation(len(self.raw_data))

        self.tanh = np.tanh(np.linspace(0,np.pi,batch_size))[np.newaxis, :, np.newaxis]


    def __len__(self):
        return int(np.floor(len(self.raw_data) / self.batch_size))


    def __getitem__(self, index):
        idxes = self.idxes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self._data_gen(idxes)
        return X, y


    def on_epoch_end(self):
        self.idxes = self.rng.permutation(len(self.raw_data))


    def _data_gen(self, idxes):
        batch = self.raw_data.iloc[idxes]

        X = np.stack(batch['data'].to_numpy())

        num_frames = X.shape[1]
        num_chirps = X.shape[2]

        chirps_s = self.rng.integers(0, num_chirps - self.chirps_slice_size)
        X_slice = X[...,chirps_s:chirps_s+self.chirps_slice_size,:,:]
        X = compute_doppler_maps_old(X_slice)
        X = X.clip(0, self.clip_value)

        # select all columns but the first two
        labels = batch[batch.columns[2:]].to_numpy().astype(np.float32)
        if self.sequence_labels:
            y = np.zeros((self.batch_size, num_frames, labels.shape[-1]))
            y[:,:] = labels[:,np.newaxis,:]
            labels = y
        return X, labels

    def get_data(self):
        x, y = [], []
        for i in range(len(self)):
            batch_x, batch_y = self[i]
            y.append(batch_y)
            x.append(batch_x)
        return np.concatenate(x), np.concatenate(y)


if __name__ == "__main__":
    # tiny tests
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('dir', help='Folder containing the dataset.')

    args = parser.parse_args()

    inputdir = Path(args.dir)

    people = get_people_from_path(inputdir)
    people = people[1:2]
    gestures = get_gestures_from_path(inputdir)

    print(f"Using {len(people)} people and {len(gestures)} gestures")

    data = load_data(inputdir, people, gestures)

    dg = DataGenerator(data, 32)
    x,y = dg.__getitem__(0)

    # generate 4 subplots
    fig, axs = plt.subplots(2, 2)

    # plot the first 4 images in the dataset
    axs[0, 0].imshow(x[0,10,:,:,0])
    axs[0, 1].imshow(x[1,10,:,:,0])
    axs[1, 0].imshow(x[2,10,:,:,0])
    axs[1, 1].imshow(x[3,10,:,:,0])
    # add label from argmax of y to each subplot
    axs[0, 0].set_title(np.argmax(y[0]))
    axs[0, 1].set_title(np.argmax(y[1]))
    axs[1, 0].set_title(np.argmax(y[2]))
    axs[1, 1].set_title(np.argmax(y[3]))

    print("X shape:", x.shape, "Y shape:", y.shape)
    print(y[0])

    plt.show()
