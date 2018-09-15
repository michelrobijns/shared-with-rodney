import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
#some change to the code

def main():
    # ***** Load the dataset *****

    # Dataset 0 - The harder problem
    data    = np.load('data0.npy')       # Shape:  (1500, 1)
    targets = np.load('targets0.npy')    # Shape:  (1500,)

    # Dataset 1 - The easier problem
    #data    = np.load('data1.npy')       # Shape:  (1500, 1)
    #targets = np.load('targets1.npy')    # Shape:  (1500,)

    # ***** Plot dataset to show you what it looks like *****

    plt.plot(data, targets)
    plt.title('I want to approximate (and overfit) this function')
    plt.xlabel('t')
    plt.ylabel('f')
    plt.show()

    # ***** Randomly shuffle dataset *****

    # Keep the ordered data to plot the results at the very end
    ordered_data = np.copy(data)
    ordered_targets = np.copy(targets)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    # Actual shuffling happens here
    data = data[indices]
    targets = targets[indices]

    # ***** Normalize the data *****

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    data -= mean
    data /= std

    # ***** Model definition *****

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=data[0].shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    # ***** Compile the model *****

    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae', 'mape'])

    # ***** Train the model *****

    history = model.fit(data,
                        targets,
                        epochs=100,
                        batch_size=None,
                        verbose=1)

    # ***** Plot losses and metrics *****

    loss = history.history['loss']
    mae = history.history['mean_absolute_error']
    mape = history.history['mean_absolute_percentage_error']
    epochs = range(1, len(loss) + 1)

    # Plot loss and mean absolute error
    #plt.plot(epochs, loss, label='Training loss')
    #plt.plot(epochs, mae, label='Training MAE')
    #plt.title('Training loss and MAE')
    #plt.xlabel('Epochs')
    #plt.legend()
    #plt.show()  # You can uncomment, but it's not that interesting

    # Plot mean absolute percentage error
    plt.plot(epochs[5:], mape[5:], label='Training MAPE')
    plt.title('Training MAPE (Mean Absolute Percentage Error)')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()

    # ***** Plot dataset and predictions *****

    plt.plot(ordered_data, ordered_targets, 'g', label="target")

    predictions = model.predict((ordered_data - mean) / std)

    plt.plot(ordered_data, predictions, 'r', label="prediction")
    plt.title('Targets and NN predictions')
    plt.xlabel('t')
    plt.ylabel('f')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
