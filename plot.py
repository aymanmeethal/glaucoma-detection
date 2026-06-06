import matplotlib.pyplot as plt
import math


def plot_predictions(predictions):
    plt.style.use('ggplot')  # safe style

    def fun(pred, i):  # creates the bar plot on predictions
        negative = pred[0]
        positive = pred[1]

        plt.bar(
            ['GlaucomaNegative', 'GlaucomaPositive'],
            [negative, positive],
            color=['#42f557', '#f54242']
        )
        plt.ylim(0, 1)
        plt.legend([f'Negative: {negative:.4f}\nPositive: {positive:.4f}'])
        plt.title(f'Image: {i}')

    length = len(predictions)

    if length > 1:  # if predictions are available for more than one image
        rows = math.ceil(length / 2)
        columns = 2

        plt.figure(figsize=(10, 5 * rows))

        for i in range(length):
            try:
                plt.subplot(rows, columns, i + 1)
                fun(predictions[i][0], i + 1)
            except Exception as e:
                print(f'Error: unable to plot image {i + 1} -> {e}')
    else:  # otherwise for single image prediction
        plt.figure(figsize=(6, 5))
        fun(predictions[0][0], 1)

    plt.tight_layout()
    plt.savefig('results.pdf')  # save figure before showing it
    print('Results are saved as results.pdf in same directory')
    plt.show()
    plt.close()


def plot_accuracy(history, epochs):  # plot training accuracy/loss
    plt.style.use('ggplot')  # safe style

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(10, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig('accuracyPlot.pdf')
    print('Accuracy plot saved as accuracyPlot.pdf')

    plt.show()
    plt.close()


def plot_samples():
    print('function not available')
    pass