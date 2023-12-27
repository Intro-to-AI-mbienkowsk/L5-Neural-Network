import matplotlib.pyplot as plt
from mnist import *
from sklearn.metrics import *


def import_data():
    train_x = train_images()
    train_y = train_labels()
    test_x = test_images()
    test_y = test_labels()
    train_x_processed = train_x.reshape(train_x.shape[0], -1) / 255.0
    test_x_processed = test_x.reshape(test_x.shape[0], -1) / 255.0
    return (train_x_processed, train_y), (test_x_processed, test_y)


def display_confusion_matrix(pred_y, actual_y):
    labels = [i for i in range(-1, 10)]
    cm = confusion_matrix(actual_y, pred_y, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    precision = precision_score(actual_y, pred_y, average='weighted',
                                labels=labels[1:])
    accuracy = accuracy_score(actual_y, pred_y)
    f1 = f1_score(actual_y, pred_y, average='weighted', labels=labels[1:])

    textstr = '\n'.join((
        f'Precision: {precision:.2f}',
        f'Accuracy: {accuracy:.2f}',
        f'F1 Score: {f1:.2f}'))
    plt.subplots_adjust(right=.7)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.73, 0.5, textstr, fontsize=14, bbox=props)

    plt.show()
