import itertools

import matplotlib.pyplot as plt
import numpy as np


# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True):
    for _i, label in enumerate(labels):
        plt.plot(
            tpr[label],
            fpr[label],
            label='{} tagger, AUC = {:.1f}%'.format(label.replace('j_', ''), auc[label] * 100.0),
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90, 'hls4ml', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)


def rocData(y, predict_test, labels):
    import pandas as pd
    from sklearn.metrics import auc, roc_curve

    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i]
        df[label + '_pred'] = predict_test[:, i]

        fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + '_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1


def makeRoc(y, predict_test, labels, linestyle='-', legend=True):
    if 'j_index' in labels:
        labels.remove('j_index')

    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
    return predict_test


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


def _collect_kernel_weights(model, layer_classes=('Conv2D', 'Dense', 'QConv2D', 'QDense', 'QConv2DBatchnorm')):
    weights_by_layer = {}
    for layer in model.layers:
        if layer.__class__.__name__ in layer_classes:
            weights = layer.get_weights()
            if weights:
                weights_by_layer[layer.name] = weights[0].flatten()
    return weights_by_layer


def _print_layer_sparsity(title, weights_by_layer):
    print(title)
    for layer_name, weights in weights_by_layer.items():
        sparsity = np.mean(weights == 0.0)
        print(f'{layer_name}: {sparsity:.1%} zeros')


def plot_weight_sparsity(before_model, after_model, bins=80):
    before_weights = _collect_kernel_weights(before_model)
    after_weights = _collect_kernel_weights(after_model)

    _print_layer_sparsity('Layer sparsity before pruning:', before_weights)
    print()
    _print_layer_sparsity('Layer sparsity after pruning:', after_weights)

    before_all = np.concatenate(list(before_weights.values()))
    after_all = np.concatenate(list(after_weights.values()))
    value_min = min(before_all.min(), after_all.min())
    value_max = max(before_all.max(), after_all.max())
    if value_min == value_max:
        value_min -= 1
        value_max += 1
    shared_bins = np.linspace(value_min, value_max, bins)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    axes[0].hist(before_all, bins=shared_bins, color='red', alpha=0.75, log=True)
    axes[0].axvline(0.0, color='black', linewidth=1)
    axes[0].set_title('Before pruning')
    axes[0].set_xlabel('Weight value')
    axes[0].set_ylabel('Count (log scale)')

    axes[1].hist(after_all, bins=shared_bins, color='blue', alpha=0.75, log=True)
    axes[1].axvline(0.0, color='black', linewidth=1)
    axes[1].set_title('After pruning')
    axes[1].set_xlabel('Weight value')

    fig.suptitle('Weight distributions before and after pruning')
    plt.tight_layout()
    return fig, axes
