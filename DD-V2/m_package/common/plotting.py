import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import confusion_matrix


def plot_history(history, valid, train, path, name, val_history = None):
    """
    Plots the training and validation loss along with the training and validation AUC over epochs.

    Parameters:
    history (list): Training loss history.
    valid (list): Validation AUC history.
    train (list): Training AUC history.
    path (str): Path to save the plot.
    name (str): Name to save the plot file.
    val_history (list, optional): Validation loss history. Defaults to None.
    """
    fig, ax = plt.subplots(1,2)
    tick = max(len(history) // 5, 1)

    #Plotting training and validation loss
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax[0].plot(history, label="train loss")
    if val_history is not None:
        ax[0].plot(val_history, label="valid loss")
    ax[0].title.set_text('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    #Plotting training and validation AUC
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax[1].plot(train, label="train AUC")
    ax[1].plot(valid, label="valid AUC")
    ax[1].title.set_text('metric AUC')
    ax[1].set_ylabel('AUC')
    ax[1].set_xlabel('epoch')
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(f'{path}/{name}_auc_and_loss.png', bbox_inches='tight') 


def plot_history_loss_final(loss, val_loss, path, name):
    """
    Plots the mean and standard deviation of training and validation loss over epochs.

    Parameters:
    loss (list of lists): Training loss from multiple experiments.
    val_loss (list of lists): Validation loss from multiple experiments.
    path (str): Path to save the plot.
    name (str): Name to save the plot file.
    """
    fig, ax = plt.subplots(figsize=(12, 6))  
    tick = max(len(loss[0]) // 5, 1)
    mean_loss = np.mean(np.array(loss), axis=0) 
    std_loss = np.std(np.array(loss), axis=0) 
    mean_val_loss = np.mean(np.array(val_loss), axis=0) 
    std_val_loss = np.std(np.array(val_loss), axis=0) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick)) 
    ax.plot(mean_loss, label="train loss", linewidth=2, color='green')
    ax.fill_between(range(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, color='green', alpha=0.2)
    ax.plot(mean_val_loss, label="validation loss", linewidth=2, color='olivedrab')
    ax.fill_between(range(len(mean_val_loss)), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='olivedrab', alpha=0.2)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'{path}/{name}_LOSS_final.png', bbox_inches='tight')


def plot_history_metric_final(auc, val_auc, path, name):
    """
    Plots the mean and standard deviation of training and validation AUC over epochs.

    Parameters:
    auc (list of lists): Training AUC from multiple experiments.
    val_auc (list of lists): Validation AUC from multiple experiments.
    path (str): Path to save the plot.
    name (str): Name to save the plot file.
    """
    fig, ax = plt.subplots(figsize=(12, 6))  
    tick = max(len(auc[0]) // 5, 1)
    mean_auc = np.mean(np.array(auc), axis=0) 
    std_auc = np.std(np.array(auc), axis=0) 
    mean_val_auc = np.mean(np.array(val_auc), axis=0) 
    std_val_auc = np.std(np.array(val_auc), axis=0) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick)) 
    ax.plot(mean_auc, label="train AUC", linewidth=2, color='green')
    ax.fill_between(range(len(mean_auc)), mean_auc - std_auc, mean_auc + std_auc, color='green', alpha=0.2)
    ax.plot(mean_val_auc, label="validation AUC", linewidth=2, color='olivedrab')
    ax.fill_between(range(len(mean_val_auc)), mean_val_auc - std_val_auc, mean_val_auc + std_val_auc, color='olivedrab', alpha=0.2)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Epoch')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'{path}/{name}_AUC_final.png', bbox_inches='tight')


def conf_matrix(y_pred_lab, y_test_lab, model_name):
    """
    Plots the confusion matrix with percentage values and standard deviation.

    Parameters:
    y_pred_lab (list of arrays): Predicted labels from multiple experiments.
    y_test_lab (list of arrays): True labels from multiple experiments.
    model_name (str): Name of the model to save the plot file.
    """
    n_experiments = len(y_pred_lab)
    cms = []
    for i in range(n_experiments):
        cm = confusion_matrix(y_test_lab[i], y_pred_lab[i])
        cms.append(cm)
    cms = np.array(cms)
    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    total_samples = np.sum(mean_cm)
    percentage_cm = (mean_cm / total_samples) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(percentage_cm, cmap="YlGn")
    fig.colorbar(cax)
    threshold = percentage_cm.max() / 2.0
    for (i, j), val in np.ndenumerate(percentage_cm):
        color = "white" if val > threshold else "black"
        ax.text(j, i, f'{val:.2f}%\n±{(std_cm[i,j] / mean_cm[i,j]) * 100 :.2f}%', ha='center', va='center', color=color)
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Dyslexic', 'Dyslexic'], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-Dyslexic', 'Dyslexic'], fontsize=12)
    fig.tight_layout()
    plt.savefig(f'Matrices_pictures/matrix_{model_name}.png',  bbox_inches='tight')