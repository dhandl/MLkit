import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,save=False,fileName="CM_test"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm is the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    fig = plt.gcf()
    fig.set_size_inches(9., 7.)
    
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_ConfusionMatrix.pdf")
        plt.savefig("plots/"+fileName+"_ConfusionMatrix.png")
        plt.close()
    
    
def plot_confusion_matrix(y_true, y_predict, filename="Test",save=False):
    """
    Plotting (and printing) the confusion matrix
    """
    yhat_cls = np.argmax(y_predict, axis=1)
    cnf_matrix = confusion_matrix(y_true, yhat_cls)
    np.set_printoptions(precision=2)
    
    draw_confusion_matrix(cnf_matrix, classes=[r'Signal', r'$t\overline{t}$', 'Single Top', r'$W$ + jets'],
                      normalize=True,
                      title='Normalized Confusion matrix',save=save,fileName=filename)