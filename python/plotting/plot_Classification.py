import matplotlib.pyplot as plt
import numpy as np

def plot_classification(y_true, y_predict, save=False, fileName="Class_test"):
    y_predict_class = np.argmax(y_predict, axis=1)
    #plt.hist(y_predict_class[y_true==0], label=r'Signal', histtype='step')
    #plt.hist(y_predict_class[y_true==1], label=r'$t\overline{t}$', histtype='step')
    #plt.hist(y_predict_class[y_true==2], label=r'Single Top', histtype='step')
    #plt.hist(y_predict_class[y_true==3], label=r'$W$ + Jets', histtype='step')
    
    
    labels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    #hists = [y_predict_class[y_true==0],y_predict_class[y_true==1],y_predict_class[y_true==2],y_predict_class[y_true==3]] Predicted Class on x-axis
    hists = [y_true[y_predict_class==0],y_true[y_predict_class==1],y_true[y_predict_class==2],y_true[y_predict_class==3]]
    
    plt.hist(hists, bins=4, stacked=True,histtype='stepfilled', label=labels, density=True)
    
    plt.xlabel('True Class')
    #plt.yscale('log')
    plt.legend(loc='best')
    plt.xticks(np.arange(4),(r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets'))
    plt.title('Classification')
    
    #print('Signal: ', y_predict_class[y_true==0].shape[0])
    #print('tt: ', y_predict_class[y_true==1].shape[0])
    #print('Single Top: ', y_predict_class[y_true==2].shape[0])
    #print('W + jets: ', y_predict_class[y_true==3].shape[0])
    
    print('Identified as:')
    print('Signal: ', y_true[y_predict_class==0].shape[0])
    print('tt: ', y_true[y_predict_class==1].shape[0])
    print('Single Top: ', y_true[y_predict_class==2].shape[0])
    print('W + jets: ', y_true[y_predict_class==3].shape[0])
    
    
    if save:
        plt.savefig("plots/"+fileName+".pdf")
        plt.savefig("plots/"+fileName+".png")
        plt.close()
    
def plot_classification(y_true, y_predict, save=False, fileName="Class_test"):
    y_predict_class = np.argmax(y_predict, axis=1)
    xaxis = [0,1,2,3]
    for i in range(0,3):
        plt.bar