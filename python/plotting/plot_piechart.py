import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pie_chart(y_true, y_predict, fileName="Test", save=False):
    print('Plotting pie charts...')    
    y_predict_class = np.argmax(y_predict, axis=1)
    sizes_predicted = np.bincount(y_predict_class)
    if(sizes_predicted.shape[0] < 4):
        for i in range(0,4-sizes_predicted.shape[0]):
            sizes_predicted = np.append(sizes_predicted, 0)
            
    sizes_true = np.bincount(y_true.astype(int))
    
    namelabels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    labels_true = []
    labels_predicted=[]
    for i in range(0,4):
        labels_true.append(namelabels[i] + ' [' + str(sizes_true[i]) + ']')
        labels_predicted.append(namelabels[i] + ' [' + str(sizes_predicted[i]) + ']')
    
    plt.pie(sizes_predicted,labels=labels_predicted)
    plt.title('predicted distribution')
    plt.axis('equal')
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_predicted_pie.pdf")
        plt.savefig("plots/"+fileName+"_predicted_pie.png")
        plt.close()
        
    plt.figure()
    plt.pie(sizes_true,labels=labels_true)
    plt.title('true distribution')
    plt.axis('equal')
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_true_pie.pdf")
        plt.savefig("plots/"+fileName+"_true_pie.png")
        plt.close()
    