import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pie_chart(y_true, y_predict, w, fileName="Test", save=False):
    print('Plotting pie charts...')    
    y_predict_class = np.argmax(y_predict, axis=1)
    sizes_predicted = np.bincount(y_predict_class)
    if(sizes_predicted.shape[0] < 4):
        for i in range(0,4-sizes_predicted.shape[0]):
            sizes_predicted = np.append(sizes_predicted, 0)
            
    sizes_true = [np.sum(w[y_true==0]), np.sum(w[y_true==1]), np.sum(w[y_true==2]), np.sum(w[y_true==3])]
    
    sum_true = np.sum(w)
    sum_predicted = float(np.sum(sizes_predicted))
    
    namelabels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    labels_true = []
    labels_predicted=[]
    for i in range(0,4):
        labels_true.append(namelabels[i] + ' [' + str('{:.1%}'.format(sizes_true[i]/sum_true)) + ']')
        labels_predicted.append(namelabels[i] + ' [' + str('{:.1%}'.format(sizes_predicted[i]/sum_predicted)) + ']')
        
    colors=['r','b','g','orange']
    
    plt.pie(sizes_predicted,labels=labels_predicted,colors=colors)
    #plt.title('predicted distribution')
    plt.axis('equal')
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_predicted_pie.pdf")
        plt.savefig("plots/"+fileName+"_predicted_pie.png")
        plt.close()
        
    plt.figure()
    plt.pie(sizes_true,labels=labels_true,colors=colors)
    #plt.title('true distribution')
    plt.axis('equal')
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_true_pie.pdf")
        plt.savefig("plots/"+fileName+"_true_pie.png")
        plt.close()


#def plot_pie_chart(y_true, y_predict, w, fileName="Test", save=False):
    #print('Plotting pie charts...')    
    #y_predict_class = np.argmax(y_predict, axis=1)
    #sizes_predicted = np.bincount(y_predict_class)
    #if(sizes_predicted.shape[0] < 4):
        #for i in range(0,4-sizes_predicted.shape[0]):
            #sizes_predicted = np.append(sizes_predicted, 0)
            
    #sizes_true = np.bincount(y_true.astype(int))
    
    #sum_true = float(np.sum(sizes_true))
    #sum_predicted = float(np.sum(sizes_predicted))
    
    #namelabels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    #labels_true = []
    #labels_predicted=[]
    #for i in range(0,4):
        #labels_true.append(namelabels[i] + ' [' + str('{:.1%}'.format(sizes_true[i]/sum_true)) + ']')
        #labels_predicted.append(namelabels[i] + ' [' + str('{:.1%}'.format(sizes_predicted[i]/sum_predicted)) + ']')
    
    #colors=['r','b','g','orange']
    
    #plt.pie(sizes_predicted,labels=labels_predicted, colors=colors)
    #plt.title('predicted distribution')
    #plt.axis('equal')
    #if save:
        #if not os.path.exists("./plots/"):
            #os.makedirs("./plots/")
            #print("Creating folder plots")
        #plt.savefig("plots/"+fileName+"_predicted_pie_old.pdf")
        #plt.savefig("plots/"+fileName+"_predicted_pie_old.png")
        #plt.close()
        
    #plt.figure()
    #plt.pie(sizes_true,labels=labels_true, colors=colors)
    #plt.title('true distribution')
    #plt.axis('equal')
    #if save:
        #if not os.path.exists("./plots/"):
            #os.makedirs("./plots/")
            #print("Creating folder plots")
        #plt.savefig("plots/"+fileName+"_true_pie_old.pdf")
        #plt.savefig("plots/"+fileName+"_true_pie_old.png")
        #plt.close()
    