import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches

def plot_classification_2(y_true, y_predict, fileName="Test", save=False):
    print('Plotting the classification for predicted labels...')
    y_predict_class = np.argmax(y_predict, axis=1)
    classes = [0,1,2,3] #Different classes
    issignal = []
    istt = []
    issinglet = []
    isWjets = []
    
    explain_patch = mpatches.Patch(color='None', label="true label")
    
    for i in range(0,4):
        n = float(y_true[y_predict_class==i].shape[0])

        u, counts = np.unique(y_true[y_predict_class==i], return_counts=True)
        
        #print(u.tolist())
        #print(counts.tolist())

        try:
            issignal.append(counts[u.tolist().index(0)]/n)
        except ValueError:
            issignal.append(0)
        try:
            istt.append(counts[u.tolist().index(1)]/n)
        except ValueError:
            istt.append(0)
        try:
            issinglet.append(counts[u.tolist().index(2)]/n)
        except ValueError:
            issinglet.append(0)
        try:
            isWjets.append(counts[u.tolist().index(3)]/n)
        except ValueError:
            isWjets.append(0)    
            
    width=1.
    
    bar0 = plt.bar(classes, issignal, width, label=r'Signal')
    bar1 = plt.bar(classes, istt, width, bottom=issignal, label=r'$t\overline{t}$')
    bar2 = plt.bar(classes, issinglet, width, bottom=np.array(istt)+np.array(issignal), label=r'Single Top')
    bar3 = plt.bar(classes, isWjets, width, bottom=np.array(issinglet)+np.array(istt)+np.array(issignal), label='$W$ + jets')
    
    plt.xlabel('predicted label')
    #plt.legend(loc='best',handles=[explain_patch, bar0, bar1, bar2, bar3])
    plt.xticks(np.arange(4),(r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets'))
    plt.title('Classification')
    
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[explain_patch, bar0, bar1, bar2, bar3])
    
    #plt.gca().set_ylim([0,1.2])
    
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_Classification2.pdf")
        plt.savefig("plots/"+fileName+"_Classification2.png")
        plt.close()