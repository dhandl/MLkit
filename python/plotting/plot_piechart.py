import matplotlib.pyplot as plt
import numpy as np

def plot_pie_chart(y_true, y_predict, fileName="Test", save=False):
    print('Plotting pie charts...')    
    y_predict_class = np.argmax(y_predict, axis=1)
    sizes_predicted = np.bincount(y_predict_class)
    sizes_true = np.bincount(y_true.astype(int))
    labels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    
    plt.pie(sizes_predicted,labels=labels)
    plt.title('Vorhergesagte Verteilung')
    plt.axis('equal')
    if save:
        if not os.path.exists("plots/"):
            os.makedirs("plots/")
            print("Creating folder plots")
        plt.savefig(os.path.join(savedir,filename+'_predicted_pie.pdf'))
        plt.savefig(os.path.join(savedir,filename+'_predicted_pie.png'))
        plt.close()
        
    plt.figure()
    plt.pie(sizes_true,labels=labels)
    plt.title('Wahre Verteilung')
    plt.axis('equal')
    if save:
        if not os.path.exists("plots/"):
            os.makedirs("plots/")
            print("Creating folder plots")
        plt.savefig(os.path.join(savedir,filename+'_true_pie.pdf'))
        plt.savefig(os.path.join(savedir,filename+'_true_pie.png'))
        plt.close()
    