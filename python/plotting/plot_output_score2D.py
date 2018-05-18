import matplotlib.pyplot as plt
import numpy as np

def plot_output_score2D(output_score, X_test, xlabel, ylabel, save=False, fileName='Test'):
    #https://stackoverflow.com/questions/30509890/how-to-plot-a-smooth-2d-color-plot-for-z-fx-y?rq=1
    
    met = X_test[:,5]
    mt = X_test[:,4]
    
    bins = np.linspace(output_score.min(),output_score.max(), num = 50)
    output_score_bins = np.digitize(output_score, bins)
    output_score_binned = np.bincount(output_score_bins)
    print output_score_binned
    output_score_color = np.diag(output_score_binned)
    
    plt.contour(output_score_color)
    
    #xi, yi = np.linspace(met.min(), met.max(), 100), np.linspace(mt.min(), mt.max(), 100)
    #x, y = np.meshgrid(xi, yi)

    #fig, ax1 = plt.subplots(figsize=(8,6))

    ## Interpolate
    #rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((met, mt)).T, values=output_score)
    #zi = rbf(x, y)

    #im = ax1.imshow(zi, vmin=0., vmax=5., origin='lower',
                #extent=[met.min(), met.max(), mt.min(), mt.max()])
    #cbar = plt.colorbar(im)
    #cbar.set_label('Output Score')
    #ax1.set_xlabel(xlabel)
    #ax1.set_xlim([met.min(), met.max()])
    #ax1.set_ylabel(ylabel)
    #ax1.set_ylim([y.min(), y.max()])
    #plt.scatter(x, y, c='black')
    #plt.plot(x, x-84., color='black')
    #plt.plot(x, x-175., color='black')
    
    
    #output_score_color = output_score
    
    #plt.pcolor(x,y,output_score_color)
    #plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Output Score')
    
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_output_score2D.pdf")
        plt.savefig("plots/"+fileName+"_output_score2D.png")
        plt.close()