import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import os

def plot_output_score2D(output_score, X_test, xlabel, ylabel, save=False, fileName='Test'):
    
    print 'Plotting the 2D output score...'
    
    output_score_threshhold = 0.
    print 'Threshold for Output Score:', output_score_threshhold
    
    x = X_test[:,5][output_score>=output_score_threshhold]*0.001 #met
    y = X_test[:,4][output_score>=output_score_threshhold]*0.001 #mt
    z = output_score[output_score>=output_score_threshhold]
    
    xi = np.linspace(100,1000,100) #met
    yi= np.linspace(90, 700,100) #mt
    xi, yi = np.meshgrid(xi,yi)
    
    rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((x, y)).T, values=z)
    zi = rbf(xi, yi)

    contours = plt.contour(xi, yi, zi, 4, colors='grey')
    #plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(zi, vmin=0., vmax=1., extent=[0, 1000, 0, 800], origin='lower', cmap='coolwarm', alpha=0.5)
    plt.colorbar()
    
    plt.xlim(100,1000)
    plt.ylim(90, 700)
    
    plt.xlabel(xlabel + '[GeV]')
    plt.ylabel(ylabel + '[GeV]')
    plt.title('Output Score')
    
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_output_score2D.pdf")
        plt.savefig("plots/"+fileName+"_output_score2D.png")
        plt.close()
        
#def plot_output_score2D(output_score, X_test, xlabel, ylabel, save=False, fileName='Test'):
    ##https://stackoverflow.com/questions/30509890/how-to-plot-a-smooth-2d-color-plot-for-z-fx-y?rq=1
    
    #print 'Plotting the 2D output score...'
    
    #averageN = 20 #Should be around 20 to prevent MemoryError
    
    #print 'Averaging over ' + str(averageN) + ' values...'
    
    #output_score_threshhold = 0.
    
    #print 'Threshold for Output Score:', output_score_threshhold
    
    #met = calcAverage(X_test[:,5][output_score>=output_score_threshhold]*0.001,averageN)
    #mt = calcAverage(X_test[:,4][output_score>=output_score_threshhold]*0.001,averageN)
    
    ##met = np.linspace(100e3,1000e3,100)
    ##mt = np.linspace(90e3, 800e3,100)
    
    #output_score_averaged = calcAverage(output_score[output_score>=output_score_threshhold],averageN)
    #output_score_color = np.diag(output_score_averaged)
    
    ##bins = np.linspace(output_score.min(),output_score.max(), num = 50)
    ##output_score_bins = np.digitize(output_score, bins)
    ##output_score_binned = np.bincount(output_score_bins)
    ##print output_score_binned
    ##output_score_color = np.diag(output_score_binned)
    
    #met,mt = np.meshgrid(met,mt)
    
    #plt.contourf(met, mt, output_score_color)
    ##plt.pcolor(met, mt, output_score_color) #Uncommenting this means pain and agony
    
    #plt.gca().set_xlim([met.min(), met.max()])
    #plt.gca().set_ylim([mt.min(),mt.max()])
    
    #plt.xlabel(xlabel + '[GeV]')
    #plt.ylabel(ylabel + '[GeV]')
    #plt.title('Output Score')
    #plt.colorbar()
    
    #if save:
        #if not os.path.exists("./plots/"):
            #os.makedirs("./plots/")
            #print("Creating folder plots")
        #plt.savefig("plots/"+fileName+"_output_score2D.pdf")
        #plt.savefig("plots/"+fileName+"_output_score2D.png")
        #plt.close()
        
def calcAverage(array, k):
    '''
    Averages over k elements in array, k must be an integer
    Returns averaged array with size len(array)/k, if possible
    '''
    array_averaged = []
    
    n = len(array)
    m = int(n)/int(k)
    if (m*k == n):
        lst = np.split(array,m)
        array_averaged = np.array([np.sum(x) for x in lst])/float(k)
    else:
        h = n - m*k
        array2 = np.delete(array,np.linspace(len(array)-h,len(array)-1,h))
        lst = np.split(array2,m)
        array3 = array[-h:]
        array_averaged = np.array([np.sum(x) for x in lst])/float(k)
        array_averaged = np.append(array_averaged, np.array(np.sum(array3))/float(len(array3)))
    return array_averaged
        