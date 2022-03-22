
import matplotlib.pyplot as plt
import numpy as np

# X
#xData = np.loadtxt('output/pca_scores_ORIGINAL_U.csv', dtype=str, delimiter=',')
#xData = np.loadtxt('output/pca_scores_ORIGINAL_S.csv', dtype=str, delimiter=',')
#xData = np.loadtxt('output/pca_scores_LR_S_M5_N83_R-AGING025-EIsupine.csv', dtype=str, delimiter=',')
xData = np.loadtxt('output/pca_scores_LRT_S_M5_N76_R-AGING025-EIsupine.csv', dtype=str, delimiter=',')

xLabel = "Lung only"  #"Original (n=83)"

# Y
#yData = np.loadtxt('output/pca_scores_LR_U_M5_N83_R-AGING025-EIsupine.csv', dtype=str, delimiter=',')
#yData = np.loadtxt('output/pca_scores_LR_S_M5_N83_R-AGING001-EIsupine.csv', dtype=str, delimiter=',')
#yData = np.loadtxt('output/pca_scores_LRT_S_M5_N76_R-AGING025-EIsupine.csv', dtype=str, delimiter=',')

yLabel = "Lung + Torso" #"New (n=83)"

#print(xData)
#print(yData)

## For sorting data... (currently must be sorted!)
## list of subjects for regression
#sbj_list = yData[:, 0]
## find indices of subjects in subject data
#sbj_idxs = []
#for s, sbj in enumerate(sbj_list):
#    #sbj = sbj.split('-')[-1]
#    where = np.where(sbj == xData[:, 0])
#    if len(where[0]) > 0:
#        sbj_idxs.append(int(where[0][0]))
## extract data for subjects
#xData = xData[sbj_idxs, :]
#
## Test if subjects match
#if np.all(xData[:,0]==yData[:,0]):
#    print("All subjects match.")
#else:
#    print(xData[:,0]==yData[:,0])
#    raise Exception("NOT ALL SUBJECTS MATCH!")
#
## PCA comparison
#n_modes = np.shape(xData)[1] - 1
#fig, axs = plt.subplots(2, 3)
#for m in range(1, n_modes+1):
#    #plt.figure()
#    ax = axs[int((m-1)/3), (m-1)%3]
#    ax.scatter(xData[:,m].astype(float), yData[:,m].astype(float)) #, marker='x', lw=0)
#    ax.plot([-3, 3], [-3, 3], color="black")
#    ax.set_title("Mode "+str(m))
#    ax.set_xlabel(xLabel)  #"Lung only")
#    ax.set_ylabel(yLabel)  #"Torso + Lung")
#    ax.set_aspect('equal', 'box')
#    plt.draw()
#
##fig.delaxes(axs[0, 3])
#fig.delaxes(axs[1, 2])
##fig.delaxes(axs[1, 3])
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#plt.tight_layout()
#plt.show()

# Modes comparison
n_modes = np.shape(xData)[1] - 1
fig, axs = plt.subplots(n_modes-1, n_modes-1)
for mi in range(1, n_modes+1):
    for mj in range(1, n_modes+1):
        print(mi, mj)
        if mj > mi:
            ax = axs[mi-1, mj-2]
            ax.scatter(xData[:,mj].astype(float), xData[:,mi].astype(float), s=1) #, marker='x', lw=0)
            ax.plot([-3, 3], [-3, 3], color="black")
            #ax.set_title("M"+str(mi)+" v M"+str(mj))
            ax.set_xlabel("M"+str(mj))  #"Lung only")
            if mj == mi+1:
                ax.set_ylabel("M"+str(mi))  #"Torso + Lung")
            #ax.set_aspect('equal', 'box')
            plt.draw()
            print("plotted")
        elif mj > 1 and mi < n_modes:
            print("deleted")
            fig.delaxes(axs[mi-1, mj-2])

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
#plt.tight_layout()
plt.show()


