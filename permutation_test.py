import numpy as np
import matplotlib.pyplot as plt
import datetime
from utils import circular_diff

# Given a list of trajectories, where each trajectory has a true orientation and predicted orientation:
# 1. Rearrange the lists so you have a list of
#    * the true values: indiv_true_values
#    * the predicted values (from the model): indiv_predicted_values
# This will do a very conservative permutation/bootstrapping/shuffling test
# The black distribution is the null distribution of RMSE values by permuting true/predicted, and bootstrapping within each permutation.
# The red distribution is the observed RMSE distribution
# The red line is the observed mean RMSE

# If the red line is far from the black distribution (on the left) then the model is a signficiantly better predictor than what would be possible from chance. 
# This can be turned into a pval, here using a 2-tailed pvalue to be conservative. But a left one-tailed pvalue would be more accurate. 

ann65_train = np.load('ANN65_trainperformance_errorpeaks.npy')
ann66_train = np.load('ANN66_trainperformance_errorpeaks.npy')
ann67_train = np.load('ANN67_trainperformance_errorpeaks.npy')

ann67_heading_success = np.load('ANN67_n5000randperformance_successful_postodor_heading.npy')
ann67_heading_unsuccess = np.load('ANN67_n5000randperformance_unsuccessful_postodor_heading.npy')

ann65_train = (4+ann65_train)*np.pi/4
ann66_train = (4+ann66_train)*np.pi/4
ann67_train = (4+ann67_train)*np.pi/4

ann67_heading_success = ann67_heading_success*np.pi/4
ann67_heading_unsuccess = ann67_heading_unsuccess*np.pi/4

ann65_train_list = [ann65_train[45*x:45*(x+1)] for x in range(30720)]
ann66_train_list = [ann66_train[45*x:45*(x+1)] for x in range(30720)]
ann67_train_list = [ann67_train[45*x:45*(x+1)] for x in range(30720)]



indiv_true_values = ann67_train_list
indiv_predicted_values = ann66_train_list

start = datetime.datetime.now()

# Helper function:
def angle_distance(angle1, angle2):
    """
    Calculate the minimum distance between two angles.
    """
    diff = angle1 - angle2
    # Wrap to [-π, π]
    distance = np.arctan2(np.sin(diff), np.cos(diff))
    return distance

fig = plt.figure()
ax = fig.add_subplot(111)

# real rmse distribution
real_rmses = []
for i in range(len(indiv_true_values)):
    err = angle_distance(indiv_true_values[i], indiv_predicted_values[i])  ## orig. floris
    # err = indiv_predicted_values[i] - indiv_true_values[i]          ## CEM addition
    # err = circular_diff(indiv_predicted_values[i],indiv_true_values[i])  ## CEM addition
    rmse = np.sqrt(np.mean(err**2))
    real_rmses.append(rmse)
    if i%10000==0:
        print(i)
ax.hist(real_rmses, density=True, color='red', bins=8, alpha=0.2) #np.arange(-0.5,8.5)
ax.vlines(np.mean(real_rmses), 0, 1.5, linestyles='--', color='red')


test_rmses = []
for i in range(len(indiv_true_values)):
    err = angle_distance(indiv_true_values,indiv_predicted_values)
    rmse = np.sqrt(np.mean(err**2))
    test_rmses.append(rmse)


# bootstrapped rmse shuffle
# real rmse distribution
null_rmses = []
for i in np.arange(0,len(indiv_true_values),50):
# for i in range(len(indiv_true_values)):
    for j in np.arange(0,len(indiv_predicted_values),50):
    # for j in range(len(indiv_predicted_values)):
        #for k in range(10):
        # print([indiv_true_values[i]])
        a = np.random.permutation(indiv_true_values[i])
        b = np.random.permutation(indiv_predicted_values[j])
        m = np.min([len(a), len(b)])
        err = angle_distance(a[0:m], b[0:m])  ## orig. floris
        # err = b[0:m] - a[0:m]                  ## CEM addition
        # err = circular_diff(b[0:m],a[0:m])      ## CEM addition
        rmse = np.sqrt(np.mean(err**2))
        null_rmses.append(rmse)
        if (j%1000==0) & (i%10==0):
            print('bootstrapping '+str(j)+', i='+str(i))
_ = ax.hist(null_rmses, density=True, color='black', bins=8)#np.arange(-0.5,8.5))

# Calculate a two-tailed p-value
null_distribution = null_rmses
observed = np.mean(real_rmses)
p_value = np.mean(np.abs(null_distribution - np.mean(null_distribution)) >= 
                  np.abs(observed - np.mean(null_distribution)))
print('p = '+str(p_value))
end = datetime.datetime.now()

print('Time taken: '+str(end-start))