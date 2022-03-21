import numpy as np
import matplotlib.pyplot as plt

s = 1
t = np.sqrt(5)
# functions to calc distribution params based on X
def full_dist_params(X):
    n_D = sum(X)
    n = len(X)
    S = n_D*(t**2)/(s**2 + (n*(t**2)))
    M = s**2 + ((t**2)*(s**2))/((s**2) + (n*(t**2)))
    # return mean and standard deviation of distribution based on X
    return M, S
def mu_MLE(X):
    return np.mean(X)
def mu_MAP(X):
    return np.sum(X)/(len(X) + (s**2)/(t**2))

D = [3.3,3.5,3.1,1.8,3.0,0.74,2.5,2.4,1.6,2.1,2.4,1.3,1.7,0.19]
fig, axs = plt.subplots(7, 2, figsize=(15,40))
axs= axs.ravel()
# loop through 14 possible datasets and plot
for i in range(1,len(D)+1):
    X = np.array(D[:i])
    M,S = full_dist_params(X)
    mu_mle = mu_MLE(X)
    mu_map = mu_MAP(X)
    xx = np.linspace(min(D)-5,max(D)+5, 100)
    # full posterior
    pdf1 = (1/(S*np.sqrt(2*np.pi)))*np.exp(-0.5*(((xx-M)/S)**2))
    # mle posterior
    pdf2 = (1/(s*np.sqrt(2*np.pi)))*np.exp(-0.5*(((xx-mu_mle)/s)**2))
    # map posterior
    pdf3 = (1/(s*np.sqrt(2*np.pi)))*np.exp(-0.5*(((xx-mu_map)/s)**2))
    axs[i-1].plot(xx, pdf1, label='full posterior predictive')
    axs[i-1].plot(xx, pdf2, label='MLE predictive')
    axs[i-1].plot(xx, pdf3, label='MAP predictive')
    axs[i-1].legend();
    axs[i-1].set_xlabel('x value');
    axs[i-1].set_ylabel('density');
    axs[i-1].set_title(f'{i} data points');