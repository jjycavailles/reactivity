#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from pycalphad.plot import triangular


# In[2]:


Color = ['red', 'green', 'blue']
Label = ["A", "B", "R"]


# In[3]:


def u_A_expected(X, param):
    xA, xB, xR = X
    alpha, l, h, = param
    if(xA+xR == 0 or xA == 0):
        return np.infty
    else:
        return alpha*1/(xA+xR) + (1-alpha)*1/(xA)


# In[4]:


def u_B_expected(X, param):
    xA, xB, xR = X
    alpha, l, h = param
    if(xB+xR == 0 or xB == 0):
        return np.infty
    else:
        return alpha*l/(xB) + (1-alpha)*h/(xB+xR)


# In[5]:


def u_R_expected(X, param):
    xA, xB, xR = X
    alpha, l, h = param
    if(xA+xR == 0 or xB+xR == 0):
        return np.infty
    else:
        return alpha*1/(xA+xR) + (1-alpha)*h/(xB+xR)


# In[6]:


def replicator_dynamics_expected(X, param):
    dX = np.zeros(3)
    uA = u_A_expected(X, param)
    uB = u_B_expected(X, param)
    uR = u_R_expected(X, param)
    U = np.array([uA, uB, uR])
    average = X.dot(U) #/ n
    for i in range(3):
        dX[i] = X[i]*(U[i] - average)
    return dX


# In[7]:


# realized
      
def u_A(X, param, Xb):
    xA, xB, xR = X
    alpha, l, h = param
    if(abs(Xb-l) <= 10**-10):
        if(xA+xR == 0):
            return np.infty
        else:
            return 1/(xA+xR)
    elif(abs(Xb-h) <= 10**-10):
        if(xA == 0):
            return np.infty
        else:
            return 1/(xA)      

def u_B(X, param, Xb):
    xA, xB, xR = X
    alpha, l, h = param
    if(abs(Xb-l) <= 10**-10):
        if(xB == 0):
            return np.infty
        else:
            return l/(xB)
    elif(abs(Xb-h) <= 10**-10):
        if(xB+xR == 0):
            return np.infty
        else:
            return h/(xB+xR)
        
def u_R(X, param, Xb):
    xA, xB, xR = X
    alpha, l, h = param
    if(abs(Xb-l) <= 10**-10):
        if(xA+xR == 0):
            return np.infty
        else:
            return 1/(xA+xR)
    elif(abs(Xb-h) <= 10**-10):
        if(xB+xR == 0):
            return np.infty
        else:
            return h/(xB+xR)    

def replicator_dynamics(X, param, Xb):
    dX = np.zeros(3)
    uA = u_A(X, param, Xb)
    uB = u_B(X, param, Xb)
    uR = u_R(X, param, Xb)
    U = np.array([uA, uB, uR])
    average = X.dot(U)# / n
    for i in range(3):
        dX[i] = X[i]*(U[i] - average)
    return dX


# In[8]:


def RK4(X, param, dt, Xb, Xb1, expected = False):
    if(expected):
        k1 = replicator_dynamics_expected(X, param)
        k2 = replicator_dynamics_expected(X+dt/2*k1, param)
        k3 = replicator_dynamics_expected(X+dt/2*k2, param)
        k4 = replicator_dynamics_expected(X+dt*k3, param)
    else:
        k1 = replicator_dynamics(X, param, Xb)
        k2 = replicator_dynamics(X+dt/2*k1, param, Xb)
        k3 = replicator_dynamics(X+dt/2*k2, param, Xb)
        k4 = replicator_dynamics(X+dt*k3, param, Xb1)
    return X + dt/6*(k1+k2+k3+k4)


# In[9]:


def iteration(X0, T, param, XXb, expected = False):
    dt = T[1] - T[0]
    XX = np.zeros((len(T), 3))
    XX[0] = X0
    for i in range(len(T)-1):
        XX[i+1] = RK4(XX[i], param, dt, XXb[i], XXb[i+1], expected=expected)
    return XX


# In[10]:


def best_reponse(X, param):
    return np.argmax(np.array([u_A_expected(X, param), u_B_expected(X, param), u_R_expected(X, param)]))


# In[11]:


def worst_reponse(X, param):
    return np.argmin(np.array([u_A_expected(X, param), u_B_expected(X, param), u_R_expected(X, param)]))


# In[12]:


def der_best(param):
    nbre_points = 101
    XXR = np.linspace(0, 1, nbre_points)
    XXB = np.linspace(0, 1, nbre_points)
    Best = np.zeros((nbre_points, nbre_points)) - 1
    Best[:] = np.NAN
    for i_r, xR in enumerate(XXR):
        for i_b, xB in enumerate(XXB):
            if(xR+xB<1):
                xA = 1-xR-xB
                X = xA, xB, xR
                Best[i_r, i_b] = best_reponse(X, param)
    return Best


# In[13]:


def der_worst(param):
    nbre_points = 101
    XXR = np.linspace(0, 1, nbre_points)
    XXB = np.linspace(0, 1, nbre_points)
    Worst = np.zeros((nbre_points, nbre_points)) - 1
    Worst[:] = np.NAN
    for i_r, xR in enumerate(XXR):
        for i_b, xB in enumerate(XXB):
            if(xR+xB<1):
                xA = 1-xR-xB
                X = xA, xB, xR
                Worst[i_r, i_b] = worst_reponse(X, param)
    return Worst

np.arange(0.2, 2, 0.8)
# In[14]:


def plot_response(param, Best, show = False, save = False):
    alpha, l, h = param
    nbre = len(Best)
    if(show):
        fig = plt.figure(figsize = (15, 10))
        fig.gca(projection='triangular')
    XXR = np.linspace(0, 1, nbre)
    XXB = np.linspace(0, 1, nbre)
    RR, BB = np.meshgrid(XXR, XXB)
    cmap = colors.ListedColormap(['red', 'green', 'blue'])
    CS2 = plt.contourf(BB, RR, Best, cmap = cmap, origin="lower")
    plt.xticks([0, 1], ["A", "R"], fontsize = 30)
    plt.yticks([1], ["B"], fontsize = 30)
    cbar = plt.colorbar()
    #cbar.set_ticks(np.arange(0.4, 3, 0.8))
    cbar.set_ticks(np.array([0.4, 1. , 1.6]))
    cbar.set_ticklabels(["A", "B", "R"])
    cbar.ax.tick_params(labelsize=20)
    
    xA = np.round(1./(1+h), 2)
    xR = np.round((h-l)/((1+h)*(1+l)), 2)
    xB = np.round(l/(1+l), 2)
    """
    plt.annotate("(A,B,R)=("+str(xA)+", "+str(xB)+","+str(xR)+")",
            xy=(xR, xB), xycoords='data',
            xytext=(0.1, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top', fontsize = 15)
    """
    
    if(save):
        plt.savefig(folder+title+".png")
    if(show):
        plt.show()


# In[15]:


def plot_time_series(XXeq, T, param, XXb, show=False, XXeq_expected=None):
    alpha, l, h = param
    #plt.figure(figsize = (10, 7))
    if(show):
        for i in range(3):
            plt.plot(T, XXeq[:,i], "-", label = Label[i], color = Color[i])
    else:
        for i in range(3):
            plt.plot(T, XXeq[:,i], "-", color = Color[i])
    
    if(XXeq_expected is not None):
        #print("expected time series!")
        for i in range(3):
            plt.plot(T, XXeq_expected[:,i], "--", color = Color[i])
        
    #mmax = 1.1*np.max(np.max(XXeq))
    plt.fill_between(T, -0.1 , 1.1, where=XXb==h, alpha = 0.2, color = "purple", label = "Xb = high")
    plt.fill_between(T, -0.1 , 1.1, where=XXb==l, alpha = 0.2, color = "orange", label = "Xb = low")
    plt.ylim(-0.1, 1.1)
    
    plt.yticks([0, 1, 1./(1+h), float(l)/(1+l), (h-l)/((1+h)*(1+l))], ["0", "1", "1/(1+h)", "l/(1+l)", "(h-l)/((1+h)*(1+l))"], fontsize = 20)
        
    plt.legend(fontsize = 20)
    plt.xlabel("time", fontsize = 20)
    plt.ylabel("frequencies", fontsize = 20)
    if(show):
        plt.show()
    return


# In[16]:


def u(x_mix, x_other, param):
    U = np.array([u_A_expected(x_other, param), u_B_expected(x_other, param), u_R_expected(x_other, param)])
    return x_mix.dot(U)


# In[17]:


def plot_ess(param, nbre_points = 101, show = False, save = False):
    alpha, l, h = param
    x_NE = np.array([1./(1+h), float(l)/(1+l), (h-l)/((1+h)*(1+l))])
    base = 10
    nbre_points = 100
    XXR = np.linspace(0, 1, nbre_points)
    XXB = np.linspace(0, 1, nbre_points)
    
    Diff = np.zeros((nbre_points, nbre_points)) - 1
    Diff[:] = np.NAN
    for i_r, xR in enumerate(XXR):
        for i_b, xB in enumerate(XXB):
            if(xR+xB<1):
                xA = 1-xR-xB
                X = np.array([xA, xB, xR])
                Diff[i_r, i_b] = u(x_NE, X, param) - u(X, X, param)
                if(Diff[i_r, i_b] < 0):
                    print(X, Diff[i_r, i_b])
                    print(u(x_NE, X, param), u(X, X, param))
    
    if(show):
        fig = plt.figure(figsize = (15, 10))
        fig.gca(projection='triangular')


    RR, BB = np.meshgrid(XXR, XXB)
    norm=colors.SymLogNorm(linthresh = 0.0000001, base=10)
    CS2 = plt.contourf(BB, RR, Diff, locator=plt.LogLocator(base))
    fmt = ticker.LogFormatterMathtext(base)
    #fmt.create_dummy_axis()
    #plt.clabel(CS2, CS2.levels, fmt=fmt)
    
    plt.xticks([0, 1], ["A", "R"], fontsize = 30)
    plt.yticks([1], ["B"], fontsize = 30)
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('u(x_NE, X) - u(X, X)\n\n', rotation=270, fontsize = 15)
    
    #title = "alpha = "+str(alpha)+", l="+str(l)+", h="+str(h)+", c="+str(c)
    #plt.title(title)
    if(save):
        plt.savefig(folder+title+".png")
    if(show):
        plt.show()


# ### Phase space

# In[18]:


def plot_phase_space(param, show=False, sub=None):
    if(show):
        ax = plt.figure(figsize = (15, 10)).add_subplot(projection='3d')
    R = np.linspace(0, 1, 15)
    B = np.linspace(0, 1, 15)
    #R = np.linspace(0, 1, 4)
    #AA, BB, RR = np.meshgrid(A, B, R)
    RR, BB = np.meshgrid(R, B)
    Da = np.zeros_like(RR)
    Db = np.zeros_like(BB)
    Dr = np.zeros_like(BB)
    #Dr = np.zeros_like(RR)
    #C = np.linspace(0,1,121)[::1]

    for i, r in enumerate(R):
        for j, b in enumerate(B):
            if(r+b<1):
                #for k, r in enumerate(R):
                    #if(a+b+r<=1):
                X = np.array([1-b-r, b, r])
                #Da[j,i,k], Db[j,i,k], Dr[j,i,k] = replicator_dynamics_expected(X, param)
                Da[i,j], Db[i,j], Dr[i,j] = replicator_dynamics_expected(X, param)
    if(show):
        q = ax.quiver(BB, RR, Dr-Da/2**0.5, Db-Da/2**0.5, length=5, normalize=True)#, cmap="rainbow")
    else:
        plt.quiver(BB, RR, Dr-Da/2**0.5, Db-Da/2**0.5)#, cmap="rainbow")
    
    plt.xticks([0, 1], ["A", "R"], fontsize = 30)
    plt.yticks([1], ["B"], fontsize = 30)
    
    #plt.plot([XXeq[-1,0]], [XXeq[-1,1]], [XXeq[-1,2]], "*r")
    #plt.legend()
    #title = "alpha = "+str(alpha)+", l="+str(l)+", h="+str(h)+", c="+str(c)
    #plt.title(title, fontsize = 20)
    #plt.savefig("phase plane (3d)"+title+".png")
    if(show):
        plt.show()
    return


# ### Parameter space

# In[19]:


def strategy(l, h):
    return np.array([1./(1+h), l/(1+l), (h-l)/((1+h)*(1+l))])


# In[20]:


def plot_parameter_space(param, show=False, sub=None):
    alpha, lambd, eta = param
    size_l = size_h = size = 200

    nn = 4

    L = np.linspace(0, nn+1, size_l)
    H = np.linspace(0, nn+1, size_h)

    Strat = np.zeros((size_l, size_h, 3))+1
    #fig, ax = plt.subplots(figsize=(5, 5))
    m = nn*size
    coef = 1./nn*size
    
    if(show):
        fig, ax = plt.figure()
    else:
        fig = sub
    plt.plot([0, m], [0, m], color = "black")
    plt.plot([], [], linewidth = 10, color = "red", label = "A")
    plt.plot([], [], linewidth = 10, color = "green", label = "B")
    plt.plot([], [], linewidth = 10, color = "blue", label = "R")

    for i_l, ll in enumerate(L):
        for i_h, hh in enumerate(H):
            if(hh>=ll):
                Strat[i_h, i_l, :] = strategy(ll, hh)

    plt.plot([lambd*coef, lambd*coef], [0,eta*coef], "--", color = "black")
    plt.plot([0, lambd*coef], [eta*coef,eta*coef], "--", color = "black")

    plt.imshow(Strat, origin="lower")

    #plt.xlim(0, m/4)
    #plt.ylim(0, m/4)

    #plt.legend(loc = "lower right", fontsize = 20)

    plt.xlabel("l", fontsize = 20)
    plt.ylabel("h", fontsize = 20)#, rotation = 3.14/2)

    plt.xticks([coef, lambd*coef], ["1", str(np.round(lambd, 2))], fontsize = 20)
    plt.yticks([coef, eta*coef], ["1", str(np.round(eta, 2))], fontsize = 20)
    
    A, B, R = np.around(strategy(lambd, eta), 2)
    plt.annotate("(A,B,R)=("+str(A)+", "+str(B)+","+str(R)+")",
            xy=(lambd*coef, eta*coef), xycoords='data',
            xytext=(0.7, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top', fontsize = 15)

    sub.spines['right'].set_visible(False)
    sub.spines['top'].set_visible(False)
    
    return


# # Plot

# In[21]:

"""
alpha = 0.4
l = 1
h = 2
param = alpha, l, h
nbreStep = 1001
tFinal = 100
T = np.linspace(0, tFinal, nbreStep)


# In[22]:


# n = infinity
x_NE = np.array([1./(1+h), float(l)/(1+l), (h-l)/((1+h)*(1+l))])
x_NE


# # Compute

# In[23]:


Best = der_best(param)
Worst = der_worst(param)

X0eq = np.array([1./3, 1./3, 1./3])
period_variable_source = 10
Random_series_simple = np.random.binomial(1, alpha, size = nbreStep//period_variable_source) # generate again only if alpha changes
Random_series = np.zeros(nbreStep)
for i in range(len(Random_series_simple)):
    Random_series[i*period_variable_source:i*period_variable_source+period_variable_source:] = Random_series_simple[i]

XXb = l + (h-l)*Random_series
XXeq = iteration(X0eq, T, param, XXb)
XXeq_expected = iteration(X0eq, T, param, XXb, expected = True)


# # Plots

# In[26]:


fig, ax = plt.subplots(3, 2, figsize = (20, 30))

plt.subplot(3,2,1)
plt.title("Time series", fontsize = 30)
plot_time_series(XXeq, T, param, XXb, show=False, XXeq_expected=XXeq_expected)


plt.subplot(3,2,2)
sub = ax[1,0]
plot_parameter_space(param, show = False, sub=sub)
plt.title("Parameter space", fontsize = 30)


plt.subplot(3,2,3, projection='triangular')
#plt.subplot(3,2,3)
plot_response(param, Best, save = False, show = False)
plt.title("Best response", fontsize = 30)

plt.subplot(3,2,4, projection='triangular')
plot_ess(param, save = False, show = False)
plt.title("ESS", fontsize = 30)

#sub = plt.subplot(3,2,5)
sub = plt.subplot(3,2,5, projection='triangular')
plot_phase_space(param, show=False, sub=sub)
plt.title("Phase space", fontsize = 30)

plt.subplot(3,2,6)
plt.axis('off')
plt.show()

#plot_response(Worst, save = False, show = True)
"""
