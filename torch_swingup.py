#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.animation as animation

from IPython.display import HTML

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

from cartpole.cartpole_class import Cartpole 
from utils.nn_utils import make_histories, fit_model, normalize_data



# In[2]:


## Define some constants

# time vector, if you use the default solver: doesn't actually affect the integration, only what times it records our state variable at
dt = 0.1
tmax = 8.0
t_eval = np.arange(0.0, tmax, dt)

# Cartpole is a class we defined that takes care of the simulation/animation of the cartpole
bot = Cartpole()


# In[3]:


## Run a bunch of trials using the energy shaping controller
num_trials = 5 # This is the number of intital conditions to try, note the total number of trials is num_trials*num_trials

min_theta = 0
max_theta = 0

min_thdot = -1
max_thdot = 1

# we'll iterate through these two
theta_vals = np.linspace(min_theta, max_theta, num_trials)
thdot_vals = np.linspace(min_thdot, max_thdot, num_trials)

# and keep these two constant
x = 0
xdot = 0

states = np.zeros((len(t_eval), num_trials, num_trials, 4))
actions = np.zeros((len(t_eval), num_trials, num_trials, 1))


for i, theta in enumerate(theta_vals):
    for j, thdot in enumerate(thdot_vals):

        # initial state
        init_state = np.array([theta, x, thdot, xdot])
        
        # integrate the ODE (by default this is equivalent to ode45)
        sol = integrate.solve_ivp(bot.derivs, (0,tmax), init_state, t_eval = t_eval, max_step = .1)
        if not sol.success:
            print("warning: solver failed with intial conditions: ", init_state )
        
        # TODO think about doing this without a dimension per changing parameter..
        states[:,i,j,:] = sol.y.T
        
        #TODO, really don't like this
        for t in range(len(t_eval)):
            actions[t,i,j] = bot.control(0, states[t,i,j,:]) 
            


# In[4]:


# Animate the cart (optional) 
ani = bot.animate_cart(t_eval, states[:,0,0,:])
#HTML(ani.to_jshtml())


# In[10]:


# Define and train the feedforward network

ff_model = nn.Sequential(
    nn.Linear(4,12),
    nn.ReLU(),
    nn.Linear(12,12),
    nn.ReLU(),
    nn.Linear(12,1)
)

state_train = states.reshape(-1,4)
action_train = actions.reshape(-1,1)

loss_hist = fit_model(ff_model, state_train, action_train, num_epochs=50, learning_rate=1e-2)

plt.plot(loss_hist)
plt.title('simple model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()


# In[11]:


# Define and train the lookback network

look_back = 3

lb_model = nn.Sequential(
    nn.Linear(4 * look_back, 12*look_back),
    nn.ReLU(),
    nn.Linear(12 * look_back, 12*look_back),
    nn.ReLU(),
    nn.Linear(12 * look_back, 1)
)

state_lb = make_histories(states.reshape(-1,4), look_back)

state_train = state_lb.reshape(-1,look_back*4)
action_train = actions.reshape(-1,1)

loss_hist = fit_model(lb_model, state_train, action_train, num_epochs=50, learning_rate=1e-2)

plt.plot(loss_hist)
plt.title('simple model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()


# In[12]:


# This is a closure that returns our NN controller
# Might make sense to just make a subclass here rather than this closure thing.. not sure yet

def make_ff_controller(model):
    def nn_controller(t, q):
        if (q[0] < (140 * (pi/180)) ) or (q[0] > (220 * (pi/180)) ):
            # apply normalization
            state_train_norm, state_train_mean, state_train_std = normalize_data(state_train)
            q = (q-state_train_mean)/state_train_std
            return model(torch.tensor(q, dtype=torch.float32))
        else:
            # balancing
            # LQR: K values from MATLAB
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
            return u
        
    return nn_controller


def make_lb_controller(model):
    def nn_controller(q):
        if (q[0,look_back-1] < (140 * (pi/180)) ) or (q[0,look_back-1] > (220 * (pi/180)) ):
            u = model(torch.tensor(q.reshape(1,-1), dtype=torch.float32))
            return u[0][0]
        else:
            # balancing
            # lqr: k values from matlab
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0,look_back-1] - pi) + k2 * q[1,look_back-1] + k3 * q[2,look_back-1] + k4 * q[3,look_back-1])
            return u
        
    return nn_controller


ff_control = make_ff_controller(ff_model)
lb_control = make_lb_controller(lb_model)

ff_bot = Cartpole()
lb_bot = Cartpole(dt, Ts=1, n=3)

ff_bot.control = ff_control
lb_bot.control = lb_control

# initial conditions
theta = .4
x = 1
th_dot = .1 # an initial velocity, triggers the swing up control
xdot = 0.1
time = np.arange(0.0, 20, dt)

# initial state
init_state = np.array([theta, x, th_dot, xdot])


# In[13]:


sol = integrate.solve_ivp(ff_bot.derivs, (0,tmax), init_state, t_eval = t_eval)
y_ff = sol.y.T

u_ff = np.zeros(len(t_eval))
for t in range(len(t_eval)):
        u_ff[t] = bot.control(t, y_ff[t]) 

plt.figure()
plt.plot(y_ff[:,2])

ani = bot.animate_cart(t_eval, y_ff)
HTML(ani.to_jshtml())


# In[14]:


# Run the simulation for the Feedforward look back network

# integrate the ODE using scipy.integrate.
#y_lb = integrate.odeint(lb_bot.derivs_dig_lb, init_state, time)

sol = integrate.solve_ivp(lb_bot.derivs_dig_lb, (0, tmax), init_state, t_eval = t_eval)
y_lb = sol.y.T

u_lb = np.zeros(len(t_eval))
for t in range(len(t_eval)):
        u_lb[t] = bot.control(t, y_lb[t]) 


plt.figure()
plt.plot(y_lb[:,2])

ani = lb_bot.animate_cart(time, y_lb)
HTML(ani.to_jshtml())


# In[ ]:


# TODO should add in normalization eventually


# Previous manual normalization in comment block
# =============================================================================
# y_train_mean = [y_train[:,i].mean() for i in range(y_train[0,:])]
# u_train_mean = [u_train[:,i].mean() for i in range(u_train[0,:])]
# 
# y_train_std = [y_train[:,i].std() for i in range(y_train[0,:])]
# u_train_std = [u_train[:,i].std() for i in range(u_train[0,:])]
# 
# 
# for i in range(len(y_train[0,:])):
#     y_train[:,i] = (y_train[:,i] - y_train[:,i].mean())/y_train[:,i].std()
# 
# for i in range(len(u_train[0,:])):
#     u_train[:,i] = (u_train[:,i] - u_train[:,i].mean())/u_train[:,i].std()
# =============================================================================

