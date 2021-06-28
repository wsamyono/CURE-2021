import numpy as np
# from math import log
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from RegscorePy import *
from scipy.interpolate import interp1d

# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
# data = np.loadtxt('datameagancontroljustindata.txt',delimiter=',')
data = np.loadtxt('dataskyler.txt',delimiter=',')
# data = np.loadtxt('HonourControlData.txt')
#data = np.loadtxt('AuNPsTreatmentHounorA.txt', skiprows=1, max_rows=5, dtype={'names': ('Concentration', 
#                     'Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'),
#                     'formats': ('S1', 'f4' , 'f4', 'f4', 'f4', 'f4', 'f4')})
# data = np.loadtxt('LM2-4LUC.txt',skiprows=1)
# u0 = data[0,1]
# yp0 = data[0,1]
# t = data[:,0].T - data[0,0]
# ID = data[:,0]
# print(ID)
# data = data.T
print("print data :\n", data)
t = data[:,0]
print(t)
# u = data[:,1].T
#yp = np.log(data[:,1])
yp = np.log(data[:,1])
print(yp)

# specify number of steps
ns = len(t)
# delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
# yp = interp1d(t,yp)

# define first-order for the Betalantfyy model    
# def fopdt(y,t,x,alpha1,alpha2,beta)
def fopdt(y,t,x,alpha1,beta):
    # arguments
    #  y      = output
    #  t      = time
    #  x      = input linear function (for time shift)
    #  alpha1 and alpha2  = coeficient for model gain
    #  beta   = model loss or dead/kill
    #  
    alpha1 = x[0]
    beta = x[1]
  #  try:
#    if t <= 4.0:
    dydt = alpha1*y**(2/3)-beta*y
#    else: 
#        if t <= 5.0:
#               alpha2 = x[1]
#                dydt = alpha2*y**(2/3)-beta*y 
#        else:
#                Km3 = x[2]
#                dydt = Km3*y**(2/3)-beta*y
  #  except:
    #    print('Error with time extrapolation: ' + str(t))
    #    um = 0
    # calculate derivative
    #    dydt = Km*y                # (-(y-yp0))*Km # + Km * (um-u0))/taum
    return dydt

# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x):
    # input arguments
    alpha1 = x[0]
#    alpha2 = x[1]
  #  Km3 = x[2]
    beta = x[1]
    # taum = x[1]
    # thetam = x[2]
    # storage for model values
    ym = np.zeros(ns)  # model
    # initial condition
#    ym[0] = 930000
    ym[0] = yp[0]

    # loop through time steps    
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
  #              y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
  #      y1 = odeint(fopdt,ym[i],ts,args=(x,alpha1,alpha1,beta))
        y1 = odeint(fopdt,ym[i],ts,args=(x,alpha1,beta))
        ym[i+1] = y1[-1]
    return ym

# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i]-yp[i])**2    
    # return result
    return obj

# initial guesses
x0 = np.zeros(4) #                    Nelder- Mead      
x0[0] = 0.01 # alpha1 = 30.0             15.0               
# x0[1] = 1.0 # alpha2 = -130.0           30.0               
# x0[2] = 15.0 # alpha33 = NA             NA                 
x0[1] = 0.01 # beta 
# x0[2] = 0.5 #beta = -0.5             1.0                
#       SSE    3.112692145853754e-07  1.0600646072056463e-07 
# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))
# print('alpha01: ' + str(x0[0]),', alpha01: ' + str(x0[1]))
print('alpha01: ' + str(x0[0]))
# ' and Kp03: ' + str(x0[2]))
print(' beta0: ' + str(x0[1]))
# optimize Km, taum, thetam
ym1 = sim_model(x0)
# ym2 = sim_model(x)
plt.plot(t,yp,'ok',linewidth=2,label='Experiment Control Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
# plt.plot(t,ym2,'r--',linewidth=3,label='Optimized Model')

plt.xlabel('Days')
plt.ylabel('Number of Cells')
plt.title('Best Fit Model with Control Data - Bellatanffy')
plt.legend(loc='best')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.show()

# solution = minimize(objective,x0,options={'xtol': 1e-8, 'maxfev':10000,'disp': True})
solution = minimize(objective,x0,method='nelder-mead',
               options={'xatol': 1e-8, 'maxfev':500000,'disp': True})
# solution = minimize(objective,x0,method='powell',
#               options={'xtol': 1e-8, 'maxfev':10000,'disp': True})

# Another way to solve: with bounds on variables
#bnds = ((0.4, 0.6), (1.0, 10.0), (0.0, 30.0))
#solution = minimize(objective,x0,bounds=bnds,method='SLSQP')
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))
print('apha1: ' + str(x[0]))
#,' and Kp3: ' + str(x[2]))
print(' beta: ' + str(x[1]))

# print('taup: ' + str(x[1]))
# print('thetap: ' + str(x[2]))

# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x)

p = 2  # number of parameters
print(" AIC : " + str(aic.aic(yp, ym2, p)))
print(" BIC : " + str(bic.bic(yp, ym2, p)))
print(' SSE : ' + str(objective(x)))
# print("AIC =",aicnum)
# plot results
# plt.figure(1)
# plt.subplot(2,1,1)
#plt.plot(t,np.log(yp),'ok',linewidth=2,label='Experiment Control Data')
#plt.plot(t,np.log(ym1),'b-',linewidth=2,label='Initial Guess')
#plt.plot(t,np.log(ym2),'r--',linewidth=3,label='Optimized Model')
plt.plot(t,yp,'ok',linewidth=2,label='Experiment Control Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized Model')

plt.xlabel('Days')
plt.ylabel('Number of Cells')
plt.title('Best Fit Model with Control Data - Bellatanffy')
plt.legend(loc='best')
#plt.subplot(2,1,2)
#plt.plot(t,x[0],'bx-',linewidth=2)
#plt.plot(t,x[1],'r--',linewidth=3)
# plt.legend(['Measured','Interpolated'],loc='best')
# plt.ylabel('Input Data')
data = np.vstack((t,yp,ym2,)) # vertical stack
data = data.T              # transpose data
np.savetxt('outputdatamd6control.txt',data,delimiter=',')
plt.savefig('outputmodel62Kmbeta.png',dpi=300, bbox_inches='tight')
plt.show()