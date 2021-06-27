from numpy import *
import numpy as np
import matplotlib.pyplot as plt

kb=0.00831 #Units:[amu*nm^2/ps^2]
c=3*10**5  #Units:[nm/ps=1000m/s]
# Using molecular units:
# ps, nm, amu, K (see https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html)
# macroscopic SI will be prone to give larger numerical errors as we work
# with very small or large numbers



#this function returns a list with both the position and. velocity of a trajectory following the Langevin equation

def xv(dt,iterations,k,m,gamma,T):
    
    position=np.zeros(iterations)
    #Generate initial position from Boltzmann distribution
    position[0]=sqrt((kb*T)/k)*np.random.normal(0.0,1)
    
    velocity=np.zeros(iterations)
    #Generate initial velocity from Boltzmann distribution
    velocity[0]=sqrt((kb*T)/m)*np.random.normal(0.0,1)

    gaussian_ctt=sqrt(2*kb*T*gamma/dt/m) #TLC: I added 1/dt correcting the units
    # TLC: I moved the m prefactor from the integration below to the equation
    # above. It should be inside the square root (a factor sqrt(m) from 8.65)
    # and 1/m from changing from force to acceleration

    for i in range(1,len(velocity)): #numerical solve of x and v by Euler integration method
        velocity[i]=velocity[i-1]-((k/m)*position[i-1]+gamma*velocity[i-1]
                    -gaussian_ctt*np.random.normal(0.0,1))*dt
        position[i]=position[i-1]+0.5*(velocity[i-1]+velocity[i])*dt

    return position, velocity 
    
    
# Realistic parameters are given, in molecular dynamics units, deriving from the wavenumber 150cm^-1 and a mass
# of the chlorophyll of approximately 900amu
# dt is kept the same and the number of iterations determine the length of the trajectory
# the gamma is equal to 2interactions per picosecond (ps^-1)

dt=0.004
iterations=2500
m=900
wavenumber=145e-7
w=2*np.pi*c*wavenumber
k=w**2*m
gamma=5
T=300


# The previous function is called and both trajectories are plotted.


def timelength(dt,iterations):
    t=np.linspace(0,dt*iterations,iterations)
    return t
t_axis=timelength(dt,iterations)

f=xv(dt,iterations,k,m,gamma,T) #trajectory for the d.o.f. associated to the coupling
f_x=f[0] #position array from previous function
f_v=f[1] #velocity array from previous function

gamma_e=50 #this gamma applies to the d.o.f. associated to the energy sites
h=xv(dt,iterations,k,m,gamma_e,T)
h_x=h[0]
h_v=h[1]


fig, (pj, pe) = plt.subplots(2, sharex=True,figsize=(10,6)) #both plots to share the x-axis
pj.plot(t_axis, h_x,'b',label='Oscillations from d.o.f linked to E sites')
pj.set_ylabel('Position (nm))')
pj.legend(loc='best')

pe.plot(t_axis, f_x,'y',label='Oscillations from d.o.f linked to J') 
pe.set_ylabel('Position (nm)')
pe.legend(loc='best')


plt.xlabel('Time (ps)')
plt.show()



# Numerical Autocorrelation : double for loop to determine the correlation manually. Not normalized.

def autocorrelate(iterations,f):
    autocorrelation=[]
    
    for shift in range(iterations):
        if shift==0:
            c1=f
        else:
            c1=f[:-shift]  #shift the array of the position by the corresponding time lag up until iterations-1,
        c2=f[shift:]       #where both lists will have only one element
    
        coeff=0
        for i,j in zip(c1,c2): #sum over all (lagged) elements of both lists (sum of f_x[i]*f_x[i+shift])
            coeff+=i*j
        autocorrelation.append(coeff/(iterations-shift)) #divide the coefficient by the number of terms summed.
    
    return autocorrelation



# Analytical Autocorrelation: Plotting the real part of the correlation function analytically from the formula in
# Mukamel 8.69 for high-frequency underdamped modes. Also not normalized.

C=[] # Coupling trajectories
for i in t_axis:
    C.append(((kb*T)/(m*w**2))*(np.cos(sqrt(w**2-(gamma**2)/4)*i)+\
                        (gamma/(2*sqrt(w**2-(gamma**2)/4)))*np.sin(sqrt(w**2-(gamma**2)/4)*i))*\
                        np.exp((-gamma*i)/2)
)

C=np.array(C)
norm=np.linalg.norm(C)

D=[] # Energy trajectories
for i in t_axis:
    D.append(((kb*T)/(m*w**2))*(np.cos(sqrt(w**2-(gamma_e**2)/4)*i)+\
                        (gamma_e/(2*sqrt(w**2-(gamma_e**2)/4)))*np.sin(sqrt(w**2-(gamma_e**2)/4)*i))*\
                        np.exp((-gamma_e*i)/2)
)
D=np.array(D)


plt.plot(t_axis,autocorrelation_x,'b',label='Coupling\'s d.o.f correlation: Numerical solution')
plt.plot(t_axis,C,'r',label='Coupling\'s d.o.f correlation: Analytical solution')
plt.xlim(right=5/gamma,left=0)
plt.legend(loc='best')
plt.xlabel('Time [ps]')
plt.ylabel('Correlation')
plt.show() #comparing Numerical & Analytical solutions

autocorrelation_xj=autocorrelate(iterations,h_x)
plt.plot(t_axis,autocorrelation_xj,'b',label='Energy site\'s d.o.f correlation: Numerical solution')
plt.plot(t_axis,D,'r',label='Energy site\'s d.o.f correlation: Analytical solution')
plt.xlim(right=5/gamma,left=0)
plt.ylim(bottom=-1e-6)
plt.legend(loc='best')
plt.xlabel('Time [ps]')
plt.ylabel('Correlation')
plt.show()




# WE MOVE TO QUANTUM DYNAMICS, and we create much longer trajectories to then divide them into sections
# for different realizations that are averaged out to determine the coherence evolution.

iterations=2500000
gamma_e=50 #fluctuations of energy are due to a more overwhelming damping in the molecule
#previous langevin equations describe fluctuations in more specific degrees of freedom that impact the J term


f_xj=xv(dt,iterations,k,m,gamma,T)[0]
f_xe1=xv(dt,iterations,k,m,gamma_e,T)[0] #this fluctuations in position due to overall movement of more degrees of
f_xe2=xv(dt,iterations,k,m,gamma_e,T)[0] #freedom are the ones that determine (negligible) fluctuations of energy

def correlation_0(lst):
    acc=0
    for i in lst:
        acc+=i**2
    return acc/len(lst)  #this returns the first value of the autocorrelation function (=standard deviation for an average value of zero)

dev_xj=correlation_0(f_xj)
dev_xe1=correlation_0(f_xe1)
dev_xe2=correlation_0(f_xe2)


# At this point we are switching units, going from nm to treating the waves as cm^-1.
# Below is the COMPUTATION OF THE ENERGY AND COUPLING FLUCTUATIONS for the realistic case. The same process can be repeated for any other case
# by varying the st_dev values.

dE=100
J_o=-500
st_dev_E=500 #all in cm^-1  #Surprisingly large deviation, will give biiiig fluctuations in energy
st_dev_J=150

E_1=np.zeros(iterations)
E_2=np.zeros(iterations)
J_f=np.zeros(iterations)  #fluctuating coupling term
J_nf=np.zeros(iterations) #non-fluctuating coupling term, just a constant -500

E_1[0]=0
E_2[0]=dE
J_f[0]=J_o
J_nf[0]=J_o

for i in range(1,len(E_1)):
    E_1[i]=sqrt((st_dev_E**2)/dev_xe1)*f_xe1[i] #we invoke the second item on the autocorrelation
                                                             #because there is an error on the first, gives 0 value
for j in range(1,len(E_2)):
    E_2[j]=dE+sqrt((st_dev_E**2)/dev_xe2)*f_xe2[j]
    
for k in range(1,len(J_f)):
    J_f[k]=J_o+sqrt((st_dev_J**2)/dev_xj)*f_xj[k]
    
for i in range(1,len(J_nf)):
    J_nf[i]=J_o
    
    
# Plot of the fluctuations due to disorder in the three terms of the Hamiltonian, the energies of the two
# sites and the coupling term J that allows for quantum coherence.

fig, (e1,e2,jf) = plt.subplots(3, sharex=True,figsize=(10,6)) 
e1.plot(t_axis, E_1[:len(t_axis)],'y',label='Energy site 1') 
e1.set_ylabel('Energy site 1 (1/cm)')

e2.plot(t_axis, E_2[:len(t_axis)],'b',label='Energy site 2')
e2.set_ylabel('Energy site 2 (1/cm)')

jf.plot(t_axis, J_f[:len(t_axis)],'g',label='Coupling term J')
jf.set_ylabel('Coupling J (1/cm)')

plt.xlabel('Time (ps)')
plt.show()





# CREATING THE TIME EVOLUTION OF THE DENSITY MATRIX, for any Hamiltonian of the form H=[[E_1,J],[J,E_2]]. Iterating over 1000 realizations
# to succesfully determine the coherence between all of them and therefore show the decoherence when plotting the population coefficients.

dt=0.004 #ps
iterations=2500
c=3e-2 # cm/ps

tmax=2450
N=2500000
dN=2500

xt=np.linspace(0.0,tmax*dt,tmax)

def fluctuating_rho(E1,E2,Jf):
    rho_11f=np.zeros((tmax),dtype=complex)
    rho_12f=np.zeros((tmax),dtype=complex)
    realizations_f=0
    # Loop over realizations
    for ti in range(0,N-tmax,dN):
        psi_f=[1,0]
        a,b=psi_f
        rho_11f[0]=rho_11f[0]+a*np.conj(a)
        rho_12f[0]=rho_12f[0]+a*np.conj(b)
        #Creating one realization, adding every other on top of each of the 2450 iterations
        for tj in range(1,tmax):
            HH=np.array([[E1[ti+tj],Jf[ti+tj]],[Jf[ti+tj],E2[ti+tj]]])
            U=myexp(2*np.pi*c*dt*HH)
            psi_f=U@psi_f
            a,b=psi_f
            
            rho_11f[tj]=rho_11f[tj]+a*np.conj(a)
            rho_12f[tj]=rho_12f[tj]+a*np.conj(b)
            
        realizations_f=realizations_f+1

    rho_11f=rho_11f/realizations_f # Averaging out the added coefficients from the 1000 realizations at each iteration separated by 4fs.
    rho_12f=rho_12f/realizations_f # Result is the time evolution of the density matrix components of our dimer with certain disorder.
    
    # calculating the evolution of the two exciton eigenstates' energies for a longer realization, to afterwards classify them in an histogram.
    # The later process to get those histograms is omitted.
    exciton1=np.zeros(5*tmax)
    exciton2=np.zeros(5*tmax)
    exciton_difference=np.zeros(5*tmax)
    for te in range(5*tmax):
        HH=np.array([[E1[te],Jf[te]],[Jf[te],E2[te]]])
        e1,e2=np.linalg.eigvals(HH)
        exciton1[te]=e1
        exciton2[te]=e2
        exciton_difference[te]=abs(e1-e2)
    
    return rho_11f, rho_12f, exciton1, exciton2, exciton_difference


def nonfluctuating_rho(E1,E2): #same thing as before for a non-fluctuating coupling
    rho_11nf=np.zeros((tmax),dtype=complex)
    rho_12nf=np.zeros((tmax),dtype=complex)
    realizations_nf=0
    for ti in range(0,N-tmax,dN):
        psi_nf=[1,0]
        a,b=psi_nf
        rho_11nf[0]=rho_11nf[0]+a*np.conj(a)
        rho_12nf[0]=rho_12nf[0]+a*np.conj(b)
        for tj in range(1,tmax):
            HH=np.array([[E1[ti+tj],-500],[-500,E2[ti+tj]]])
            U=myexp(2*np.pi*c*dt*HH)
            psi_nf=U@psi_nf
            a,b=psi_nf
        
            rho_11nf[tj]=rho_11nf[tj]+a*np.conj(a)
            rho_12nf[tj]=rho_12nf[tj]+a*np.conj(b)
        realizations_nf=realizations_nf+1

    rho_11nf=rho_11nf/realizations_nf
    rho_12nf=rho_12nf/realizations_nf
    
    exciton1nf=np.zeros(5*tmax)
    exciton2nf=np.zeros(5*tmax)
    exciton_differencenf=np.zeros(5*tmax)
    for te in range(5*tmax):
        HH=np.array([[E1[te],-500],[-500,E2[te]]])
        e1,e2=np.linalg.eigvals(HH)
        exciton1nf[te]=e1
        exciton2nf[te]=e2
        exciton_differencenf[te]=abs(e1-e2)
    
    return rho_11nf, rho_12nf, exciton1nf, exciton2nf, exciton_differencenf
    
    
# At this point is recommendable to load txt files with different evolutions of the Hamiltonian matrix,
# different cases with different disorder parameters, in the case of this project

rho_11_r,rho_12_r,e1_r,e2_r,de_r=fluctuating_rho(E_1,E_2,J_f)
rho_11_nr,rho_12_nr,e1_nr,e2_nr,de_nr=nonfluctuating_rho(E_1,E_2) #example of the creation of such density matrix, for the trajectories plotted above

#recommended to store the density matrix evolution on another txt file for easier recall


# PLOTTING THE COMPUTED DENSITY MATRIX
fig, (fdr11, fdr12) = plt.subplots(2, sharex=True,figsize=(8,6)) #both plots to share the x-axis
fdr11.plot(xt,rho_11_r,'r',label='Site 1 population')
fdr11.plot(xt,1-rho_11_r,'b',label='Site 2 population')
fdr11.legend(loc='upper right')
fdr11.set_ylabel('Population coefficient')

fdr12.plot(xt,sqrt(np.real(rho_12_r)**2+np.imag(rho_12_r)**2),'purple')
fdr12.set_ylabel('Coherence coefficient')

plt.xlabel('Time [fs]')
plt.xlim(left=-10,right=150)
plt.show()



# FITTING THE DENSITY EVOLUTION

from scipy.optimize import curve_fit
def exponential_2(t,A_c,A_dc,b,c):
    return 0.5+A_c*np.cos((2*np.pi/31)*t)*np.exp(-b*t)+A_dc*np.exp(-c*t) #combination of coherent and incoherent terms for strong-disorder regime

dt=4
iterations=65

t=np.linspace(0,dt*iterations,iterations)

pars, cov = curve_fit(exponential_2, t, rho_11_r[:iterations])
A_c,A_dc,b,c=pars #fitting parameters, b is the coherence decay
stdevs = np.sqrt(np.diag(cov))
dev_Ac,dev_Adc,dev_b,dev_c=stdevs
plt.plot(t,np.real(rho_11_r[:iterations]),'r',label='Site 1 population')
plt.plot(t,exponential_2(t,*pars),'--',) #plotting the function with its fit according to the optimal parameters obtained
plt.legend(loc='best')
plt.show()

print(b,'+/-',dev_b,'    ',c,A_c,A_dc) #printing the coherence decay with its error, plus the other fitting parameters


# Similar fitting procedure is done to plot the relationship between standard deviation of the difference between excitonic energies (obtained from de_r)
# and the coherence decay parameter. Process is omitted.





# BLOCH SPHERE

import qutip as qt

def bloch_coord(rho11,rho12):
    z=np.zeros(len(rho11))
    y=np.zeros(len(rho11))
    x=np.zeros(len(rho11))
    for i in range(len(rho11)):
        z[i]=2*np.real(rho11[i])-1
        y[i]=1j*(rho12[i]-np.conj(rho12[i]))
        x[i]=rho12[i]+np.conj(rho12[i])
    return z,y,x
    
#at this point, load txt files with the density matrix evolution for your particular case

#Example of Bloch sphere:
z_r,y_r,x_r=bloch_coord(rho_11_r,rho_12_r)
z_nr,y_nr,x_nr=bloch_coord(rho_11_nr,rho_12_nr)

b=qt.Bloch()
pnts=b.add_points([x_r,y_r,z_r])
b.view=[-90,0]
b.point_color=['r']
b.zlabel=[r'$\left|1\right>$', r'$\left|2\right>$']
b.make_sphere()
b.show()

b=qt.Bloch()
pnts=b.add_points([x_r,y_r,z_r])
b.view=[-60,30]
b.size=[300,300]
b.point_color=['r']
b.zlabel=[r'$\left|1\right>$', r'$\left|2\right>$']
b.render()
#creates the Bloch Sphere for the realistic case viewed from two different angles.
    
