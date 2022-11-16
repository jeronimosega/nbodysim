import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import math
import time
import sys

def calc_acclerations(n, x, y, z, m, G, e, threads=1):
    '''
    Calculate the x/y/z acceleration for each body in the simulation

    Parameters:
    n: numer of particles
    r: Nx3 matrix of the positions of each of the bodies in the COM frame
    m: Nx1 vector of the mass of each body
    e: softening paramter

    Returns:
    ax, ay, az: Accelerations of each body in each direction in xyz plane
    '''
    dx = np.array([x]).T - x
    dy = np.array([y]).T - y
    dz = np.array([z]).T - z
    
    inv_r3 = np.sqrt(dx**2 + dy**2 + dz**2 + e**2)
    inv_r3[inv_r3>0] = 1/inv_r3[inv_r3>0]**3

    ax = -G * (dx * inv_r3) @ m
    ay = -G * (dy * inv_r3) @ m
    az = -G * (dz * inv_r3) @ m

    return ax, ay, az


def calc_energy(n, x, y, z, vx, vy, vz, m, G, threads=1):
    '''
    Calculate the total (E), kinetic (T) and potential (U) energy in the simulation

    Parameters:
    n: numer of particles
    r: Nx3 matrix of the positions of each of the bodies in the COM frame
    v: Nx3 matrix of the velocities of each of the bodies in the COM frame
    m: Nx1 vector of the mass of each body

    Returns:
    E, T, U: Total, Kinetic and Potential energies of the system 
    '''

    dx = np.array([x]).T - x
    dy = np.array([y]).T - y
    dz = np.array([z]).T - z
    T = 0.5*np.sum((vx**2 + vy**2 + vz**2) * m)

    
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    U = - G * np.sum(np.sum(np.triu(m * np.array([m]).T * inv_r),1))

    E = T + U

    return E, T, U        


def main():
    '''
    N-body simulation using parallel computing methods
    '''
    np.random.seed(233)

    ### Set simulation parameters
    N       = 256            # Number of bodies to model
    threads = 1 
    end_t   = 10          # Number of timesteps
    t       = 0             # Start time of simulation
    dt      = 0.01             # Timestep
    NT      = int(end_t / dt)
    e       = 0.01           # Softening Parameter
    G       = 5           # Value of big G
    vs = 2
    ps = 25
    plot = True

    ### Generate initial conditions of the system
    mass = 0.1*np.ones(N)             # Masses are restricted to be in the range 1 < M <= 3
    x = ps * 2*(np.random.rand(N) - 0.5)                # Random positions in x/y/z space with values between -ps/2 and ps/2
    y = ps * 2*(np.random.rand(N) - 0.5)                # Random velocities in x,y,z space with values between -vs/2 and vs/2
    z = ps * 2*(np.random.rand(N) - 0.5)
    
    
    vx = vs * 2*(np.random.rand(N) - 0.5)
    vy = vs * 2*(np.random.rand(N) - 0.5)
    vz = vs * 2*(np.random.rand(N) - 0.5)
    vx -= np.mean(mass * vx) / np.mean(mass)
    vy -= np.mean(mass * vy) / np.mean(mass)
    vz -= np.mean(mass * vz) / np.mean(mass)
    
    ax, ay, az = calc_acclerations(N, x, y, z, mass, G, e)
    
    E = np.zeros([NT])
    U = np.zeros([NT])
    T = np.zeros([NT])
    
    if plot: fig, (ax1, ax2) = plt.subplots(2,1)
    start = time.time()
    for i in range(NT):
        print(f'{i/NT:.3f}%')
        
        vx += ax*dt / 2
        vy += ay*dt / 2
        vz += az*dt / 2
        
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        ax, ay, az = calc_acclerations(N, x, y, z, mass, G, e)
        
        vx += ax*dt / 2
        vy += ay*dt / 2
        vz += az*dt / 2
        
        E[i], T[i], U[i] = calc_energy(N, x, y, z, vx, vy, vz, mass, G)
        
        if plot or (i == NT):
            plt.sca(ax1)
            plt.cla()
            plt.scatter(x,y,s=10,color='blue')
            ax1.set(xlim=(-4*ps, 4*ps), ylim=(-4*ps, 4*ps))
            ax1.set_aspect('equal', 'box')
            
            plt.sca(ax2)
            plt.cla()
            plt.scatter(np.arange(0,NT)*dt, E, color='tab:blue', s=1, label='Total energy')
            plt.plot(np.arange(0,NT)*dt, T,color='tab:red',label='Kinetic energy', ls='--')
            plt.plot(np.arange(0,NT)*dt, U,color='tab:green',label='Potential Energy', ls='--')
            ax2.set(xlim=(0, end_t))
            ax2.legend()
            
            plt.tight_layout()
            plt.pause(0.0001)

            
    end = time.time() - start
    print(end)
    #plt.close('all')
    plt.plot(np.arange(0,NT)*dt, E, color='tab:blue')
    plt.plot(np.arange(0,NT)*dt, T, color='tab:red')
    plt.plot(np.arange(0,NT)*dt, U, color='tab:green')
    plt.show()
    
    #plt.close('all')
    data = np.load(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/numpy_1.npy', 
                   allow_pickle=True)
    data = np.append(data, [N, end])
    print(data)
    np.save(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/numpy_1.npy', 
            data)  

    
main()
    












