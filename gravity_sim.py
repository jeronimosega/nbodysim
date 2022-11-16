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
    ax = np.zeros([n], dtype=np.double)
    ay = np.zeros([n], dtype=np.double)
    az = np.zeros([n], dtype=np.double)
    dist = np.zeros([n,n])

    for j in range(n):
        for i in range(n):
            dist[j,i] = math.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2 + (z[j] - z[i])**2 + e**2)
            ax[j] += -G * m[i] * (x[j] - x[i]) / dist[j,i]**3
            ay[j] += -G * m[i] * (y[j] - y[i]) / dist[j,i]**3
            az[j] += -G * m[i] * (z[j] - z[i]) / dist[j,i]**3

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
    T = 0

    dist = np.zeros([n,n])
    U = 0

    for j in range(0, n):
        T += m[j] + vx[j]**2 + vy[j]**2 + vz[j]**2
        for i in range(j+1,n):
            dist[j,i] = math.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2 + (z[j] - z[i])**2)
            U += -G * m[i] * m[j] / dist[j,i]
    T *= 0.5
    E = T + U

    return E, T, U

def update_half_vel(n, vx, vy, vz, ax, ay, az, dt):
    
    for i in range(n):
        vx[i] += 0.5*ax[i]*dt
        vy[i] += 0.5*ay[i]*dt
        vz[i] += 0.5*az[i]*dt
        
    return vx, vy, vz

def update_pos(x, y, z, vx, vy, vz, dt):
    
    for i in range(len(x)):
        x[i] += + vx[i]*dt
        y[i] += + vy[i]*dt
        z[i] += + vz[i]*dt
        
    return x, y, z
        


def main():
    '''
    N-body simulation using parallel computing methods
    '''
    np.random.seed(592)

    ### Set simulation parameters
    N       = 512            # Number of bodies to model
    threads = 1 
    end_t   = 10.0          # Number of timesteps
    t       = 0             # Start time of simulation
    dt      = 0.01             # Timestep
    NT      = int(end_t / dt)
    e       = 0.025           # Softening Parameter
    G       = 1           # Value of big G
    vs = 5
    ps = 15
    plot = True

    ### Generate initial conditions of the system
    mass = np.ones(N)             # Masses are restricted to be in the range 1 < M <= 3
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
    
    E[0], T[0], U[0] = calc_energy(N, x, y, z, vx, vy, vz, mass, G)
    
    if plot:
        fig = plt.figure(figsize=(6,8), dpi=100)
        grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        ax1 = plt.subplot(grid[0:2,0])
        ax2 = plt.subplot(grid[2,0])
    
    start = time.time()
    for i in range(1, NT):
        
        vx, vy, vz = update_half_vel(N, vx, vy, vz, ax, ay, az, dt)
        x, y, z = update_pos(x, y, z, vx, vy, vz, dt)
        ax, ay, az = calc_acclerations(N, x, y, z, mass, G, e, threads)
        vx, vy, vz = update_half_vel(N, vx, vy, vz, ax, ay, az, dt)
        E[i], T[i], U[i] = calc_energy(N, x, y, z, vx, vy, vz, mass, G)
        print(f'{i/NT}%')
        
        if plot:
            plt.sca(ax1)
            plt.cla()
            plt.scatter(x,y,s=10,color='blue')
            ax1.set(xlim=(-4*ps, 4*ps), ylim=(-4*ps, 4*ps))
            ax1.set_xlabel('x')
            ax1.set_aspect('equal', 'box')
            
            plt.sca(ax2)
            plt.cla()
            plt.scatter(np.arange(0,NT)*dt, E, color='tab:blue', s=1, label='Total energy')
            plt.scatter(np.arange(0,NT)*dt, T,color='tab:red',s=1,label='Kinetic energy')
            plt.scatter(np.arange(0,NT)*dt, U,color='tab:green',s=1,label='Potential Energy')
            ax2.set(xlim=(0, end_t))
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Energy')
            ax2.legend()
            plt.tight_layout()
            plt.pause(0.001)            
        
    end = time.time() - start
    print(end)
    
    plt.plot(np.arange(0,NT)*dt, E, color='tab:blue')
    plt.plot(np.arange(0,NT)*dt, T, color='tab:red')
    plt.plot(np.arange(0,NT)*dt, U, color='tab:green')
    plt.show()
    
    #plt.close('all')
    data = np.load(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/python_1.npy', 
                   allow_pickle=True)
    data = np.append(data, [N, end])
    print(data)
    np.save(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/python_1.npy', 
            data)
    
    
    
main()
    












