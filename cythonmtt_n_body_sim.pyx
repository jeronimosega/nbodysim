cimport cython
cimport openmp
import numpy as np
#np.import_array()
DTYPE = np.double
#ctypedef np.double_t DTYPE_t
import time
import yaml
from libc.math cimport sqrt
from cython.parallel cimport prange
import sys

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_acc_energy(int n, double[:] x, double[:] y, double[:] z, double[:] vx, double[:] vy, double[:] vz, double[:] m, float G, float e, int THREADS, int cs):
    '''
    Calculate the x/y/z acceleration for each body in the simulation

    Parameters:
    n: numer of particles
    x: Nx1 matrix of the positions of each of the bodies in the COM frame
    y: Nx1 matrix of the positions of each of the bodies in the COM frame
    z: Nx1 matrix of the positions of each of the bodies in the COM frame
    m: Nx1 vector of the mass of each body
    e: softening paramter

    Returns:
    ax, ay, az: Accelerations of each body in each direction in xyz plane
    '''
    cdef:
        double[:] ax = np.zeros([n], dtype=np.double)
        double[:] ay = np.zeros([n], dtype=np.double)
        double[:] az = np.zeros([n], dtype=np.double)
        double U = 0
        double T = 0
        double E = 0
        double dist = 0
        double dx = 0
        double dy = 0
        double dz = 0
        int i = 0
        int j = 0
        double mi = 0
        double mj = 0
        double xi = 0
        double yi = 0
        double zi = 0

    for i in prange(n, nogil=True, num_threads=THREADS, schedule='dynamic', chunksize=cs):
        mi = m[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]
        T += mi + vx[i]**2 + vy[i]**2 + vz[i]**2
        for j in range(n):
            mj = m[j]
            dx = xi - x[j]
            dy = yi - y[j]
            dz = zi - z[j]
            dist = sqrt(dx**2 + dy**2 + dz**2 + e**2)
            ax[i] += -G * mj * dx / dist**3
            ay[i] += -G * mj * dy / dist**3
            az[i] += -G * mj * dz / dist**3
            if j > i:
                U -= G * mj * mi / dist
    T *= 0.5
    E = T + U

    return ax, ay, az, E, T, U

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_acc_energy2(int n, double[:] x, double[:] y, double[:] z, double[:] vx, double[:] vy, double[:] vz, double[:] m, float G, float e, int THREADS, int cs):
    '''
    Calculate the x/y/z acceleration for each body in the simulation

    Parameters:
    n: numer of particles
    x: Nx1 matrix of the positions of each of the bodies in the COM frame
    y: Nx1 matrix of the positions of each of the bodies in the COM frame
    z: Nx1 matrix of the positions of each of the bodies in the COM frame
    m: Nx1 vector of the mass of each body
    e: softening paramter

    Returns:
    ax, ay, az: Accelerations of each body in each direction in xyz plane
    '''
    cdef:
        double[:] ax = np.zeros([n], dtype=np.double)
        double[:] ay = np.zeros([n], dtype=np.double)
        double[:] az = np.zeros([n], dtype=np.double)
        double[:,:] axx = np.zeros([n,n], dtype=np.double)
        double[:,:] ayy = np.zeros([n,n], dtype=np.double)
        double[:,:] azz = np.zeros([n,n], dtype=np.double)
        double U = 0
        double T = 0
        double E = 0
        double dist = 0
        double dist3 = 0
        double dx = 0
        double dy = 0
        double dz = 0
        int i = 0
        int j = 0
        double mi = 0
        double mj = 0
        double xi = 0
        double yi = 0
        double zi = 0

    for i in prange(n, nogil=True, num_threads=THREADS, schedule='dynamic', chunksize=cs):
        mi = m[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]
        T += mi + vx[i]**2 + vy[i]**2 + vz[i]**2
        for j in range(i+1, n):
            mj = m[j]
            dx = xi - x[j]
            dy = yi - y[j]
            dz = zi - z[j]
            dist = sqrt(dx**2 + dy**2 + dz**2 + e**2)
            dist3 = dist**3
            axx[i,j] = -G * mj * dx / dist3
            ayy[i,j] = -G * mj * dy / dist3
            azz[i,j] = -G * mj * dz / dist3
            axx[j,i] = -axx[i,j]
            ayy[j,i] = -ayy[i,j]
            azz[j,i] = -azz[i,j]
            U -= G * mj * mi / dist

        for j in range(n):    
            ax[i] += axx[i,j]
            ay[i] += ayy[i,j]
            az[i] += azz[i,j]

    T *= 0.5
    E = T + U

    return ax, ay, az, E, T, U


@cython.boundscheck(False)
@cython.wraparound(False)
def update_half_vel(int n, double[:] vx, double[:] vy, double[:] vz, double[:] ax, double[:] ay, double[:] az, float dt):
    '''
    Calculate the updated velocities of each particle after half a timestep

    Parameters:
    n: numer of particles
    vx: Nx1 matrix of the x velocity of each of the bodies in the COM frame
    vy: Nx1 matrix of the y velocity of each of the bodies in the COM frame
    vz: Nx1 matrix of the z velocity of each of the bodies in the COM frame
    ax: Nx1 matrix of the x acceleration of each of the bodies in the COM frame
    ay: Nx1 matrix of the y acceleration of each of the bodies in the COM frame
    az: Nx1 matrix of the z acceleration of each of the bodies in the COM frame
    dt: timestep

    Returns:
    vx, vy, vz: The updated velocities in each direction for each particle
    '''
    
    cdef: 
        int i = 0

    for i in range(n):
        vx[i] += 0.5*ax[i]*dt
        vy[i] += 0.5*ay[i]*dt
        vz[i] += 0.5*az[i]*dt
        
    return vx, vy, vz

@cython.boundscheck(False)
@cython.wraparound(False)
def update_pos(int n, double[:] x, double[:] y, double[:] z, double[:] vx, double[:] vy, double[:] vz, float dt):
    '''
    Calculate the updated velocities of each particle after half a timestep

    Parameters:
    n: numer of particles
    x: Nx1 matrix of the x position of each of the bodies in the COM frame
    y: Nx1 matrix of the y position of each of the bodies in the COM frame
    z: Nx1 matrix of the z position of each of the bodies in the COM frame
    vx: Nx1 matrix of the x velocity of each of the bodies in the COM frame
    vy: Nx1 matrix of the y velocity of each of the bodies in the COM frame
    vz: Nx1 matrix of the z velocity of each of the bodies in the COM frame
    dt: timestep

    Returns:
    vx, vy, vz: The updated velocities in each direction for each particle
    '''
    cdef: 
        int i = 0

    for i in range(n):
        x[i] += vx[i]*dt
        y[i] += vy[i]*dt
        z[i] += vz[i]*dt

    return x, y, z

@cython.boundscheck(False)
@cython.wraparound(False)
def main( PROGNAME, int N, int STEPS, int THREADS):
    '''
    N-body simulation using parallel computing methods
    '''
    np.random.seed(553)

    params = yaml.safe_load(open('/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/params.yaml'))['cython']
    ### Set simulation parameters
    cdef:
        int     cs      = params['chunksize']
        int     s       = params['schedule']
        float   t       = params['start']             # Start time of simulation
        float   dt      = params['timestep']             # Timestep
        float   e       = params['softening']           # Softening Parameter
        float   G       = params['G']           # Value of big G
        int     ps      = params['max_pos']
        int     vs      = params['max_vel']
        int i = 1

    print(f'N: {N} --- THREADS: {THREADS}')

    ### Generate initial conditions of the system
    cdef:
        double[:] mass = np.ones(N)
        double[:] x = np.random.rand(N)                # Random positions in x/y/z space with values between -ps/2 and ps/2
        double[:] y = np.random.rand(N)                # Random velocities in x,y,z space with values between -vs/2 and vs/2
        double[:] z = np.random.rand(N)
        double[:] vx = np.random.rand(N)
        double[:] vy = np.random.rand(N)
        double[:] vz = np.random.rand(N)

    ### Change all random positions and velocities to be in the range (-ps, ps) and (-vs to vs)

    for i in range(N):
        x[i], y[i], z[i] = 2*ps*(x[i]-0.5), 2*ps*(y[i]-0.5), 2*ps*(z[i]-0.5)
        vx[i], vy[i], vz[i] = 2*vs*(vx[i]-0.5), 2*vs*(vy[i]-0.5), 2*vs*(vz[i]-0.5)

    ### Create arrays to store the energy values
    cdef:
        double[:] E = np.zeros([STEPS])
        double[:] U = np.zeros([STEPS])
        double[:] T = np.zeros([STEPS])

    ### Get initial acceleration and energy
    ax, ay, az, E[0], T[0], U[0] = calc_acc_energy(N, x, y, z, vx, vy, vz, mass, G, e, THREADS, cs)
    
    start = time.time()
    ### Main loop
    for i in range(1,STEPS):
        vx, vy, vz = update_half_vel(N, vx, vy, vz, ax, ay, az, dt)
        x, y, z = update_pos(N, x, y, z, vx, vy, vz, dt)
        ax, ay, az, E[i], T[i], U[i] = calc_acc_energy2(N, x, y, z, vx, vy, vz, mass, G, e, THREADS, cs)
        vx, vy, vz = update_half_vel(N, vx, vy, vz, ax, ay, az, dt)
        
    end = time.time() - start

    # Save energy data to check simulation is workiing correctly
    np.savez(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/cython_mtt/mttcython_sim_energy.npz', tot=E, kin=T, pot=U)

    times = np.array([N, THREADS, cs, s, STEPS, end])
    try:
        timesx = np.load(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/cython_mtt/mtt_times_optmat.npy')
        times = np.vstack((timesx, times))
        np.save(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/cython_mtt/mtt_times_optmat.npy', times)

    except:
        np.save(f'/Users/jeronimo/Library/CloudStorage/OneDrive-Personal/year_4/advanced_computing/mini_project/code/cython_mtt/mtt_times_optmat.npy', times)

    print(f'Simulation duration = {end:.4f}')















