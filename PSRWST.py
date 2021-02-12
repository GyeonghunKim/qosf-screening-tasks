# -*- coding: utf-8 -*-
"""Product State Reproducer With SWAP-Test (PSRWST)

This class demonstrates the product state reproducer with SWAP test and COBYLA algorithm for classical optimization. 


Example:


Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * Adopt multi-dimensional non-convex optimizer such as partical swarm optimizatoin(PSO) or reinforcement learning. 
"""

# numerical packages
from copy import copy, deepcopy
import numpy as np
import scipy as sp
from scipy.optimize import minimize

# for visualization (plot and animation)
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

# for drawing Bloch sphere
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch

class PSRWSTSolver:
    def __init__(self):
        pass
    
    ## Single qubit operation methods
    def Rx(self, theta):
        return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]])

    def Rz(self, phi):
        return np.array([[np.exp(-0.5j * phi), 0], [0, np.exp(0.5j * phi)]])

    def circuit_embeding(self, parameters, initial_state = np.array([1, 0])[:, np.newaxis]): # parameters[0] = theta, parameters[1] = phi
        state = initial_state
        state = np.matmul(self.Rz(parameters[1]), np.matmul(self.Rx(parameters[0]), state))
        return state
    
    ## Multi qubit operation methods
    def N_qubit_embeding(parameters, initial_states = None): # parameters[:, 0] = theta, parameters[:, 1] = phi
        if not initial_states:
            initial_states = np.array([1, 0])[:, np.newaxis][np.newaxis, :].repeat(len(parameters), axis = 0)
        states = []
        for parameter, initial_state in zip(parameters, initial_states):
            states.append(circuit_embeding(parameter, initial_state))
        state = np.array(states).reshape((len(states), 2))/np.sqrt(len(parameters))
        return state
    
    def apply_CSWAP(state, target_index):
        N = len(state)      # number of basis
        n = N//8 # number of qubits
        state = state.reshape((2, n, 2, 2))
        state[1, target_index] = state[1, target_index].transpose()
        state = state.reshape((N))
        return state
    
    ## SWAP test
    def SWAP_test(state1, state2):
        zero_state = np.array([1, 0])[:, np.newaxis]
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        N = len(state1)
        n = N # int(np.log2(N)) # number of qubits

        zero_state = np.matmul(H, zero_state).flatten()
        state = []

        for qubit1, qubit2 in zip(state1, state2):
            state.append(np.kron(qubit1, qubit2))

        state = np.array(state).flatten()
        state = np.kron(zero_state, state)
        for i in range(n):
            state = apply_CSWAP_product_state(state, i)
        processed_state = deepcopy(state)
        length = len(processed_state)
        processed_state[:length//2] = (state[:length//2] + state[length//2:])/np.sqrt(2)
        processed_state[length//2:] = (state[:length//2] - state[length//2:])/np.sqrt(2)
        transposed_state = processed_state.T.conj()
        processed_state[length//2:] *= -1

        return_value = 1 - np.matmul(transposed_state, processed_state).real
        return return_value # 0 for two identical states and 1 for orthogonal states

    ## Loss function for the optimize.minimize
    def N_qubit_loss_val(parameter, random_state, optimization_traj = []):
        if parameter.ndim == 1:
            parameter = parameter.reshape((len(parameter)//2, 2))
        optimization_traj.append(copy(parameter))
        current_state = N_qubit_embeding(parameter)
        return N_qubit_swap_test(current_state, random_state)

    ## Visualization methods
        
    def qstate_to_bloch_vector(self, state):
        dm = np.matmul(state, state.conj().T)
        bloch_vector = np.array([2 * dm[0, 1].real, 2 * dm[1, 0].imag, dm[0, 0] - dm[1, 1]]).real
        return bloch_vector
    
    def create_parameter_trajectory(self, t_list, *args):
        return np.array([f(t_list) for f in args]).T

    def sweep_on_bloch_sphere(self):
        def f_theta(t_list):
            return 2 * np.pi * t_list

        def f_phi(t_list):
            return 40 * np.pi * t_list

        def update_bloch_sphere(idx, traj):
            ax.clear()
            B = Bloch(axes=ax)
            B.add_vectors(traj[idx])
            if idx > 0:
                B.add_points(traj[:idx].T)
            B.render(title=None)

        traj = self.create_parameter_trajectory(np.linspace(0, 1, 400), *[f_theta, f_phi])
        traj_on_Bloch_sphere = np.array([self.qstate_to_bloch_vector(self.circuit_embeding(point)) for point in traj])
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-0.7, 0.7)
        ax.set_ylim3d(-0.7, 0.7)
        ax.set_zlim3d(-0.7, 0.7)
        fig.set_size_inches(5, 5)
        
        ani = matplotlib.animation.FuncAnimation(fig, update_bloch_sphere, frames = np.arange(400), interval=30, blit=False, repeat = True, fargs = (traj_on_Bloch_sphere,))
            
        return ani

    def show_N_qubit_optimization_trajectory_on_bloch_sphere(self, optimization_traj, figsize = (5, 5), view = (-60, 30)):
        axs = []
        n_qubits = len(optimization_traj[0])

        figure_array_shape = (1, n_qubits)
        if n_qubits > 4:
            figure_array_shape = (n_qubits//4 + 1, 4)

        figsize = (figsize[0] * figure_array_shape[1], figsize[1] * figure_array_shape[0])

        def update(idx, trajectories):
            for qubit_idx, trajectory in enumerate(trajectories):
                axs[qubit_idx].clear()
                B = Bloch(axes=axs[qubit_idx])
                B.add_vectors(trajectory[idx])
                B.add_vectors(qstate_to_bloch_vector(decomposed_random_state[qubit_idx]))
                if idx > 10:
                    B.add_points(np.array(trajectory[idx - 10:idx]).T)
                elif idx > 0:
                    B.add_points(np.array(trajectory[:idx]).T)
                B.render(title=None)

        fig = plt.figure(figsize=figsize)



        for i in range(n_qubits):
            axs.append(fig.add_subplot(figure_array_shape[0], figure_array_shape[1], i+1, projection='3d'))
            axs[-1].set_axis_off()
            axs[-1].set_xlim3d(-0.7, 0.7)
            axs[-1].set_ylim3d(-0.7, 0.7)
            axs[-1].set_zlim3d(-0.7, 0.7)

        fig.set_size_inches(figsize[0], figsize[1])
        trajectories_on_Bloch_sphere = [np.array([qstate_to_bloch_vector(circuit_embeding(point)) for point in one_optimization_traj]) for one_optimization_traj in np.array(optimization_traj).transpose(1, 0, 2)]
        ani = matplotlib.animation.FuncAnimation(fig, 
                                                 update, 
                                                 frames = np.arange(len(trajectories_on_Bloch_sphere[0])), 
                                                 interval=50, blit=False, repeat = False, 
                                                 fargs = (trajectories_on_Bloch_sphere,))
        return ani

    