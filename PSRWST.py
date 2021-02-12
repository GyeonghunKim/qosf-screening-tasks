# -*- coding: utf-8 -*-
"""Product State Reproduce With SWAP-Test (PSRWST) Solver

This class demonstrates the product state reproducer with SWAP test and COBYLA algorithm for classical optimization. 

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
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

# for drawing Bloch sphere
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch

class PSRWSTSolver:
    def __init__(self):
        self.decomposed_random_state = []
        self.optimization_traj = []
    
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
    def N_qubit_embeding(self,parameters, initial_states = None): # parameters[:, 0] = theta, parameters[:, 1] = phi
        if not initial_states:
            initial_states = np.array([1, 0])[:, np.newaxis][np.newaxis, :].repeat(len(parameters), axis = 0)
        states = []
        for parameter, initial_state in zip(parameters, initial_states):
            states.append(self.circuit_embeding(parameter, initial_state))
        state = np.array(states).reshape((len(states), 2))/np.sqrt(len(parameters))
        return state
    
    def apply_CSWAP(self,state, target_index):
        N = len(state)      # number of basis
        n = N//8 # number of qubits
        state = state.reshape((2, n, 2, 2))
        state[1, target_index] = state[1, target_index].transpose()
        state = state.reshape((N))
        return state
    
    ## SWAP test
    def SWAP_test(self,state1, state2):
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
            state = self.apply_CSWAP(state, i)
        processed_state = deepcopy(state)
        length = len(processed_state)
        processed_state[:length//2] = (state[:length//2] + state[length//2:])/np.sqrt(2)
        processed_state[length//2:] = (state[:length//2] - state[length//2:])/np.sqrt(2)
        transposed_state = processed_state.T.conj()
        processed_state[length//2:] *= -1

        return_value = 1 - np.matmul(transposed_state, processed_state).real
        return return_value # 0 for two identical states and 1 for orthogonal states
    def fideltity_two_qubits(self, state1, state2):
        return abs(np.matmul(state1.T.conj(), state2))**2

    ## Loss function for the optimize.minimize
    def N_qubit_loss_val(self,parameter, random_state, optimization_traj = []):
        if parameter.ndim == 1:
            parameter = parameter.reshape((len(parameter)//2, 2))
        self.optimization_traj.append(copy(parameter))
        current_state = self.N_qubit_embeding(parameter)
        return self.SWAP_test(current_state, random_state)

    def reproduce_quantum_state(self, random_parameters, verbose = False):
        number_of_qubits = len(random_parameters)
        # randomly generated quantum state:  random_state
        random_parameters = 2 * np.pi * np.random.random((number_of_qubits, 2))
        random_state = self.N_qubit_embeding(random_parameters)
        self.decomposed_random_state = [self.circuit_embeding(random_parameter) for random_parameter in random_parameters]
        self.optimization_traj = []
        res = minimize(self.N_qubit_loss_val, np.zeros((number_of_qubits, 2)), method='COBYLA', args = (random_state, self.optimization_traj)) # , tol=1e-4, options={'maxiter': 3000})
        if verbose:
            print(res.success)
            print("answer: \n" + str(random_parameters[:, 0]%(np.pi)))
            print("\n" + str(random_parameters[:, 1]%(2 * np.pi)))
            print("optimized solution: \n" + str(res.x[:, 0]%(np.pi)))
            print("\n" + str(res.x[:, 1]%(2 * np.pi)))
        return res.x, self.optimization_traj
    
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

    def show_N_qubit_optimization_trajectory_on_bloch_sphere(self, figsize = (5, 5), view = (-60, 30), number_of_animation_frame = 30):
        if (not self.optimization_traj) or (not self.decomposed_random_state):
            print("You should run the function reproduce_quantum_state first")
            return
        
        optimization_traj = np.array(self.optimization_traj)[::len(self.optimization_traj) // number_of_animation_frame if len(self.optimization_traj) > number_of_animation_frame else 1]
        
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
                B.add_vectors(self.qstate_to_bloch_vector(self.decomposed_random_state[qubit_idx]))
                if idx > 10:
                    B.add_points(np.array(trajectory[idx - 10:idx]).T)
                elif idx > 0:
                    B.add_points(np.array(trajectory[:idx]).T)
                if qubit_idx == 1:
                    B.render(title=str(qubit_idx) + "st qubit")
                elif qubit_idx == 2:
                    B.render(title=str(qubit_idx) + "nd qubit")
                elif qubit_idx == 3:
                    B.render(title=str(qubit_idx) + "rd qubit")
                else:
                    B.render(title=str(qubit_idx) + "th qubit")
            if idx == 0:
                fig.suptitle('Initial quantum states. Blue: Random Product State. Red: Initial Guess', fontsize=16)
            else:
                fig.suptitle(str(idx * (len(self.optimization_traj) // number_of_animation_frame)) + ' iterations after', fontsize=16)
                

        fig = plt.figure(figsize=figsize)

        for i in range(n_qubits):
            axs.append(fig.add_subplot(figure_array_shape[0], figure_array_shape[1], i+1, projection='3d'))
            axs[-1].set_axis_off()
            axs[-1].set_xlim3d(-0.7, 0.7)
            axs[-1].set_ylim3d(-0.7, 0.7)
            axs[-1].set_zlim3d(-0.7, 0.7)

        fig.set_size_inches(figsize[0], figsize[1])
        trajectories_on_Bloch_sphere = [np.array([self.qstate_to_bloch_vector(self.circuit_embeding(point)) for point in one_optimization_traj]) for one_optimization_traj in np.array(optimization_traj).transpose(1, 0, 2)]
        ani = matplotlib.animation.FuncAnimation(fig, 
                                                 update, 
                                                 frames = np.arange(len(trajectories_on_Bloch_sphere[0])), 
                                                 interval=50, blit=False, repeat = False, 
                                                 fargs = (trajectories_on_Bloch_sphere,))
        return ani

    def show_static_parameter_space_trajectory(self, number_of_animation_frame = 30, figsize = (20, 20), fontsize = 10):
        if (not self.optimization_traj) or (not self.decomposed_random_state):
            print("You should run the function reproduce_quantum_state first")
            return
        
        optimization_traj = np.array(self.optimization_traj)[::len(self.optimization_traj) // number_of_animation_frame if len(self.optimization_traj) > number_of_animation_frame else 1]
        optimization_traj[:, :, 0]%=(np.pi)
        optimization_traj[:, :, 1]%=(2 * np.pi)
        axs = []
        n_qubits = len(optimization_traj[0])

        figure_array_shape = (1, n_qubits)
        if n_qubits > 4:
            figure_array_shape = (n_qubits//4 + 1, 4)

        figsize = (figsize[0] * figure_array_shape[1], figsize[1] * figure_array_shape[0])
        fig = plt.figure(figsize=figsize)
        for i in range(n_qubits):
            axs.append(fig.add_subplot(figure_array_shape[0], figure_array_shape[1], i+1))
            # axs[-1].plot(optimization_traj[:, i, 0], optimization_traj[:, i, 1])
            for (x1, y1), (x2, y2) in zip(optimization_traj[:-1, i, :], optimization_traj[1:, i, :]):
                axs[-1].arrow(x1, y1, x2-x1, y2-y1, width = 0.03)
            axs[-1].set_xlabel(r'$\mathrm{\theta}$', fontsize=fontsize)
            axs[-1].set_ylabel(r'$\mathrm{\phi}$', fontsize=fontsize)
            axs[-1].tick_params(axis='both', which='major', labelsize=fontsize)
            axs[-1].set_xlim(0, np.pi)
            axs[-1].set_ylim(0, 2 * np.pi)
            axs[-1].grid(True)
        