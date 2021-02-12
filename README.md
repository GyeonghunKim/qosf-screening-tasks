# qosf-screening-tasks

## Introduction
I solved task 1 as the screening task of the [qosf mentoring program](https://qosf.org/qc_mentorship/ "qosf mentorship link"). I tried to prove my computational/numerical skills, visualization skills, and quantum computing knowledge in this project. I implemented the SWAP test of the multi-qubit system in product state with [numpy](https://numpy.org/ "numpy official page link") and [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html "scipy documentation link about optimize.minimize") function, then visualized the optimization results with static diagrams on parameter space of each qubit and animation on Bloch sphere with [matplotlib](https://matplotlib.org/ "matplotlib official page link")and [qiskit.visualization](https://qiskit.org/documentation/apidoc/visualization.html "qiskit official documentation link about qiskit.visualization library"). While implementing the SWAP test, for making codes with smaller time and space complexity, I changed the computation strategy several times. I described my [final results](#Final Results) and [trials](#Trials) separately not to make any confusion. 

## 1. Final Results

![Fidelities](/images/fidelities.png)


## 2. Trials

### 2-1. SWAP test with matrix multiplication
Firstly, I tried to simulate the SWAP test by constructing a circuit matrix, then multiplying it into the state vector to calculate the test's return value. In this phase, I used the state vector in the Hilbert space H_1, where the first C^2 space denotes the reference qubit, next each n qubits represent the first and second quantum state. The apply_CSWAP function creates a CSWAP matrix with a fixed control index(0) and two variable target index of the qubit system. I used a CSWAPs memoization dictionary to reduce the waste of computation time. The total algorithm is based on the fact that the three-qubit CSWAP gate can be constructed by swapping the rows corresponding to the column basis |110> and |101>. For the n-qubit system, the CSWAP gate matrix can be created by swapping all values corresponding to the basis |1x...x1x...x0x...x> and |1x...x0x...x1x...x> where x can be arbitrary binary values. Therefore, the code iterates the possible binary strings xxxxx then change rows from the identity matrix. This works nicely with the small number of qubits; however, I noticed that this algorithm is very wasteful. 

```python
def apply_CSWAP(state, target_index1, target_index2, CSWAPs = {}):
    CSWAP = None
    if (target_index1, target_index2) in CSWAPs:
        CSWAP = CSWAPs[(target_index1, target_index2)]
    elif (target_index2, target_index1) in CSWAPs:
        CSWAP = CSWAPs[(target_index2, target_index1)]
    else:
        N = len(state)      # number of basis
        n = int(np.log2(N)) # number of qubits
        
        if n > 10:
            CSWAP = sparse.lil_matrix(np.eye(len(state)))
        else:
            CSWAP = np.eye(len(state))
            
        fixed_qubit_idxes = list(range(1, n))
        fixed_qubit_idxes.remove(target_index1)
        fixed_qubit_idxes.remove(target_index2)
        for i in range(2**len(fixed_qubit_idxes)): # number of fixed qubits while applying swap gate (except control qubit)
            binary = np.zeros(n, dtype = int)
            binary[0] = 1
            binary[fixed_qubit_idxes] = [int(x) for x in bin(i)[2:].zfill(len(fixed_qubit_idxes))]
            binary[target_index1] = 1
            binary_string1 = [str(x) for x in binary]
            swap_index1 = int('0b' + ''.join(binary_string1), 2)

            binary[target_index1] = 0
            binary[target_index2] = 1
            binary_string2 = [str(x) for x in binary]
            swap_index2 = int('0b' + ''.join(binary_string2), 2)
            CSWAP[:, [swap_index1, swap_index2]] = CSWAP[:, [swap_index2, swap_index1]]
        CSWAPs[(target_index1, target_index2)] = CSWAP
        print("create CSWAP for " + str(target_index1) + ", " + str(target_index2))
    return CSWAP.dot(state)
    
def N_qubit_swap_test(state1, state2):
    zero_state = np.array([1, 0])[:, np.newaxis]
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(len(state1))
    Z = np.array([[1, 0], [0, -1]])
    N = len(state1)
    n = int(np.log2(N)) # number of qubits
    state = np.matmul(H, zero_state)
    state = np.kron(np.kron(state, state1), state2)
    for i in range(n):
        state = apply_CSWAP(state, 1 + i, 1 + n + i, CSWAPs)
    state = np.matmul(np.kron(np.kron(H, I), I), state)
    return 1 - np.matmul(np.matmul(state.conj().T, np.kron(np.kron(Z, I), I)), state)[0, 0].real # 0 for same two state and 1 for orthogonal state
```

### 2-2. SWAP test without matrix multiplication
Secondly, I made a code more effective by removing matrix-vector multiplications. Since the circuit matrices are very sparse, direct manipulation on the state vector is a much efficient way to implement the SWAP test. Moreover, I found a more productive and straightforward method for applying a single swap on the quantum state. In the previous method, I have to iterate over basis vectors to apply the SWAP gate. However, by reshaping the state vector into the tensor in C^(2n+1), the problem is reduced to the transposing two dimensions related to the swap targets.

```python
def apply_CSWAP(state, target_index1, target_index2, CSWAPs = {}):
    N = len(state)      # number of basis
    n = int(np.log2(N)) # number of qubits

    swap_idx = np.arange(n-1)
    swap_idx[target_index1 - 1], swap_idx[target_index2 - 1] = swap_idx[target_index2 - 1], swap_idx[target_index1 - 1]
    state[N//2:] = state.reshape([2]*n)[1].transpose(swap_idx).reshape((2**(n-1),1))
    return state

def N_qubit_swap_test(state1, state2, CSWAPs = {}):
    zero_state = np.array([1, 0])[:, np.newaxis]
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    N = len(state1)
    n = int(np.log2(N)) # number of qubits
    state = np.matmul(H, zero_state)
    state = np.kron(np.kron(state, state1), state2)
    for i in range(n):
        state = apply_CSWAP(state, 1 + i, 1 + n + i, CSWAPs)
    processed_state = deepcopy(state)
    length = len(processed_state)
    processed_state[:length//2] = (state[:length//2] + state[length//2:])/np.sqrt(2)
    processed_state[length//2:] = (state[:length//2] - state[length//2:])/np.sqrt(2)
    transposed_state = deepcopy(processed_state).T.conj()
    processed_state[length//2:] *= -1
    return_value = 1 - np.matmul(transposed_state, processed_state).real
    return return_value # 0 for two identical states and 1 for orthogonal states

```

### 2-3. SWAP test with reduced state vector
Lastly, I changed the structure of the state vector to reduce the space complexity. Generally, (2n+1) qubits are required for the n qubit SWAP test. However, since the problem restricted the target state as the product state, the state does not need to have the structure of C^x(2n+1). By considering the qubits' entanglement relations, I observed that only a small subspace of it, C^2 x [+n (C^2 x C^2)], is required. The size of the state vector was then reduced significantly from 2^(2n+1) to 8n. 

```python
def apply_CSWAP_product_state(state, target_index):
    N = len(state)      # number of basis
    n = N//8 # number of qubits
    state = state.reshape((2, n, 2, 2))
    state[1, target_index] = state[1, target_index].transpose()
    state = state.reshape((N))
    return state

def N_qubit_SWAP_test_product_state(state1, state2):
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


```