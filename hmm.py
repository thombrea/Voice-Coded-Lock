from scipy.stats import multivariate_normal
import numpy as  np

def initialize_hmm(X_list, nstates):
    '''Initialize hidden Markov models by uniformly segmenting input waveforms.

    Inputs:
    X_list (list of mfccs, (nframes[n],nmfccs) arrays): 
        X_list[n][t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    nstates (scalar): 
        the number of states to initialize

    Returns:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i), estimates as
        (# times q[t]=j and q[t-1]=i)/(# times q[t-1]=i).
    Mu (nstates,nmfccs):
        Mu[i,:] = mean vector of the i'th state, estimated as
        average of the frames for which q[t]=i.
    Sigma (nstates,nmfccs,nmfccs):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        unbiased sample covariance of the frames for which q[t]=i.
    '''
    
    #get states, divide  feature matrix uniformly for each state
    N = len(X_list)
    nmfccs = np.shape(X_list[0])[1]
    
    nframes = np.array([X_list[n].shape[0] for n in range(N)]) # get list of number of frames
    
    state = []
    for i in range(nstates):
        arr = np.array(X_list[0][int(i * nframes[0]/nstates) : int((i+1)*nframes[0]/nstates),:]) # first part
        for n in range(1, N):
            arr = np.concatenate((arr, X_list[n][int(i * nframes[n]/nstates) : int((i+1)*nframes[n]/nstates),:]), axis=0) # vertical stack
        state.append(arr) # append to our states list
    
    
    #calculate A
    A = np.zeros((nstates,nstates)) # transition matrix
    for i in range(nstates-1):
        length = state[i].shape[0] / N # get length of each state
        A[i,i] = (length-1)/length # all but 1 are stay the same
        A[i,i+1] = 1/length # 1 transition to next state
    A[nstates-1,nstates-1] = 1.0 # last just goes to itself

    #get Mu
    Mu = np.zeros((nstates,nmfccs)) # avg vec of ith state
    
    for i in range(nstates):
        Mu[i,:] = np.average(state[i],axis=0) # average down the cols
    
    #sigma
    Sigma = np.array([np.cov(np.transpose(state[i]), bias=False) for i in range(nstates)])

    return A, Mu, Sigma
 

def observation_pdf(X, Mu, Sigma):
    '''Calculate the log observation PDFs for every frame, for every state.

    Inputs:
    X (nframes,nmfccs):
        X[t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    Mu (nstates,nmfccs):
        Mu[i,:] = mean vector of the i'th state
    Sigma (nstates,nmfccs,nmfccs):
        Sigma[i,:,:] = covariance matrix, i'th state

    Returns:
    B (nframes,nstates):
        B[t,i] = max(p(X[t,:] | Mu[i,:], Sigma[i,:,:]), 1e-100)
    '''
    nframes = np.shape(X)[0]
    nstates = np.shape(Mu)[0]
    B = np.zeros((nframes,nstates))
    
    for t in range(nframes):
        for i in range(nstates):
            B[t,i] = max(multivariate_normal.pdf(X[t,:], Mu[i,:], Sigma[i,:,:], allow_singular=True), 1e-100)
            
            
    return B
    
    
def scaled_forward(A, B):
    '''Perform the scaled forward algorithm.

    Inputs:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q[t]=i | X[:t,:], A, Mu, Sigma)
    G (nframes):
        G[t] = p(X[t,:] | X[:t,:], A, Mu, Sigma)
    '''
    nframes,nstates = np.shape(B)
    Alpha_Hat = np.zeros((nframes,nstates))
    G = np.zeros(nframes)
    
    pi = np.zeros((nstates))
    pi[0] = 1.0 # initial state = 0
    
    # Initialize: alpha_1 [i] = pi_i * B_i[x_0]
    Alpha_Hat[0,:] = pi * B[0,:] 
    
    G[0] = np.sum(Alpha_Hat[0,:]) # g_t = sum_i alpha_t [i]
    Alpha_Hat[0,:] /= G[0]
     
    # Iterate: alpha_t [j] = sum_i alpha_(t-1) [j] * A_ij * B_j [x_t]
    for t in range(1,nframes):
        for j in range(nstates):
            Alpha_Hat[t,j] = Alpha_Hat[t-1,:] @ A[:,j] * B[t,j]
            
        G[t] = np.sum(Alpha_Hat[t,:]) # g_t = sum_i alpha_t [i]
        Alpha_Hat[t,:] /= G[t]        
    
    return Alpha_Hat, G
    
def scaled_backward(A, B):
    '''Perform the scaled backward algorithm.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / max_j p(X[t+1:,:]| q[t]=j, A, Mu, Sigma)
    '''
    nframes,nstates = np.shape(B)
    Beta_Hat = np.zeros((nframes,nstates))
    
    Beta_Tilde = np.zeros((nframes,nstates))
    c = np.zeros((nframes))
    
    #Initialize
    for i in range(nstates):
        Beta_Hat[nframes-1,i] = 1.0
    
    #Iterate
    for t in range(nframes-2,-1,-1):
        for i in range(nstates):
            for j in range(nstates):
                Beta_Tilde[t,i] += A[i,j] * B[t+1,j] * Beta_Hat[t+1,j]
                
        c[t] = np.max(Beta_Tilde[t,:])
        for i in range(nstates):
            Beta_Hat[t,i] = Beta_Tilde[t,i] / c[t]
    
    return Beta_Hat

def posteriors(A, B, Alpha_Hat, Beta_Hat):
    '''Calculate the state and segment posteriors for an HMM.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q=i | X[:t,:], A, Mu, Sigma)
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / prod(G[t+1:])

    Returns:
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
                   = Alpha_Hat[t,i]*Beta_Hat[t,i] / sum_i numerator
    Xi (nframes-1,nstates,nstates):
        Xi[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
                  = Alpha_Hat[t,i]*A{i,j]*B[t+1,j]*Beta_Hat[t+1,j] / sum_{i,j} numerator
    '''
    nframes,nstates = np.shape(Alpha_Hat)
    Gamma = np.zeros((nframes,nstates))
    Xi = np.zeros((nframes-1,nstates,nstates))
    
    Gamma = Alpha_Hat * Beta_Hat
    for t in range(nframes):
        g_num = np.sum(Gamma[t,:])
        if(g_num > 0):
            Gamma[t,:] /= g_num
    
    for t in range(nframes-1):
        x_num = 0
        for i in range(nstates):
            for j in range(nstates):
                Xi[t,i,j] = Alpha_Hat[t,i] * A[i,j] * B[t+1, j] * Beta_Hat[t+1,j]
                x_num += Xi[t,i,j]

        if(x_num > 0):
            Xi[t,:,:] /= x_num
    
    return Gamma, Xi

def E_step(X, Gamma, Xi):
    '''Calculate the expectations for an HMM.

    Inputs:
    X (nframes,nmfccs):
        X[t,:] = feature vector, t'th frame of n'th waveform
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
    Xi (nsegments,nstates,nstates):
        Xi_list[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
        WARNING: rows of Xi may not be time-synchronized with the rows of Gamma.  

    Returns:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nmfccs): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nmfccs,nmfccs): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    '''
    nframes,nmfccs = np.shape(X)
    nstates = np.shape(Gamma)[1]
    nsegments = np.shape(Xi)[0]
    
    #A_num
    A_num = np.zeros((nstates,nstates))
    for i in range(nstates):
        for j in range(nstates): 
            A_num[i,j] = np.sum(Xi[:,i,j])
    
    #A_den
    A_den = np.zeros((nstates))
    for i in range(nstates):
        A_den[i] = np.sum(Xi[:,i,:])
                
    #Mu_num
    Mu_num = np.zeros((nstates,nmfccs))
    for i in range(nstates):
        for t in range(nframes):
            Mu_num[i,:] += Gamma[t,i] * X[t,:]

            
    #Mu_den
    Mu_den = np.zeros((nstates))
    for i in range(nstates):
        Mu_den[i] = np.sum(Gamma[:,i])
        
    #Sigma_num
    Sigma_num = np.zeros((nstates,nmfccs,nmfccs))
    
    Mu = np.zeros((nstates,nmfccs))
    for i in range(nstates):
        Mu[i,:] = Mu_num[i,:]/Mu_den[i]
    
    for i in range(nstates):
        for t in range(nframes):
            Sigma_num[i,:,:] += Gamma[t,i] * (np.outer((X[t,:] - Mu[i,:]), (X[t,:] - Mu[i,:])))
    
    #Sigma_den
    Sigma_den = np.zeros((nstates))
    for i in range(nstates):
        Sigma_den[i] = np.sum(Gamma[:,i])

    return A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den

def M_step(A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den, regularizer):
    '''Perform the M-step for an HMM.

    Inputs:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nmfccs): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nmfccs,nmfccs): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    regularizer (scalar):
        Coefficient used for Tikohonov regularization of each covariance matrix.

    Returns:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i), estimated as
        E[# times q[t]=j and q[t-1]=i]/E[# times q[t-1]=i)].
    Mu (nstates,nmfccs):
        Mu[i,:] = mean vector of the i'th state, estimated as
        E[average of the frames for which q[t]=i].
    Sigma (nstates,nmfccs,nmfccs):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        E[biased sample covariance of the frames for which q[t]=i] + regularizer*I
    '''
    nstates,nmfccs = np.shape(Mu_num)
    

    Sigma = np.zeros((nstates,nmfccs,nmfccs))
    
    A = np.array([A_num[i,:] / A_den[i] for i in range(nstates)])
    
    Mu = np.array([Mu_num[i,:] / Mu_den[i] for i in range(nstates)]) 

    Sigma = np.array([Sigma_num[i,:,:] / Sigma_den[i] + regularizer * np.identity(nmfccs) for i in range(nstates)]) 
        
    return A, Mu, Sigma

def recognize(X, Models):
    '''Perform isolated-word speech recognition using trained Gaussian HMMs.

    Inputs:
    X (list of (nframes[n],nmfccs) arrays):
        X[n][t,:] = feature vector, t'th frame of n'th waveform
    Models (dict of tuples):
        Models[y] = (A, Mu, Sigma) for class y
        A (nstates,nstates):
             A[i,j] = p(state[t]=j | state[t-1]=i, Y=y).
        Mu (nstates,nmfccs):
             Mu[i,:] = mean vector of the i'th state for class y
        Sigma (nstates,nmfccs,nmfccs):
             Sigma[i,:,:] = covariance matrix, i'th state for class y

    Returns:
    logprob (dict of numpy arrays):
       logprob[y][n] = log p(X[n] | Models[y] )
    Y_hat (list of strings):
       Y_hat[n] = argmax_y p(X[n] | Models[y] )
    '''
    
    Y = len(Models)
    N = len(X)
    
    logprob = { y:[] for y in Models.keys() } 
    Y_hat = [0 for n in range(N)]
    
    for y in Models.keys():
        for n in range(N):
            
            B = observation_pdf(X[n], Models[y][1], Models[y][2])
            Alpha_Hat, G = scaled_forward(Models[y][0], B)
            G = np.log(G)

            logprob[y].append(np.sum(G))
    
    return logprob