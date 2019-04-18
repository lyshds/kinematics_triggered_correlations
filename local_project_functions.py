import numpy as np

def wiener_deconvolution(signal, kernel, snr):
    "lambd is the SNR"
    from scipy import fft,ifft
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + snr**2)))
    return deconvolved

def make_single_kernel(times,tauon1,tauoff1):
    kx = np.copy(times)
    kon1 = lambda x:np.exp(((-1*tauon1)/(x)))
    koff1 = lambda x:np.exp((-1*x)/tauoff1)
    k1 = (kon1(kx)*koff1(kx))
    return k1/np.max(k1)

def forward_optimize(resampled_ca,
                     resample_times,
                     impulse_idxs,
                     kernel,
                     kernel_window_s = 2,
                     kernel_gain = 0.05):
    kernel = kernel*kernel_gain
    recon = np.zeros_like(resample_times)
    impulses = np.zeros_like(resample_times)
    resample_rate = resample_times[1]-resample_times[0]
    kernel_window_idx = np.int(kernel_window_s*resample_rate)
    max_idx = len(resample_times)
    end_first_loop = resample_times[-1] - kernel_window_s
    last_imp = np.argwhere(impulse_idxs < end_first_loop)[-1][0]
    for imp_idx,nxt_idx in zip(impulse_idxs[:last_imp],impulse_idxs[1:last_imp+1]):
        imp_window = impulses[imp_idx:imp_idx+kernel_window_idx]
        n1 = np.linalg.norm(recon[imp_idx:nxt_idx]-resampled_ca[imp_idx:nxt_idx])
        recon[imp_idx:imp_idx+kernel_window_idx] += kernel[:kernel_window_idx]
        n2 = np.linalg.norm(recon[imp_idx:nxt_idx] - resampled_ca[imp_idx:nxt_idx])
        if n2>n1:
            recon[imp_idx:imp_idx+kernel_window_idx] -= kernel[:kernel_window_idx]
        else:
            impulses[imp_idx] = 1
    #finish up - bit slower
    for imp_idx,nxt_idx in zip(impulse_idxs[last_imp:-2],impulse_idxs[last_imp+1:-1]):
        kw = max_idx-imp_idx
        imp_window = impulses[imp_idx:imp_idx+kw]
        n1 = np.linalg.norm(recon[imp_idx:nxt_idx]-resampled_ca[imp_idx:nxt_idx])
        recon[imp_idx:imp_idx+kw] += kernel[:kw]
        n2 = np.linalg.norm(recon[imp_idx:nxt_idx] - resampled_ca[imp_idx:nxt_idx])
        if n2>n1:
            recon[imp_idx:imp_idx+kw] -= kernel[:kw]
        else:
            impulses[imp_idx] = 1
    return recon,impulses


def make_state_matrix(flylist,
                     sorted_keys,
                     block_key = 'cl_blocks, g_x=-1, g_y=0 b_x=0, b_y=0'):
    state_mtrxs = []
    left = []
    right = []
    lmr = []
    stim_key = ('common','idx',block_key)
    for fly in flylist:
        state_mtrx = np.vstack([fly.spikestates[key] for key in sorted_keys])
        #key = ('common', 'idx', 'cl_blocks, g_x=-1, g_y=0 b_x=-8, b_y=0')
        #key = ('common', 'idx', 'cl_blocks, g_x=-1, g_y=0 b_x=8, b_y=0')
        idx_list = fly.block_data[stim_key]
        state_mtrxs.extend([np.array(state_mtrx[:,idx[100:]]) for idx in idx_list])
        left.extend([np.array(fly.left_amp)[idx[100:]] for idx in idx_list])
        right.extend([np.array(fly.right_amp)[idx[100:]] for idx in idx_list])
        lmr.extend([np.array(fly.left_amp)[idx[100:]]-np.array(fly.right_amp)[idx[100:]] 
                    for idx in idx_list])
    state_mtrx = np.hstack(state_mtrxs)
    state_mtrx = state_mtrx.astype(int)
    return state_mtrx,np.vstack(left),np.vstack(right)

def get_transiton_prob(state_mtrx):
    tprob = {}
    state_list = [tuple(row) for row in state_mtrx.T]
    state_set = set(state_list)
    state_counts = {}
    for state in state_set:
        state_counts[state] = np.sum(np.sum(state==state_mtrx.T,axis = 1)==8)
    tprob = {}
    for col1,col2 in zip(state_mtrx.T[:-1],state_mtrx.T[1:]):
        if (tuple(col1),tuple(col2)) in tprob.keys():
            tprob[tuple(col1),tuple(col2)] += 1
        else:
            tprob[tuple(col1),tuple(col2)] = 1
    return tprob,state_counts

def make_transition_matrix(tprob,
                           state_counts,
                           min_tran_num = 1,
                           min_state_num = 10):
    filtered = {}
    for key,tnum in tprob.items():
        if (tnum > min_tran_num) & \
              (state_counts[key[0]] > min_state_num) & \
              (state_counts[key[1]]> min_state_num):
            filtered[key] = tnum

    inkeys = [x[0] for x in filtered.keys()]
    outkeys = [x[1] for x in filtered.keys()]

    filterd_set = list(set(inkeys + outkeys))
    transition_mtrx = np.zeros((len(filterd_set),len(filterd_set)))

    for i,state1 in enumerate(filterd_set):
        for j,state2 in enumerate(filterd_set):
            try:
                transition_mtrx[i,j] = filtered[state1,state2]
            except KeyError:
                pass

    transition_mtrx = transition_mtrx/np.sum(transition_mtrx,axis = 1)[:,None]
    transition_mtrx[np.isnan(transition_mtrx)] = 0
    sidx = np.argsort(np.diag(transition_mtrx))[::-1]
    transition_mtrx = transition_mtrx[sidx].T[sidx]
    state_table = np.array(filterd_set)[sidx,:]
    return transition_mtrx,state_table

def next_state(current_state,state_table,tmtrx):
    """simulate a markov step using transition matrx"""
    from numpy import random
    state_idx = np.squeeze(np.argwhere(np.all(state_table == current_state,axis = 1)))
    #print state_idx
    prob_vector = tmtrx[:,state_idx]
    #print prob_vector
    idx = random.choice(np.arange(len(state_table)),p = prob_vector)
    return state_table[idx]


