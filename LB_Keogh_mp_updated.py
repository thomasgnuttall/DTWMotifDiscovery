import numpy as np
from matlab import movmax, movmean

def LB_Keogh_mp_updated(ts, subseqlen, minlag, warpmax, DoNotCompute, normalize=False):
    """
    Computes all pairs LB_Keogh in a 1 to many fashion
    Query is used to generate the envelope in each case
    Inputs are time series: ts,
    subsequence len (integer): subseqlen
    minimum index offset between subsequence pairs (integer): minlag
    maximum warping offset (integer):  warpmax
    DoNotCompute: vector of true and false booleans showing which subsequences to
    prune
    Outputs are matrix profile and index using LB_Keogh
    """

    if any([not isinstance(ts, np.ndarray), not isinstance(DoNotCompute, np.ndarray)]):
        Exception('Expected a vector input')

    istransposed = not ts.shape[0]

    if istransposed:
        ts = ts[..., None]
    
    if istransposed:
        DoNotCompute = DoNotCompute[..., None]
    
    if subseqlen < 4 or minlag < 4:
        Exception('Bad parameters')

    subcount = len(ts) - subseqlen + 1 # number of subsequences that fit
    mu = movmean(ts, [0, subseqlen-1])
    sig = movmean(ts, [0, subseqlen-1], std=True)

    subs = np.zeros((subseqlen, subcount))

    # Normalize everything in advance. If this is memory prohibitive,
    # subdivide your problem and do the same thing. Just don't normalize every
    # single time on the fly. That is generally slower and more complicated.
    if normalize:
        for i in range(0, subcount):
            subs[:, i] = (ts[i:i+subseqlen] - mu[i])/sig[i]

    # These are stored in
    mp = np.zeros((subcount, ))
    mpi = np.zeros((subcount, ))
    
    U = np.zeros((subseqlen, subcount))
    L = np.zeros((subseqlen, subcount))
    
    mp[:] = np.Inf
    mpi[:] = np.Inf

    U[:] = np.nan
    L[:] = np.nan

    if warpmax > 0:
        for i in range(0, subcount):
            if DoNotCompute[i]:
                continue
            U[:, i] = movmax(subs[:, i], [warpmax, warpmax])
            L[:, i] = movmax(subs[:, i], [warpmax, warpmax], return_min=True)

    else:
        for i in range(1, subcount):
            if DoNotCompute[i]:
                continue

            U[:, i] = subs[:, i]
            L[:, i] = subs[:, i]


    # In C++ code, you would set this up in a slightly different manner. It
    # always takes some memory to set all of these up, but you can store 1
    # sided U and L in O(n) memory, which can be used to quickly form
    # per subsequence U,L with simple merges. Next, you can tile this.
    # You may repeat the normalization step a few times, but each time you
    # normalize and form U,L for some subset of the problem, you use them to form
    # O(tilelen^2) pairs. This way the cost of data movement is trivialized at
    # typical subsequence lens relative to the cost of the main loop of each tile. 

    for i in range(0, subcount):
        for j in range(i+minlag, subcount):
            if DoNotCompute[i] or DoNotCompute[j]:
                continue

            A = np.square(np.subtract(U[:, i], subs[:, j]))
            B = U[:,i] < subs[:, j]
            C = np.square(np.subtract(L[:, i], subs[:, j]))
            D = L[:, i] > subs[:, j]
            E = np.square(np.subtract(U[:, j], subs[:, i]))
            F = U[:,j] < subs[:, i]
            G = np.square(np.subtract(L[:, j], subs[:, i]))
            H = L[:, j] > subs[:, i]

            d = max(np.sum(np.add(np.multiply(A, B), np.multiply(C, D))),\
                    np.sum(np.add(np.multiply(E, F), np.multiply(G, H))))
            if d < mp[i]:
                mp[i] = d
                mpi[i] = j

            if d < mp[j]:
                mp[j] = d
                mpi[j] = i

    mp = [max([x,0])**0.5 for x in mp]

    if istransposed:
        mp = mp.T[0]
        mpi = mpi.T[0]
        
    return mp, mpi
