%load_ext autoreload
%autoreload 2

import os
import sys
import time
sys.path.append('../')
sys.path.append('../melodic_analysis')

import numpy as np
import numpy.matlib as np_matlab

from src.evaluation import filter_annotations, evaluate_sequences, classification_metrics
from src.recording.obj import Recording, compute_distance_matrix, cluster_dist_matrix
from src.visualisation import plot_all_sequences, double_timeseries
from src.utils import detect_local_minima, get_timestamp, get_minima, search_and_cluster, interpolate_below_length, get_keyword_path, random_city
from src.io import write_all_sequence_audio, write_json, load_annotations

from simmusic.dtw import dtw

from PAA import PAA

performance_path = '/Users/thomasnuttall/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Live at Kamarajar Hall by Sanjay Subrahmanyan/Kamakshi/'

# Interpolate 0 pitch gaps shorter than or equal to <gap_interp> seconds
gap_interp = 0 # 60ms
rec = Recording(performance_path, gap_interp=gap_interp)

TS = rec.pitch

m_secs = 4
MP = rec.self_matrix_profile(m=m_secs, smooth=None, cache=False)
mp = MP[:,0].astype(float)

##################
### dtw_motif ####
##################
# args
a = TS
mp_ed = mp
maxwarp = subseqlen
subseqlen = int(m_secs/0.0029)

import tqdm
silence_mask = []
for i in tqdm.tqdm(range(len(TS))):
    val = sum(TS[i:i+subseqlen] == 0)/subseqlen > 0.25
    silence_mask.append(1 if val else 0)

silence_mask = np.array(silence_mask)

mp_ed = np.ma.MaskedArray(mp, silence_mask[:-subseqlen+1]) #np.copy(mp_ed)

minspacing = subseqlen

################### first best-so-far=ED motif distance ###########################
first_min = np.argmin(mp_ed)
best_so_far = mp_ed[first_min]

mp_ed[max(first_min-subseqlen, 0):first_min+subseqlen] = np.Inf

sec_min = np.argmin(mp_ed)

################### second best-so-far=DTW distance between ED motifs ###########################
aa = a[first_min:first_min + subseqlen-1]
bb = a[sec_min:sec_min + subseqlen-1]
#aa = (aa -  np.mean(aa)) / np.std(aa, ddof=1)
#bb = (bb - np.mean(bb)) / np.std(bb, ddof=1)

dist, _, _, _ = dtw.dtw(aa, bb)
best_so_far = dist

############## Hierarchically downsample(i:1 to 1:1), compute Lower bound MPs and prune TS ##############
i=64
dnc = np.copy(silence_mask)
lb_t = []
sp_rates = []
prrate = []

while len(a)/i >= 102:
    print(f'------ i = {i} (best so far={best_so_far})------')
    window_size = round(len(a)/i)
    if round(window_size) == 1:
        ds = a
        dncs = dnc
    else:
        ds = PAA(a, window_size)
        dncs = PAA(dnc, window_size)
    
    if round(subseqlen*len(ds)/len(a)) < 4:
        i *= 2
        continue

    
    ######## 1. Compute Downsampled Lower bound #######
    t = time.time()
    print('1. Compute Downsampled Lower bound')
    mpa_lbk, mpi_lbk = LB_Keogh_mp_updated(ds, max(round(subseqlen*len(ds)/len(a)),4), max(round(subseqlen*len(ds)/len(a)),4), maxwarp, dncs)
    sp_rates.append(round(window_size))
    print(f'   time={time.time()-t}')

    
    ########## 2. Upsample the vector ##########
    t = time.time()
    print('2. Upsample the vector')
    mpa_lbk_stretched = np_matlab.repmat(mpa_lbk, window_size, 1) #np.tile(mpa_lbk, int(np.floor(len(a)/len(ds)))) 
    scale = window_size**0.5
    mpa_lbk_stretched = scale*mpa_lbk_stretched
    mpa_lbk_stretched = np.concatenate(np.column_stack(mpa_lbk_stretched))
    #len_stretch = min([len(mp_ed), len(mpa_lbk_stretched)])
    mpa_lbk_stretched = mpa_lbk_stretched[0:len(mp_ed)]
    print(f'   time={time.time()-t}')

    
    ########### 3. Update Best so far if needed ###########
    t = time.time()
    print('3. Update best so far if needed')
    if np.all(mpa_lbk_stretched == np.Inf):
        pr_rate = 0
        DTW_time = 0
        prrate.append(sum(dnc))
        motiffirst = first_min
        motifsec = sec_min
        #return

    mpa_lbk_stretched = mpa_lbk_stretched #np.copy(mp_ed)
    samind = np.argmin(mpa_lbk_stretched)
    temp = np.copy(mpa_lbk_stretched)
    mpa_lbk_stretched[max([samind-subseqlen, 0]):samind+subseqlen] = np.Inf
    samind_sec = np.argmin(mpa_lbk_stretched)
    aa = a[samind:samind+subseqlen-1]
    bb = a[samind_sec:samind_sec+subseqlen-1]
    aa = (aa -  np.mean(aa)) / np.std(aa, ddof=1)
    bb = (bb - np.mean(bb)) / np.std(bb, ddof=1)

    dist, _, _, _ = dtw.dtw(aa, bb) #maxwarp

    if dist < best_so_far:
        best_so_far = dist
        first_min = samind
        sec_min = samind_sec
    print(f'   time={time.time()-t}')

    
    ############ 4. Prune time series if needed ###########  
    t = time.time()
    print('4. Prune time series if needed')
    mpa_lbk_stretched = temp
    dnc[np.where(mpa_lbk_stretched > best_so_far)[0]] = 1 ### pruning vector   
    prrate.append(sum(dnc))
    mpa_lbk_stretched[np.where(dnc[:len(mpa_lbk_stretched)]==1)[0]] = np.nan

    i *= 2
    print(f'   time={time.time()-t}')

mpa_lbk_stretched[np.where(dnc[:len(mpa_lbk_stretched)]==1)[0]] = np.nan


########### If all were pruned, return the ED motifs ###########
if sum(dnc)>=len(a)-subseqlen:
    pr_rate = 0
    DTW_time = 0
    motiffirst=first_min
    motifsec = sec_min
    # return

######## Compute DTW Matrix Profile for the pruned TS ##########
mp_sorted = np.array(sorted(mpa_lbk_stretched))




####### dtw_mp #################
#pr_rate,best_so_far,first_min,sec_min = dtw_mp(a, mpi_lbk, mp_sorted, subseqlen, minspacing, maxwarp,dnc,best_so_far,first_min,sec_min)
ts = a
mp_ind = mpi_lbk
mp_sorted = mp_sorted
subseqlen = subseqlen
minlag = minspacing
warpmax = maxwarp
dnc = dnc
best_so_far = best_so_far
motiffirst = first_min
motifsec = sec_min


# return lb_t, sp_rates, pr_rate, prrate, best_so_far, first_min, sec_min, DTW_time
