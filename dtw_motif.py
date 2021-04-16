import numpy as np

import ctypes
import sys
import os 

from simmusic.dtw import dtw

dir_path = ''#os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(os.path.join(dir_path,"dtw_upd.so"))

handle.compute_dtw.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double]

def compute_dtw(A, B, cb, m, r, best_so_far):
    return handle.compute_dtw(A, B, cb, m, r, best_so_far)

def dtw_motif(a, subseqlen, maxwarp, mp_ed):
    """
    TODO: Create documentation
    """
    mp_ed = np.copy(mp_ed)

    minspacing = subseqlen

    ################### first best-so-far=ED motif distance ###########################
    first_min = np.argmin(mp_ed)
    best_so_far = mp_ed[first_min]

    mp_ed[max(first_min - subseqlen + 1, 0):first_min+subseqlen-1] = np.PINF
    
    sec_min = np.argmin(mp_ed)

    ################### second best-so-far=DTW distance between ED motifs ###########################
    aa = a[first_min:first_min + subseqlen-1]
    bb = a[sec_min:sec_min + subseqlen-1]
    aa = (aa -  np.mean(aa)) / np.std(aa, ddof=1)
    bb = (bb - np.mean(bb)) / np.std(bb, ddof=1)
    dist = dtw.dtw_vector(aa, bb)
    if dist < best_so_far:
        best_so_far = dist
    
#################### Hierarchically downsample(i:1 to 1:1), compute Lower bound MPs and prune TS ######################
i=64
dnc = zeros(1,length(a))
lb_t = []
k = 1

while (length(a)/i) >= 0.75
    if (round(length(a)/i) == 1)
        ds = a
        dncs = dnc
    else
        ds = PAA_updated(a,i)#newpaa(a,i) 
        dncs = PAA_updated(dnc,i) #newpaa(dnc,i)#
    end
    tic    
    if round(subseqlen*length(ds)/length(a)) < 4
        i = i*2       
        continue
    end
############## 1. Compute Downsampled Lower bound ##################
    [mpa_lbk, mpi_lbk] = LB_Keogh_mp_updated(ds, max(round(subseqlen*length(ds)/length(a)),4), max(round(subseqlen*length(ds)/length(a)),4), maxwarp, dncs)    
    lb_t(k)=toc
    sp_rates(k) = round(length(a)/i)
    
############## 2. Upsample the vector ##################
    mpa_lbk_stretched = repmat(mpa_lbk', floor(length(a)/length(ds)), 1)
    scale = sqrt(length(a)/i)
    mpa_lbk_stretched = scale*mpa_lbk_stretched(:) 
    len_stretch = min(length(mp_ed),length(mpa_lbk_stretched))
    mpa_lbk_stretched = mpa_lbk_stretched(1:len_stretch)
    
############## 3. Update Best so far if needed ##################
    if all(mpa_lbk_stretched == Inf)
        pr_rate = 0
        DTW_time = 0
        prrate(k) = sum(dnc)
        motiffirst=first_min
        motifsec = sec_min
        return
    end
    [~,samind]=min(mpa_lbk_stretched)
    temp = mpa_lbk_stretched
    mpa_lbk_stretched(max(samind-subseqlen+1,1):samind+subseqlen-1) = inf
    [~,samind_sec]=min(mpa_lbk_stretched)         
    aa = a(samind:samind+subseqlen-1)
    bb = a(samind_sec:samind_sec+subseqlen-1)
    aa = (aa -  mean(aa)) ./ std(aa,1)
    bb = (bb - mean(bb)) ./ std(bb,1)
    dist = dtw_upd(aa', bb', maxwarp)    
    if dist < best_so_far        
        best_so_far = dist
        first_min = samind
        sec_min = samind_sec 
        
        
        
    end
    
############## 4. Prune time series if needed ##################  
    mpa_lbk_stretched = temp    
    dnc(mpa_lbk_stretched > best_so_far) = 1 ### pruning vector   
    prrate(k) = sum(dnc)#(sum(mpa_lbk_stretched > best_so_far))  
    mpa_lbk_stretched(dnc==1)=nan    
#     to_plot = a
#     to_plot(dnc~=1) = nan
#     figure plot(a)          
#     hold on plot(to_plot,'r','Linewidth',2)
    i = i*2
    k = k+1    
end
mpa_lbk_stretched(dnc==1)=nan

############## If all were pruned, return the ED motifs ##################  
if sum(dnc)>=length(a)-subseqlen       
    pr_rate = 0
    DTW_time = 0
    motiffirst=first_minmotifsec = sec_min
    
    
    
    return
end

################ Compute DTW Matrix Profile for the pruned TS ###########
[mp_sorted,~] = sort(mpa_lbk_stretched)
[pr_rate,best_so_far,first_min,sec_min,DTW_time] = dtw_mpGUI(a', mpi_lbk, mp_sorted, subseqlen, minspacing, maxwarp,dnc,best_so_far,


end

return lb_t, sp_rates, pr_rate, prrate, best_so_far, first_min, sec_min, DTW_time
