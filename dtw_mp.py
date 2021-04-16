from matlab import movmax, movmean

def dtw_mp(ts, mp_ind, mp_sorted, subseqlen, minlag, warpmax, dnc, best_so_far, motiffirst, motifsec, normalize=False):
    
    mu = movmean(ts, [0, subseqlen-1])
    sig = movmean(ts, [0, subseqlen-1], std=True)

    subcount = len(ts) - subseqlen + 1

    lb_kim_time = 0
    lb_keogh_time = 0
    dtw_time = 0
    lb_kim_iter = 0
    lb_keogh_iter = 0
    dtw_iter = 0
    ea_cnt = 0

    tr = dnc == 1
    
    mp_ind[tr] = -1
    
    for i in range(subcount):
        id_ = mp_ind[i]
        if id_ == -1 or dnc[id_] == 1:
            continue
          
        neighbors = list(range(id_+minlag,subcount+1))
        mp_ind[dnc==1] = -1
        neigh = mp_ind[id_]
        neighbors[neighbors==neigh] = []
        neighbors = [neigh, neighbors]
        
        for j in neighbors:
            idp = j        
            if idp == -1 or dnc[idp] == 1:
                continue
                   
            a = ts[id_ : id_ + subseqlen - 1]
            b = ts[idp : idp + subseqlen - 1]

            if normalize:
                a = (a - mu(id_))/sig(id_)
                b = (b - mu(idp))/sig(idp)
            
            ############## Compute LB_Kim and skip DTW computation if higher than best-so-far ##################  
            lb_Kim = max(abs(a[0]-b[0]), abs(a[-1]-b[-1]))
            if lb_Kim >= best_so_far:
                lb_kim_iter = lb_kim_iter + 1
                continue
  
            ############## Compute LB_Keogh and skip DTW computation if higher than best-so-far ##################  
            else:
                Ua = movmax(a, [warpmax, warpmax])
                La = movmax(a, [warpmax, warpmax], return_min=True)
                
                LB_Keogh = lb_upd(b, Ua, La, best_so_far)            
                
                lb_keogh_time = temp + lb_keogh_time            
                if np.sqrt(LB_Keogh) >= best_so_far:
                    lb_keogh_iter = lb_keogh_iter + 1
                    continue
            
            ############## Compute the DTW distance if the previous two checks failed ################## 
            dist,_,_,_ = dtw(a,b) #,warpmax,best_so_far)   
            dtw_iter = dtw_iter + 1

            ############## Update best-so-far if needed ##################
            if dist < best_so_far:
                best_so_far = dist
                motiffirst = id_
                motifsec = idp              
                dnc[mp_ind[mp_sorted>best_so_far]>0] = 1            
                mp_ind[mp_sorted>best_so_far] = -1
                mp_sorted[mp_sorted>best_so_far] = -1

    pr_rate = dtw_iter

    return pr_rate, best_so_far, motiffirst, motifsec