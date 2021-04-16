import stumpy

from dtw_motif import dtw_motif

def DTWMotifDiscovery(TS, subseqlen, maxwarp):
	"""
	TODO: Create documentation
	"""
    mp, _, _, _ = mpx_v2(TS, subseqlen, subseqlen) # mpx(TS, subseqlen, subseqlen)

    lb_t, sp_rates, pr_rate, prrate, best_so_far, motiffirst, motifsec, DTW_time = dtw_motif(TS, subseqlen, maxwarp, mp)

    return best_so_far, motiffirst, motifsec


def mpx_v2(timeSeries, minlag, subseqlen):
	"""
	TODO: Get documentation from matlab code
	"""
	stump_ret = stumpy.stump(timeSeries, m=subseqlen)

	matrixProfile = stump_ret[:, 0]
	matrixProfileIdx = stump_ret[:, 1]
	motifsIdx = stump_ret[:, 2]
	discordsIdx = stump_ret[:, 3]

	return matrixProfile, matrixProfileIdx, motifsIdx, discordsIdx