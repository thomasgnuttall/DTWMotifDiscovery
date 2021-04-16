import numpy as np

def movmean(A, k, discard_endpoints=True, std=False):
	"""
	M = movmean(A, k) returns an array of local k-point mean values, 
	where each mean is calculated over a sliding window of length k
	across neighboring elements of A. 
	"""

	k1 = k[0]
	k2 = k[1]
	
	new_array = []
	for i in range(len(A)):
		low = i-k1
		high = i+k2+1
		if low < 0:
			if discard_endpoints:
				continue
			else:
				low = 0

		if high > len(A):
			if discard_endpoints:
				continue
			else:
				high = len(A)

		this = A[low:high]
		if std:
			to_append = np.std(this, ddof=1)
		else:
			to_append = np.mean(this)
		new_array.append(to_append)
	return np.array(new_array)


def movmax(A, k, discard_endpoints=False, return_min=False):
	"""
	M = movmax(A,k) returns an array of local k-point maximum values, 
	where each maximum is calculated over a sliding window of length 
	k across neighboring elements of A
	"""

	k1 = k[0]
	k2 = k[1]
	
	new_array = []
	for i in range(len(A)):
		low = i-k1
		high = i+k2+1
		if low < 0:
			if discard_endpoints:
				continue
			else:
				low = 0

		if high > len(A):
			if discard_endpoints:
				continue
			else:
				high = len(A)

		this = A[low:high]
		if return_min:
			to_append = min(this)
		else:
			to_append = max(this)
		new_array.append(to_append)
	return np.array(new_array)
