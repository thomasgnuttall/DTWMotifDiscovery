import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation
from matlab import movmean

def PAA(timeseries, window_size):
    transformer = PiecewiseAggregateApproximation(window_size=window_size)
    return transformer.transform(np.array([timeseries]))[0]

def PAA_from_paper(timeseries, numcoeffs):
    """
    This is not a very good approach. I do not recommend it. I recommend
    truncating sections that do not fill PAA sections, because it maintains
    correct correspondence between PAA output and time series input. This
    does __not__. 
    """
    assert isinstance(timeseries, np.ndarray), "<timeseries> should be an np.array"
    
    N = len(timeseries)
    if any([N < 1, N < numcoeffs]):
        raise Exception('Invalid input')

    per_section = N/numcoeffs
    if np.floor(per_section) == per_section:
        PAA = movmean(timeseries, [0, per_section-1])
    else:
        PAA = np.empty((numcoeffs, ))
        PAA[:] = np.nan
        paa_index = 0
        ts_index = 0
        carry = 1
        while ts_index < N and paa_index < numcoeffs:
            if carry > 0:
                if carry > 1:
                    Exception('Invalid carry, something is wrong')
                p = carry * timeseries[int(ts_index)]
                ts_index += 1
                post_carry = per_section - carry
                full = np.floor(post_carry)
                fractional = post_carry - full
            else:
                p = 0
                full = np.floor(per_section)
                fractional = per_section - full
            if ts_index + full > N:
                print('!Possible loss of precision in determining intervals!')
                # just run it to the end. It may not be correct. Floating point
                # arithmetic for this kind of thing is a bit tenuous and not
                # always well conditioned
                
                # This isn't the only way it can fail. This just happens to be
                # easily detectable. Notice, skips fractional component, which
                # is definitely corrupted by roundoff if you're hitting this
                # point.
                full = N - ts_index
                p = p + sum(timeseries[int(ts_index):int(ts_index+full-1)])   
                PAA[paa_index] = p
                break
            ts_index = ts_index + full
            if ts_index <= N and fractional > 0:
                p = p + fractional * timeseries[int(ts_index)]
                carry = 1 - fractional 
            else:
                if fractional < 0:
                    # again, not the only way it can fail, just simple and
                    # obvious
                    print('Loss of precision in carry value due to rounding')
                carry = 0
            PAA[paa_index] = p
            paa_index += 1
        PAA = PAA/per_section
        return PAA
