#include<vector>
#include<limits>
#include<algorithm>
#include<cmath>
#include <iostream>

using namespace std;

constexpr double INF{std::numeric_limits<double>::infinity()};

/// Calculate Dynamic Time Wrapping distance
/// A,B: data and query, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band


double* compute_dtw_(double* A, double* B, double *cb, int m, int r, double best_so_far) {
    double *cost;
    double *cost_prev;
    double *cost_tmp;
    int i, j, k;
    double x, y, z, min_cost;
    int ea = 0;
    double* res;
    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two arrays of size O(r).
    res = (double*)calloc(2, sizeof(double));
    cost = (double*) calloc(2 * r + 1, sizeof(double));
    cost_prev = (double*) calloc(2 * r + 1, sizeof(double));
    for (k = 0; k < 2 * r + 1; k++) {
        cost[k] = INF;
        cost_prev[k] = INF;
    }

    for (i = 0; i < m; i++) {
        k = std::max(0, r - i);
        min_cost = INF;

        for (j = std::max(0, i - r); j <= std::min(m - 1, i + r); j++, k++) {
            /// Initialize all row and column
            if ((i == 0) && (j == 0)) {
                double c = (A[0] - B[0]);
                cost[k] = c * c;
                min_cost = cost[k];
                continue;
            }

            if ((j - 1 < 0) || (k - 1 < 0)) {
                y = INF;
            } else {
                y = cost[k - 1];
            }
            if ((i < 1) || (k > 2 * r - 1)) {
                x = INF;
            } else {
                x = cost_prev[k + 1];
            }
            if ((i < 1) || (j < 1)) {
                z = INF;
            } else {
                z = cost_prev[k];
            }

            /// Classic DTW calculation
            double d = A[i] - B[j];
            cost[k] = std::min(std::min( x, y) , z) + d * d;

            /// Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if (cost[k] < min_cost) {
                min_cost = cost[k];
            }
        }

        /// We can abandon early if the current cummulative distance with lower bound together are larger than best_so_far
        if (double((i+r+1))/m<=0.5 && min_cost + cb[i + r + 1] >= best_so_far) {
            free(cost);
            free(cost_prev);
           // cout << "Early abondoned " << i+r+1 << endl;
           // if (((i+r+1)/m) <= 0.01)
                ea = ea + 1;//double((i+r+1))/m;
            
            res[0] = min_cost + cb[i + r + 1];
            res[1] = ea;
            return res;
        }

        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;

    /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    double final_dtw = cost_prev[k];
    free(cost);
    free(cost_prev);
    res[0] = final_dtw;
    res[1] = ea;
    //return final_dtw;
    return res;
}

extern "C" {
    double* compute_dtw(double* A, double* B, double *cb, int m, int r, double best_so_far)
    {
        return compute_dtw_(A, B, cb, m, r, best_so_far);
    }
}