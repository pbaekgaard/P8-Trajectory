#include "trajectory.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "distance.hpp"

double euclideanDistance(const SamplePoint& a, const SamplePoint& b) {
    double dx = a.longitude - b.longitude;
    double dy = a.latitude - b.latitude;
    return std::sqrt(dx * dx + dy * dy);
}

double MaxDTW(const Trajectory& A, const Trajectory& B) {
    const auto& P = A.points;
    const auto& Q = B.points;

    // Base cases
    if (P.empty() && Q.empty()) return 0.0;
    if (P.empty() || Q.empty()) return std::numeric_limits<double>::infinity();

    int n = P.size();
    int m = Q.size();

    // dp[i][j] is MaxDTW for P[0..i] and Q[0..j] (inclusive, 0-based)
    std::vector<std::vector<double>> dp(n, std::vector<double>(m, std::numeric_limits<double>::infinity()));

    // Base case: first points
    dp[0][0] = euclideanDistance(P[0], Q[0]);

    // Fill first row: P[0] aligned with Q[0..j]
    for (int j = 1; j < m; ++j) {
        dp[0][j] = std::max(dp[0][j - 1], euclideanDistance(P[0], Q[j]));
    }

    // Fill first column: P[0..i] aligned with Q[0]
    for (int i = 1; i < n; ++i) {
        dp[i][0] = std::max(dp[i - 1][0], euclideanDistance(P[i], Q[0]));
    }

    // Fill the rest of the table
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            double cost = euclideanDistance(P[i], Q[j]);
            // Min of the three possible alignments, taking max with current cost
            dp[i][j] = std::max(cost, std::min({
                dp[i - 1][j - 1], // Diagonal: match P[i] with Q[j]
                dp[i - 1][j],     // Vertical: skip P[i]
                dp[i][j - 1]      // Horizontal: skip Q[j]
            }));
        }
    }

    return dp[n - 1][m - 1];
}