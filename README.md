## Setup ##

Initial data matrix y (size NxM), where:
N = number of feature points
M = number of frames (time in seconds * 50 fps)

Interpolated data matrix y_filtered (N*5M):
Applied cubic spline interpolation from 50 to 250 HZ

Stable data matrix y_stable (aN*5M):
Some unstable feature points are removed, 0 < a < 1

Filtered data matrix y_filtered (aN*5M):
Butterworth 5th filter applied

Component Analysis matrix y_xxx (b*5M):
b = number of desired extracted components (default = 5)
