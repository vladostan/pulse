## Description ##

Initial data matrix y (size NxM), where:
    N = number of feature points
    M = number of frames (time in seconds * fps)

Interpolated data matrix y_filtered (N*5M):
    Applied cubic spline interpolation from 'fps' Hz to 250 HZ

Stable data matrix y_stable (aN*5M):
    Some unstable feature points are removed, 0 < a < 1

Filtered data matrix y_filtered (aN*5M):
    Butterworth 5th filter applied

Component Analysis matrix y_xxx (b*5M):
    b = number of desired extracted components (default = 5)

## Data from video ##

    'G1','G2','V1','V2' data: 50 fps
    'face' and other data: 30 fps

## Average Pulse (Ground Truth data) ##

    G1: 85 bpm
    G2: 93 bpm
    V1: 83 bpm
    V2: 106 bpm
    face: 53 bpm