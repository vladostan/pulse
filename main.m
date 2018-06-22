clc; clear; close;

% Videos
V = VideoReader('data\Videos\pid_normal.mp4');
% V = VideoReader('data\Videos\pid_physical.mp4');

frameRate = V.FrameRate;
numFr = V.NumberOfFrames;

[forehead, nose] = roi(V);
% foreheadnose = insertShape(read(V,1), 'rectangle', forehead, 'LineWidth', 5, 'Color', 'red');
% foreheadnose = insertShape(foreheadnose, 'rectangle', nose, 'LineWidth', 5, 'Color', 'red');
% imshow(foreheadnose);

[x, y] = featureTracking(V, forehead, nose);
imshow(insertMarker(foreheadnose,[x(:,1) y(:,1)],'*', 'Size', 5, 'Color', 'green'));

% save('data/xy_data/pid_physical_xy.mat','x','y');

samplingRate = 250;

x_interp = cubicSplineInterp(V, x, samplingRate);
y_interp = cubicSplineInterp(V, y, samplingRate);

x_stable = removeUnstable(x_interp);
y_stable = removeUnstable(y_interp);

x_filtered = temporalFiltering(x_stable);
y_filtered = temporalFiltering(y_stable);

%% Component Analysis Part
addpath(genpath('ComponentAnalysis'),'ca_data');

% Perform PCA
tic
y_pca = PCA(y_filtered,5);
time_pca = toc;

% Perform Fast ICA
tic
y_fica = fastica(y_filtered, 'lastEig', 10, 'numOfIC', 5);
time_fica = toc;

% Perform Jade
tic
[~,y_jade] = jade(y_filtered,5);
time_jade = toc;

% Perform Shibbs
tic
y_shibbs = shibbs(y_filtered',5);
time_shibbs = toc;

% save('data/ca_data/pid_physical_ca.mat','y_pca','y_fica','y_jade','y_shibbs','time_pca','time_fica','time_jade','time_shibbs');