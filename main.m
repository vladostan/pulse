clc; clear; close;

% READ AND CREATE FRAMES:
V = VideoReader('Videos\face.mp4');
% V = VideoReader('Videos\face2.mp4');
% V = VideoReader('Videos\V66aCUT.mp4');
% V = VideoReader('Videos\V66bCUT.mp4');
% V = VideoReader('Videos\V85CUT.mp4');
% V = VideoReader('Videos\V118CUT.mp4');

frameRate = V.FrameRate;
numFr = V.NumberOfFrames;

[forehead, nose] = roi(V);
% imshow(insertShape(read(V,1), 'rectangle', nose));

[x, y] = featureTracking(V, forehead, nose);
% imshow(insertMarker(read(V,1),[x(:,1) y(:,1)],'+'));
% figure, imshow(insertMarker(read(V,1),[x2(:,1) y2(:,1)],'+'));

x_interp = cubicSplineInterp(V, x);
y_interp = cubicSplineInterp(V, y);

x_stable = removeUnstable(x_interp);
y_stable = removeUnstable(y_interp);

x_filtered = temporalFiltering(x_stable);
y_filtered = temporalFiltering(y_stable);

x_eigVecs1 = applyPCA(x_filtered, 10);
y_eigVecs1 = applyPCA(y_filtered, 10);

x_eigVecs2 = applyKICA(x_filtered, 10);
y_eigVecs2 = applyKICA(y_filtered, 10);

x_eigVecs3 = applyJade(x_filtered, 10);
y_eigVecs3 = applyJade(y_filtered, 10);

[signal_x, signal_number_x] = signalSelection(x_filtered, x_eigVecs3);
[signal_y, signal_number_y] = signalSelection(y_filtered, y_eigVecs3);

% peakDetection
% pks_x = findpeaks(signal_x(:,signal_number_x));
% pks_y = findpeaks(signal_y(:,signal_number_y));

% :::PULSE:::
% x-component
T = size(x_filtered,2);

for i = 1:size(signal_x, 2)
    figure(i); plot(signal_x(:,i));
end
close all;

pks = findpeaks(signal_x(:,1));
[pks, locs] = findpeaks(signal_x(:,1),'MinPeakHeight',max(pks)/40,'MinPeakDistance',frameRate/(150/60)*T/numFr);
pulse = 60*frameRate*T/numFr./diff(locs)'
averagePulse = mean(pulse)

% y-component
T = size(y_filtered,2);

for i = 1:size(signal_y, 2)
    figure(i); plot(signal_y(:,i));
end
close all;

pks = findpeaks(signal_x(:,1));
[pks, locs] = findpeaks(signal_y(:,1),'MinPeakHeight',max(pks)/40,'MinPeakDistance',frameRate/(150/60)*T/numFr);
pulse = 60*frameRate*T/numFr./diff(locs)'
averagePulse = mean(pulse)

%face
%True = 53 approximately
%PCA x = 52.5 av, y = 64.4 av
%ICA x = 55.22 av, y = 54.48 av
%Jade x = 52.51 av, y = 53.90 av

%face2
%True = 54-55 approximately
%PCA x = 78 av, y = 73 av
%ICA x = 107 av, y = 83 av
%Jade x = 106 av, y = 99 av

%V66a
%True = 66
%PCA x = 67.14 av, y = 64.50 av
%ICA x = 63.71 av, y = 77.73 av
%Jade x = 63.71 av, y = 77.68 av

%V66b
%True = 66
%PCA x = 73.2 av, y = 78.7 av
%ICA x = 81 av, y = 90.4 av
%Jade x = 73.2 av, y = 78.7 av

%V85
%True = 85
%PCA x = 67.2 av, y = 90.5 av
%ICA x = 89.9 av, y = 89.9 av
%Jade x = 83.8 av, y = 90.1 av

%V118
%True = 118
%PCA x = 92 av, y = 78 av
%ICA x = 79 av, y = 85 av
%Jade x = 81 av, y = 95 av