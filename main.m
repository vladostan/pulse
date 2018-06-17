clc; clear; close;

% Videos
% V = VideoReader('data\Videos\Vlad_normal.mp4');
% V = VideoReader('data\Videos\Vlad_ph.mp4');
% V = VideoReader('data\Videos\Geesara_normal.mp4');
% V = VideoReader('data\Videos\Geesara_ph.mp4');
% V = VideoReader('data\Videos\Stanislav_normal.mp4');
% V = VideoReader('data\Videos\Stanislav_ph.mp4');
% V = VideoReader('data\Videos\Ilya_normal.mp4');
% V = VideoReader('data\Videos\Ilya_ph.mp4');
% V = VideoReader('data\Videos\Alexander_normal.mp4');
% V = VideoReader('data\Videos\Alexander_ph.mp4');
% V = VideoReader('data\Videos\Mikhail_normal.mp4');
% V = VideoReader('data\Videos\Mikhail_ph.mp4');
% V = VideoReader('data\Videos\Vadim_normal.mp4');
% V = VideoReader('data\Videos\Vadim_ph.mp4');
% V = VideoReader('data\Videos\Maksim_normal.mp4');
% V = VideoReader('data\Videos\Maksim_ph.mp4');
% V = VideoReader('data\Videos\Sergey_normal.mp4');
% V = VideoReader('data\Videos\Sergey_ph.mp4');
% V = VideoReader('data\Videos\Mike_normal.mp4');
% V = VideoReader('data\Videos\Mike_ph.mp4');

frameRate = V.FrameRate;
numFr = V.NumberOfFrames;

[forehead, nose] = roi(V);
foreheadnose = insertShape(read(V,1), 'rectangle', forehead, 'LineWidth', 5, 'Color', 'red');
foreheadnose = insertShape(foreheadnose, 'rectangle', nose, 'LineWidth', 5, 'Color', 'red');
imshow(foreheadnose);

[x, y] = featureTracking(V, forehead, nose);
% imshow(insertMarker(foreheadnose,[x(:,1) y(:,1)],'*', 'Size', 5, 'Color', 'green'));

%%
save('data/xy_data/Mike_normaldata.mat','x','y');

samplingRate = 250;

x_interp = cubicSplineInterp(V, x, samplingRate);
y_interp = cubicSplineInterp(V, y, samplingRate);
% imshow(insertMarker(foreheadnose,[x_interp(:,1) y_interp(:,1)],'*', 'Size', 5, 'Color', 'green'));

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

% Perform max-kurtosis ICA
tic
y_mkica = mkICA(y_filtered,5);
time_mkica = toc;

% Perform Jade
tic
[~,y_jade] = jade(y_filtered,5);
time_jade = toc;

% Perform Shibbs
tic
y_shibbs = shibbs(y_filtered',5);
time_shibbs = toc;

save('data/ca_data/Mike_normalca.mat','y_pca','y_fica','y_mkica','y_jade','y_shibbs','time_pca','time_fica','time_mkica','time_jade','time_shibbs');

%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Other stuff
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