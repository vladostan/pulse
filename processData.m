clc; close all; clear;

data = load('data/G1data.mat');
% data = load('data/G2data.mat');

y = data.y;

frameRate = 50;
numFr = size(y,2);

V.FrameRate = frameRate;
V.NumberOfFrames = numFr;

y_interp = cubicSplineInterp(V, y);

y_stable = removeUnstable(y_interp);

y_filtered = temporalFiltering(y_stable);

%TODO
% y_eigVecs = %someComponentAnalysisAlgorithm(y_filtered, number_of_extracted_components)
%Ex: y_eigVecs = applyPCA(y_filtered, 5);


%Intuitive selection of component
[signal_y, signal_number_y] = signalSelection(y_filtered, y_eigVecs);

%Plot components
for i = 1:size(signal_y, 2)
    figure(i); plot(signal_y(:,i));
end

T = size(y_filtered,2);

pks = findpeaks(signal_y(:,signal_number_y));
[pks, locs] = findpeaks(signal_y(:,signal_number_y),'MinPeakHeight',max(pks)/40,'MinPeakDistance',frameRate/(150/60)*T/numFr);
pulse = 60*frameRate*T/numFr./diff(locs)'
averagePulse = mean(pulse)