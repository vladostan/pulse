clc; close all; clear;

load('data/G1data.mat');
% load('data/G2data.mat');
% load('data/V1data.mat');
% load('data/V2data.mat');
% load('data/facedata.mat');

V.FrameRate = 50;
V.NumberOfFrames = size(y,2);

y_interp = cubicSplineInterp(V, y);

y_stable = removeUnstable(y_interp);

y_filtered = temporalFiltering(y_stable);

% Perform PCA
y_pca = PCA(y_filtered,5);

% Perform Fast ICA
y_fica = fastICA(y_filtered,5,'negentropy');
y_fica2 = fastica2(y_filtered,'numOfIC',5);

% Perform max-kurtosis ICA
y_kica = kICA(y_filtered,5);

% Perform Jade
[~,y_jade] = jade(y_filtered,5);

% Perform Shibbs
y_shibbs = shibbs(y_filtered',5);

%plot components
for i = 1:5
    subplot(2,3,1)
        plot(y_pca(i,:))
        title('PCA')
        
    subplot(2,3,2)
        plot(y_fica(i,:))
        title('Fast ICA')
        
    subplot(2,3,3)
        plot(y_fica2(i,:))
        title('Fast ICA 2')

    subplot(2,3,4)
        plot(real(y_kica(i,:)))
        title('Kurtosis ICA')

    subplot(2,3,5)
        plot(y_jade(i,:))
        title('Jade')

    subplot(2,3,6)
        plot(y_shibbs(i,:))
        title('Shibbs')

    ginput(1);
end

%% TODO
%Intuitive selection of component
[signal_y, signal_number_y] = signalSelection(y_filtered, y_eigVecs);

%Plot components
for i = 1:size(signal_y, 2)
    figure(i); plot(signal_y(:,i));
end

T = size(y_filtered,2);

pks = findpeaks(signal_y(:,1));
[pks, locs] = findpeaks(signal_y(:,1),'MinPeakHeight',max(pks)/40,'MinPeakDistance',frameRate/(150/60)*T/numFr);
pulse = 60*V.frameRate*T/V.numFr./diff(locs)'
averagePulse = mean(pulse)