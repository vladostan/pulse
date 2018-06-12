clc; close all; clear;

addpath(genpath('ComponentAnalysis'),'ca_data');

% OLD STUFF
% load('data/xy_data/G1data.mat');
% load('data/xy_data/G2data.mat');
% load('data/xy_data/V1data.mat');
% load('data/xy_data/V2data.mat');
% load('data/xy_data/facedata.mat');

% load('data\ca_data\g1ca.mat');
% load('data\ca_data\g2ca.mat');
% load('data\ca_data\v1ca.mat');
% load('data\ca_data\v2ca.mat');
% load('data\ca_data\faceca.mat');

% NEW STUFF
load('data/xy_data/Vlad_normaldata.mat');
% load('data/xy_data/Vlad_phdata.mat');
% load('data/xy_data/Geesara_normaldata.mat');
% load('data/xy_data/Geesara_phdata.mat');

load('data/ca_data/Vlad_normalca.mat'); %63
% load('data/ca_data/Vlad_phca.mat'); %73
% load('data/ca_data/Geesara_normalca.mat'); %76
% load('data/ca_data/Geesara_phca.mat'); %77

V = VideoReader('data\Videos\Vlad_normal.mp4');
% V = VideoReader('data\Videos\Vlad_ph.mp4');
% V = VideoReader('data\Videos\Geesara_normal.mp4');
% V = VideoReader('data\Videos\Geesara_ph.mp4');

%  y_interp = cubicSplineInterp(V, y, 250);
% 
%  y_stable = removeUnstable(y_interp);
% 
%  y_filtered = temporalFiltering(y_stable);

% clear V x y y_interp y_stable 

%% Component Analysis Part
% Perform PCA
% tic
% y_pca = PCA(y_filtered,5);
% time_pca = toc;

% Perform Fast ICA
% tic
% y_fica = fastica(y_filtered, 'lastEig', 10, 'numOfIC', 5);
% time_fica = toc;

% Perform max-kurtosis ICA
% tic
% y_mkica = mkICA(y_filtered,5);
% time_mkica = toc;

% Perform Jade
% tic
% [~,y_jade] = jade(y_filtered,5);
% time_jade = toc;

% Perform Shibbs
% tic
% y_shibbs = shibbs(y_filtered',5);
% time_shibbs = toc;

% Perform fast RADICAL (VERY SLOW!!!)
% [y_radical, ~] = fast_RADICAL(y_filtered);

%save('faceca.mat','y_pca','y_fica','y_mkica','y_jade','y_shibbs','time_pca','time_fica','time_mkica','time_jade','time_shibbs');
%csvwrite('v1pca.csv', y_pca');
%csvwrite('v1fica.csv', y_fica');
%csvwrite('v1mkica.csv', y_mkica');
%csvwrite('v1jade.csv', y_jade');
%csvwrite('v1shibbs.csv', y_shibbs');

%% Plot components
for i = 1:5
    subplot(2,3,1)
        plot(y_pca(i,:))
        title(['PCA ' num2str(time_pca) ' sec'])
        
    subplot(2,3,2)
        plot(y_fica(i,:))
        title(['Fast ICA ' num2str(time_fica) ' sec'])

    subplot(2,3,3)
        plot(real(y_mkica(i,:)))
        title(['Max-Kurtosis ICA ' num2str(time_mkica) ' sec'])

    subplot(2,3,4)
        plot(y_jade(i,:))
        title(['Jade ' num2str(time_jade) ' sec'])

    subplot(2,3,5)
        plot(y_shibbs(i,:))
        title(['Shibss ' num2str(time_shibbs) ' sec'])

    ginput(1);
end

%% Calculate average pulse for every component of each algorithm
fsamp = 250;
fprintf('PCA\n')
for i = 1:size(y_pca,1)
    averagePulse(y_pca(i,:), fsamp)
end
fprintf('FAST ICA\n')
for i = 1:size(y_fica,1)
    averagePulse(y_fica(i,:), fsamp)
end
fprintf('MAX KURTOSIS ICA\n')
for i = 1:size(y_mkica,1)
    averagePulse(y_mkica(i,:), fsamp)
end
fprintf('JADE\n')
for i = 1:size(y_jade,1)
    averagePulse(y_jade(i,:), fsamp)
end
fprintf('SHIBBS\n')
for i = 1:size(y_shibbs,1)
    averagePulse(y_shibbs(i,:), fsamp)
end