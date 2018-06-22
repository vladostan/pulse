clc; close all; clear;

addpath(genpath('ComponentAnalysis'), 'data');

% XY DATA
load('data/xy_data/p1_normal_xy.mat');
% load('data/xy_data/pid_physical_xy.mat');

% CA DATA
load('data/ca_data/pid_normal_ca.mat');
% load('data/ca_data/pid_physical_ca.mat');

% Videos
V = VideoReader('data/Videos/p1_normal.mp4');
% V = VideoReader('data/Videos/pid_physical.mp4');

 y_interp = cubicSplineInterp(V, y, 250);

 y_stable = removeUnstable(y_interp);

 y_filtered = temporalFiltering(y_stable);

% clear V x y y_interp y_stable 

%% Component Analysis Part
% Perform PCA
% tic
y_pca = PCA(y_filtered,5);
% time_pca = toc;

% Perform Fast ICA
% tic
y_fica = fastica(y_filtered, 'lastEig', 10, 'numOfIC', 5);
% time_fica = toc;

% Perform Jade
% tic
[~,y_jade] = jade(y_filtered,5);
% time_jade = toc;

% Perform Shibbs
% tic
y_shibbs = shibbs(y_filtered',5);
% time_shibbs = toc;

%save('data/ca_data/pidca.mat','y_pca','y_fica','y_jade','y_shibbs','time_pca','time_fica','time_jade','time_shibbs');
%csvwrite('v1pca.csv', y_pca');
%csvwrite('v1fica.csv', y_fica');
%csvwrite('v1jade.csv', y_jade');
%csvwrite('v1shibbs.csv', y_shibbs');

%% Plot components
for i = 1:5
    subplot(2,2,1)
        plot(y_pca(i,:))
        title(['PCA ' num2str(time_pca) ' sec'])
        
    subplot(2,2,2)
        plot(y_fica(i,:))
        title(['Fast ICA ' num2str(time_fica) ' sec'])

    subplot(2,2,3)
        plot(y_jade(i,:))
        title(['Jade ' num2str(time_jade) ' sec'])

    subplot(2,2,4)
        plot(y_shibbs(i,:))
        title(['Shibss ' num2str(time_shibbs) ' sec'])

    ginput(1);
end