function data_filtered = temporalFiltering(data_stable)

% 5th order Butterworth filter
% 0.75 to 5 Hz @ 250 Hz sampling rate

[b,a] = butter(5,[0.75 5]/125);

data_filtered = zeros(size(data_stable,1), size(data_stable,2));
for i = 1:size(data_stable,1)
    data_filtered(i,:) = filtfilt(b, a, data_stable(i,:)); 
end

%LOWPASS FIR
% Fpass = 0.75;
% Fstop = 5;
% Fs = 250;
% 
% d = designfilt('lowpassfir', ...
%   'PassbandFrequency',Fpass,'StopbandFrequency',Fstop, ...
%   'DesignMethod','equiripple','SampleRate',Fs);
% 
% data_filtered = zeros(size(data_stable,1), size(data_stable,2));
% for i = 1:size(data_stable,1)
%     data_filtered(i,:) = filter(d, data_stable(i,:)); 
% end

% figure, plot(y_filtered(1,:));
end