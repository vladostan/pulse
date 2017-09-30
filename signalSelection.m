function [signal, signal_number] = signalSelection(data_filtered, eigVecs)

m = data_filtered';
T = size(data_filtered,2);

% Clearest dominant frequency
% 1-D position signal s(t):
signal = m*eigVecs;

% Choosing signal with best periodicity

percentage = 0;
for j = 1:size(signal,2)
    x = signal(:,j);
    % Signal's periodicity:
    pxx = periodogram(x,rectwin(length(x)),length(x),T);

    sum = 0;
    for i = 1:length(pxx)
    sum = sum + pxx(i); 
    end
    if 100*max(pxx)/sum > percentage
        percentage = 100*max(pxx)/sum;
        signal_number = j;
    end
end
end


