function av = averagePulse(signal, fsamp) 

% Frequency domain average pulse calculation

T = 1/fsamp;                     % Sample time
L = length(signal);              % Length of signal

% figure(1)
% plot(signal)

NFFT = 2^nextpow2(L); % Next power of 2 from length of signal
Y = fft(signal,NFFT)/L;
f = fsamp/2*linspace(0,1,NFFT/2+1);
amp = 2*abs(Y(1:NFFT/2+1));

% Plot single-sided amplitude spectrum.
% figure(2)
% plot(f,amp) 
% title('Single-Sided Amplitude Spectrum of y(t)')
% xlabel('Frequency (Hz)')
% ylabel('|Y(f)|')

av = f(amp == max(amp))*60;