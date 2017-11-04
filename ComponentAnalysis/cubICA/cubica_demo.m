%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo mixes three source signal components
% by a random mixing matrix A and finds the 
% unmixing matrix R with the help of cubica34.
%
% type cubica_demo to start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

 %load sample source signal

 load kennedy.mat

 %generate random mixing matrix
 
 A=rand(N);
 
 %mix source signal components
 
 x=A*s;
 
 %plot source signal components
 
 hdl=figure('Name','CUBICA-DEMO Source Signals','NumberTitle','off','MenuBar','none','Position',[450 600 400 400]);
 
 % define X-axis
 
 t=0:(P/fs/P):(P-1)/fs;
  
 for i=1:3,
 
   subplot(3,1,i)
 
   
   plot(t,s(i,:));
   
   xlim([0 P/fs]);
   
   xlabel('seconds');
   
   ylabel('Amplitude');
 
 end
 
 
 %plot mixed source signal components
 
 hdl=figure('Name','CUBICA-DEMO Mixed Source Signals','NumberTitle','off','MenuBar','none','Position',[860 600 400 400]);
 
 % define X-axis
 
 t=0:(P/fs/P):(P-1)/fs;
  
 for i=1:3,
 
   subplot(3,1,i)
 
   
   plot(t,x(i,:),'r');
   
   xlim([0 P/fs]);
   
   xlabel('seconds');
   
   ylabel('Amplitude');
 
 end

 fprintf('####################################\n');
 fprintf('####Press key to start unmixing!####\n');
 fprintf('####################################\n');
 pause;
 
 %start unmixing
 
 [R,y]=cubica34(x);

 fprintf('\n\nDone!....\n\n\n');
 fprintf('###################################\n');
 fprintf('#####Press key to plot results#####\n');
 fprintf('###################################\n');
 pause;
 
 %plot estimated source signal components
 
 hdl=figure('Name','CUBICA-DEMO Estimated Source Signals','NumberTitle','off','MenuBar','none','Position',[450 170 400 400]);
 
 % define X-axis
 
 t=0:(P/fs/P):(P-1)/fs;
  
 for i=1:3,
 
   subplot(3,1,i)
 
   
   plot(t,y(i,:),'g');
   
   xlim([0 P/fs]);
   
   xlabel('seconds');
   
   ylabel('Amplitude');
 
 end
 
 % compare source signal with estimated source signal
 
 hdl=figure('Name','CUBICA-DEMO Input-Output','NumberTitle', ...
	     'off','MenuBar','none','Position',[860 170 400 400]);
  
 gplotmatrix(s',y')
 
 fprintf('\n\nunimxing Error (0 means perfect unmixing):%2.3f\n',sir(R*A));