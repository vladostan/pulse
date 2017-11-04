function  [R,y]=cubica4a(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CubICA (IMPROVED CUMULANT BASED ICA-ALGORITHM)
%
% This algorithm performes ICA by diagonalization of fourth-order cumulants.
%
%  [R,y]=cubica4a(x)
%
% - x is and NxP matrix of observations
%     (N: Number of components; P: Number of datapoints(samplepoints)) 
% - R is an NxN matrix such that u=R*x, and u has 
%   (approximately) independent components.
% - y is an NxP matrix of independent components
% 
% Ref: T. Blaschke and L. Wiskott, "An Improved Cumulant Based
% Method for Independent Component Analysis", Proc. ICANN-2002,
% Madrid, Spain, Aug. 27-30.
%
% questions, remarks, improvements, problems to: t.blaschke@biologie.hu-berlin.de.
%
% Copyright : Tobias Blaschke, t.blaschke@biologie.hu-berlin.de.
%
% 2002-02-22
%  
%
% Last change:2003-05-19 
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
  [N,P]=size(x);

  Q=eye(N);
  resolution=0.001;
  schalter=1;
  
  % centering and whitening 
  
  fprintf('\ncentering and whitening!\n\n');
  
  x=x-mean(x,2)*ones(1,P);
  [V,D]=eig(x*x'/P);
  W=diag(real(diag(D).^(-0.5)))*V';
  y=W*x;
  
  fprintf('rotating\n'); 
  
  % start rotating
  
  for t=1:(1+round(sqrt(N))),      
    for i=1:N-1,
      for j=i+1:N,
	
	%calculating the new cumulants
	
	u=y([i j],:);
	
	sq=u.^2;
	
	sq1=sq(1,:);
	sq2=sq(2,:);
	u1=u(1,:)';
	u2=u(2,:)';
	
	C111=sq1*u1/P;
	C112=sq1*u2/P;
	C122=sq2*u1/P;
	C222=sq2*u2/P;
	
	C1111=sq1*sq1'/P-3;
	C1112=(sq1.*u1')*u2/P;
	C1122=sq1*sq2'/P-1;
	C1222=(sq2.*u2')*u1/P;
	C2222=sq2*sq2'/P-3;
	
	% coefficients
	
	c_44=(1/16)*(7*(C1111^2+C2222^2)-16*(C1112^2+C1222^2)-12*(C1111*C1122+C1122*C2222)-36*C1122^2-32*C1112*C1222-2*C1111*C2222);
	
	s_44=(1/32)*(56*(C1111*C1112-C1222*C2222)+48*(C1112*C1122-C1122*C1222)+8*(C1111*C1222-C1112*C2222));

	% calculating the angle 
	
	phi_max=0.25*atan2(s_44,c_44);

	%Givens-rotation-matrix Q_ij
	  
	Q_ij=eye(N);

	c=cos(phi_max);
	s=sin(phi_max);
	
	Q_ij(i,j)=s;
	Q_ij(j,i)=-s;
	Q_ij(i,i)=c;
	Q_ij(j,j)=c;
	
	Q=Q_ij*Q;

	% rotating y
	
	y([i j],:)=[c s;-s c]*u;
	
      end %j
    end %i
  end %t
    
  R=Q*W;

  return