function [R,y]=cubica3(x)
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % IMPROVED CUMULANT BASED ICA-ALGORITHM
  %
  % This algorithm performes ICA on skewsymmetric data by diagonalization of third-order cumulants.
  %
  %  R=comon34(x)
  %
  % - x is and NxP matrix of observations
  % - R is an NxN matrix such that u=R*x, and u has 
  %   (approximately) independent components.
  %
  % This algorithm does exactly (1+round(sqrt(N)) sweeps.
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
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
  [N,P]=size(x);

  Q=eye(N);
  
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
	
	% coefficients
	
	c_34=(1/8)*(3*(C111^2+C222^2)-9*(C112^2+C122^2)-6*(C111*C122+C112*C222));
	
	s_34=(1/4)*(6*(C111*C112-C122*C222));
	
	%calculating the angle

	phi_max=0.25*atan2(s_34,c_34);
	
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
	
	y([i j],:)=[c s;-s c]*y([i j],:);

      end %j
    end %i
  end %t
    
  R=Q*W;

  return