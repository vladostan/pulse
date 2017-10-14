function B =  shibbs(X,m)
%
%  Developpment version 
%================================================================
% Blind separation of real signals with SHIBBS.  Version 1.5 Dec. 1997.
%
% Usage: 
%   * If X is an nxT data matrix (n sensors, T samples) then
%     B=shibbsR(X) is a nxn separating matrix such that S=B*X is an nxT
%     matrix of estimated source signals.
%   * If B=shibbsR(X,m), then B has size mxn so that only m sources are
%     extracted.  This is done by restricting the operation of shibbsR
%     to the m first principal components. 
%   * Also, the rows of B are ordered such that the columns of pinv(B)
%     are in order of decreasing norm; this has the effect that the
%     `most energetically significant' components appear first in the
%     rows of S=B*X.
%
% Copyright : Jean-Francois Cardoso.  cardoso@sig.enst.fr

[n,T]	= size(X);

verbose	= 0 ;          %% Set to 0 for quiet operation
seuil	= 0.01/sqrt(T) ;  %% A statistically significant threshold for joint diag.

% Finding the number of sources
if nargin==1, m=n ; end; 	% Number of sources defaults to # of sensors
if m>n ,    fprintf('shibbs -> Do not ask more sources than sensors here!!!\n'), return,end
if verbose, fprintf('shibbs -> Looking for %d sources\n',m); end ;



% Mean removal
%=============
if verbose, fprintf('shibbs -> Removing the mean value\n'); end 
MeanX	= mean(X,2) ;
X	= X - MeanX(:,ones(1,T));
clear   MeanX ;


%%% whitening & projection onto signal subspace
%   ===========================================
if verbose, fprintf('shibbs -> Whitening the data\n'); end
 [U,D] 		= eig((X*X')/T)	; 
 [puiss,k]	= sort(diag(D))	;
 rangeW		= n-m+1:n			; % indices to the m  most significant directions
 scales		= sqrt(puiss(rangeW))		; % scales
 B  		= diag(1./scales)  * U(1:n,k(rangeW))'	;	% whitener

 X		= B*X;  

 
%%% Estimation of the cumulant matrices.
%   ====================================


nbcm 		= m  ; 	%% number of cumulant matrices
CM 		= zeros(m,m*nbcm);  % Storage for cumulant matrices

%% Mainly Temps
Rk 		= zeros(m);  	%% 
R 		= eye(m);  	%% 
Xk              = zeros(m,T) ; 
xk              = zeros(1,T) ; 
Uns             = ones(m,1)  ; % for The Trick

OneMoreStep     = 1 ;  %

while OneMoreStep ,
  
  if verbose, fprintf('shibbs -> Estimating cumulant matrices\n'); end
  %% Computing a `small number' of cumulant matrices.
  %% -------------------------------------------------
  Range = 1:m ; % will index the columns of CM where to store the cum. mats.
  %%  fprintf('Cumulant matrices: ')
  for k = 1:m
  %% if verbose, fprintf('shibbs -> Cum. Mat. #%d\n',k); end
    xk          = X(k,:) ; 
    Xk          = X .* xk(Uns,:) ; % Oooch
    Rk          = (Xk*Xk')/T - R ;
    Rk(k,k)     = Rk(k,k) - 2 ;
    CM(:,Range) = Rk ;  
    Range       = Range + m ;
  end;
  %%  fprintf('\n')
  
  %% Joint diagonalization of the cumulant matrices
  %% ----------------------------------------------

  %% Init
  V     = eye(m) ; % la rotation initiale
  nbrs  = 1;       % Number of rotations in this sweep. Also used for control
  sweep	= 0;       % Number of sweeps
  updates = 0 ;
  g	= zeros(2,nbcm);
  gg	= zeros(2,2);
  G	= zeros(2,2);
  c	= 0 ;
  s 	= 0 ;
  ton	= 0 ;
  toff	= 0 ;
  theta	= 0 ;
  
  %% Joint diagonalization proper
  if verbose, fprintf('shibbs -> Contrast optimization by joint diagonalization\n'); end
  
  while nbrs, nbrs=0;   % Will start again unless there is at least one update

    sweep=sweep+1; if verbose, fprintf('shibbs -> Sweep #%d',sweep); end

    for p=1:m-1,
      for q=p+1:m,
	
	Ip = p:m:m*nbcm ;
	Iq = q:m:m*nbcm ;
	
	%%% computation of Givens angle
	g	= [ CM(p,Ip)-CM(q,Iq) ; CM(p,Iq)+CM(q,Ip) ];
	gg	= g*g';
	ton 	= gg(1,1)-gg(2,2); 
	toff 	= gg(1,2)+gg(2,1);
 	theta	= 0.5*atan2( toff , ton+sqrt(ton*ton+toff*toff) );
	
	%%% Givens update
	if abs(theta) > seuil,	nbrs = nbrs + 1 ;
	  c	= cos(theta); 
	  s	= sin(theta);
	  G	= [ c -s ; s c ] ;
	  
	  pair 		= [p;q] ;
	  V(:,pair) 	= V(:,pair)*G ;
	  CM(pair,:)	= G' * CM(pair,:) ;
	  CM(:,[Ip Iq]) = [ c*CM(:,Ip)+s*CM(:,Iq) -s*CM(:,Ip)+c*CM(:,Iq) ] ;
	  
	  %% fprintf('shibbs -> %3d %3d %12.8f\n',p,q,s);

	end%%of the if
      end%%of the loop on q
    end%%of the loop on p
    
    if verbose, fprintf(' completed in %d rotations.\n',nbrs); end
    updates = updates + nbrs ;
    
  end%%of the while loop

  RotSize = norm(V-eye(m),'fro') ;
  if verbose, fprintf('shibbs -> Amount of rotation in this pass: %14.6f \n',RotSize); end

  X       = V'*X ;
  B       = V'*B  ;

%  if updates == 0 , OneMoreStep = 0;  end
  if RotSize < (m*seuil) , OneMoreStep = 0;  end
  
end ; % ijd


%%% We permut its rows to get the most energetic components first.
%%% Here the **signals** are normalized to unit variance.  Therefore,
%%% the sort is according to the norm of the columns of A = pinv(B)

if verbose, fprintf('shibbs -> Sorting the components\n',updates); end
A		= pinv(B) ;
[vars,keys]	= sort(sum(A.*A)) ;
B		= B(keys,:);
B		= B(m:-1:1,:) ; % Is this smart ?

% Signs are fixed by forcing the first column of B to have
% non-negative entries.
if verbose, fprintf('shibbs -> Fixing the signs\n',updates); end
b	= B(:,1) ;
signs	= sign(sign(b)+0.1) ; % just a trick to deal with sign=0
B	= diag(signs)*B ;


return ;
