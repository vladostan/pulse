function W=KDICA(x,varargin)
%KERNEL DENSITY ICA
% ICA Model -  x=As, where s have independent components, A is called mixing matrix;
% KDICA :    prewhitened version of semiparametric profile MLE methods for ICA models 
%            with fast kernel density estimation;
%              Returns an unmixing matrix W such that s=W*x 's components are as close 
%                 as possible to mutually independence;
%	       It first whitens the data and then minimizes
%                 the given contrast function over orthogonal matrices.
% x          : observed mixtures, dim * samplesize;
% W0         : unmixing matric initialization
%
% Version 1.0, CopyRight: Aiyou Chen
% Jan. 2004 in UC Village at Albany, CA
%
% Acknowledgement: 
% Some optimization subroutines such as Gold_Search(), Bracket_IN() were written 
%   by Francis Bach for "Kernel Independent Component Analysis" (JMLR02)

% Contact: aychen@research.bell-labs.com

% default values
verbose=0;

% data dimension
[m,N]=size(x);
ncomp=m;
   
% first centers and scales data
if (verbose), fprintf('\nStart KDICA \nwhiten ...\n'); end
  xc=x-repmat(mean(x,2),1,N);  % centers data
  covmat=(xc*xc')/N;

  sqcovmat=sqrtm(covmat);
  invsqcovmat=inv(sqcovmat);
  xc=invsqcovmat*xc;           % scales data
%  if (verbose), fprintf('unmixing ...\ndone\n\n'); end
  
% initial uses JADE
if (length(varargin)<=1),
    if 1,
        A0=jade(xc); 
    else
        A0=randn(m);
    end
    [U,S,V]=svd(A0);
    W0=V*U';
end
   
% optional values
if (rem(length(varargin),2)==1)
   error('Optional parameters should always go by pairs');
else
   for i=1:2:(length(varargin)-1)
      switch varargin{i}
      case 'W0'
         W0= varargin{i+1};
         [U,S,V]=svd(W0*sqcovmat);
         W0=U*V';
      end
   end
end

[J,W]=localopt(xc,W0);

W=W*invsqcovmat;

% normalize each row of W
%for i=1:m
%     W(i,:)=W(i,:)/norm(W(i,:)); 
%end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grad=gradcklica(X,W);
% calculate gradient of the negative log profile likelihood function

grad=W;
for i=1:size(grad,1),
    grad(i,:)=gradordkde(X,W(i,:)*X)';
end
grad=grad-inv(W');
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z=cklica(s);

% Negative log profile likelihood function using Laplacian Kernel Density Estimates

z=0; m=size(s,1);
for i=1:m,
    z=z-ordkde(sort(s(i,:)));
end
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Jopt,Wopt] = localopt(x,W)
% LOCAL-OPT  -  Conjugate gradient method for finding a minima in the
%               Stiefel manifold of orthogonal matrices Wopt such 
%               that Wopt*x are the independent sources.
% x          - data (whitened mixtures);
% W          - orthogonal matric, starting point of the search;
% tolW    :    precision in amari distance in est. demixing matrix;
% tolJ    :    precision in objective function;
% maxit   :    maximum number of iterations;
% verbose :    1 if verbose required.

% initializations

m=size(W,1);
N=size(x,2);
tolW=1e-2;
tolJ=0.01; % linear form in 1/(N*m)
maxit=10*m;

verbose=0;
tmin=1;
iter = 0;
errW = tolW*2;
errJ = tolJ*2;
fret = cklica(W*x);
totalneval=1;
transgradJ=0;
k=0;

% starting minimization
while (((errW > tolW)|(errJ > tolJ)) & (iter < maxit)  )
   Jold=fret;
   iter=iter+1;
   if (verbose), fprintf('iter %d, J=%.5e',iter,fret); end
   
   % calculate derivative
   gradJ=gradcklica(x,W);
   iterneval=1;
   normgradJ=sqrt(.5*trace(gradJ'*gradJ));
   
   dirSearch=gradJ-W*gradJ'*W;
   normdirSearch=sqrt(.5*trace(dirSearch'*dirSearch));
   
   % bracketing the minimum along the geodesic and performs golden search
   [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(W,dirSearch,x,0,tmin,Jold);
   iterneval=iterneval+neval;
   goldTol=max(abs([tolW/normdirSearch, mean([ ax, bx, cx])/10]));
   [tmin, Jmin,neval] = golden_search(W,dirSearch,x,ax, bx, cx,goldTol,20);
   iterneval=iterneval+neval;
   oldtransgradJ=transgradJ;
   Wnew=stiefel_geod(W',dirSearch',tmin);  
   oldnormgradJ=sqrt(.5*trace(gradJ'*gradJ));
   
   errW=amari(W,Wnew);
   errJ=Jold/Jmin-1;
   totalneval=totalneval+iterneval;
   if (verbose)
      fprintf(', dJ= %.1e',errJ);
      fprintf(',errW=%.1e,dW= %.3f, neval=%d\n',errW,tmin*normdirSearch,iterneval);
   end
   
   if (errJ>0) 
      W=Wnew;
      fret=Jmin;
   else
      errJ=0; errW=0;
   end
   
end

Jopt= fret;
Wopt=W;

fprintf('iteration times=%d, contrast function evaluation times=%d\n',iter,totalneval);
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wt,Ht]=stiefel_geod(W,H,t)

% STIEFEL_GEOD - parameterizes a geodesic along a Stiefel manifold

% W  - origin of the geodesic
% H  - tangent vector
% Ht - tangent vector at "arrival"
% Alan Edelman, Tomas Arias, Steven Smith (1999)

if nargin <3, t=1; end
A=W'*H; A=(A-A')/2;
MN=expm(t*A);
Wt=W*MN;
if nargout > 1, Ht=H*MN; end
Wt=Wt';
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xmin,fmin,neval] = golden_search(W,dirT,x,ax,bx,cx,tol,maxiter)

% GOLDEN_SEARCH - Minimize contrast function along a geodesic of the Stiefel
%                 manifold using golden section search.
% W              - initial value
% x              - mixed components
% dirT           - direction of the geodesic
% ax,bx,cx       - three abcissas such that the minimum is bracketed between ax and cx,
%                  as given by bracket_mini.m
% tol            - relative accuracy of the search
% maxit          - maximum number of iterations
% neval          - outputs the number of evaluation of the contrast function

neval=0;
% golden ratios
C = (3-sqrt(5))/2;
R = 1-C;

x0 = ax;
x3 = cx;

% gets the smaller segment
if (abs(cx-bx) > abs(bx-ax)),
   x1 = bx;
   x2 = bx + C*(cx-bx);
else
   x2 = bx;
   x1 = bx - C*(bx-ax);
end

Wtemp=stiefel_geod(W',dirT',x1);
f1=cklica(Wtemp*x);
Wtemp=stiefel_geod(W',dirT',x2);
f2=cklica(Wtemp*x);

neval=neval+2;
k = 1;

% starts iterations
while ((abs(x3-x0) > tol) & (k<maxiter)), 
   if f2 < f1,
      x0 = x1;
      x1 = x2;
      x2 = R*x1 + C*x3;   
      f1 = f2;
      Wtemp=stiefel_geod(W',dirT',x2);
      f2=cklica(Wtemp*x);
      neval=neval+1;
   else
      x3 = x2;
      x2 = x1;
      x1 = R*x2 + C*x0;  
      f2 = f1;
      Wtemp=stiefel_geod(W',dirT',x1);
      f1=cklica(Wtemp*x);
      neval=neval+1;
   end
   k = k+1;
end

% best of the two possible
if f1 < f2,
   xmin = x1;
   fmin = f1;
else
   xmin = x2;
   fmin = f2;
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(W,dirT,x,ax, bx,fax)

% BRACKET_MIN - Brackets a minimum by searching in both directions along a geodesic in
%               the Stiefel manifold
% W              - initial value
% x              - mixed components
% dirT           - direction of the geodesic
% ax,bx          - Initial guesses
% tol            - relative accuracy of the search
% maxit          - maximum number of iterations
% neval          - outputs the number of evaluation of the contrast function

neval=0;
GOLD=1.618034;
TINY=1e-10;
GLIMIT=100;
Wtemp=stiefel_geod(W',dirT',bx);
fbx=cklica(Wtemp*x);

neval=neval+1;

if (fbx > fax)   
   temp=ax;
   ax=bx;
   bx=temp;
   temp=fax;
   fax=fbx;
   fbx=temp;
end

cx=(bx)+GOLD*(bx-ax);
Wtemp=stiefel_geod(W',dirT',cx);
fcx=cklica(Wtemp*x);

neval=neval+1;

while (fbx > fcx) 
   
   r=(bx-ax)*(fbx-fcx);
   q=(bx-cx)*(fbx-fax);
   u=(bx)-((bx-cx)*q-(bx-ax)*r)/(2.0*max([abs(q-r),TINY])*sign(q-r));
   ulim=(bx)+GLIMIT*(cx-bx);
   if ((bx-u)*(u-cx) > 0.0)
      Wtemp=stiefel_geod(W',dirT',u);
      fux=cklica(Wtemp*x);
      
      neval=neval+1;
      
      if (fux < fcx) 
         ax=(bx);
         bx=u;
         fax=(fbx);
         fbx=fux;
         return;
      else 
         if (fux > fbx) 
            cx=u;
            fcx=fux;
            return;
         end
      end
      
      u=(cx)+GOLD*(cx-bx);
      Wtemp=stiefel_geod(W',dirT',u);
      fux=cklica(Wtemp*x);
      neval=neval+1;
      
   else 
      if ((cx-u)*(u-ulim) > 0.0) 
         Wtemp=stiefel_geod(W',dirT',u);
         fux=cklica(Wtemp*x);
         neval=neval+1;
         
         if (fux < fcx) 
            bx=cx;
            cx=u;
            u=cx+GOLD*(cx-bx);
            
            fbx=fcx;
            fcx=fux;
            Wtemp=stiefel_geod(W',dirT',u);
            fux=cklica(Wtemp*x);
            neval=neval+1;
         end
      else 
         if ((u-ulim)*(ulim-cx) >= 0.0) 
            
            u=ulim;
            Wtemp=stiefel_geod(W',dirT',u);
            fux=cklica(Wtemp*x);
            neval=neval+1;
            
         else 
            u=(cx)+GOLD*(cx-bx);
            Wtemp=stiefel_geod(W',dirT',u);
            fux=cklica(Wtemp*x);
            neval=neval+1;
            
         end
      end
   end
   
   ax=bx;
   bx=cx;
   cx=u;
   
   fax=fbx;
   fbx=fcx;
   fcx=fux;
   
end
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: 
%   Fast Kernel Density Independent Component Analysis, 
%      Volume 3889 of Lecture Notes in Computer Sciences, Springer, 2006;
%      6th International Conference on ICA & BSS, 2006, Charleston, SC, USA.
%
%   Chapter 4 of Efficient Independent Component Analysis (2004), Ph.D Thesis, 
%      by Aiyou Chen, Advisor: Peter J. Bickel, Department of Statistics, UC Berkeley


% Revision history
% Aug 27, 2006: 
%  1. update gradient calculation (replace previous finite difference approximation)
%  2. remove multiple restartings (most time Cardoso's JADE initials well)
%  3. test ordkde.c (matlab 6.5 can do cumsum iteration as well,but c is 1/3 faster)
