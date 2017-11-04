function Ergebnis=sir(Perf)

% function Ergebnis=sir(Perf)
%
% Berechnet den Performance-Index fuer ICA. Kleine Werte zeigen
% gute Performance an
  
  [N,P]=size(Perf);
  
  Summe=0;
  term1=0;
  term2=0;
  
  for i=1:N,
    for j=1:N,
      
      term1=term1+(abs(Perf(i,j))/max(abs(Perf(i,:))));
      
      term2=term2+(abs(Perf(j,i))/max(abs(Perf(:,i))));
      
    end
    
    Summe=Summe+term1+term2-2;
    
    term1=0;
    term2=0;
    
  end
  
  Ergebnis=Summe/(N*N);
   