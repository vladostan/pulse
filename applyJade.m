function eigVecs = applyJade(data_filtered, num)

%num - number of principal components to compute

[eigVecs,S] = jade(data_filtered, num);

end