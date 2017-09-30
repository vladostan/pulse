function eigVecs = applyKICA(data_filtered, num)

%num - number of principal components to compute

[Zpca, U, mu, eigVecs] = kICA(data_filtered, num);

end

