function eigVecs = applyPCA(data_filtered, num)

%num - number of principal components to compute

[Zpca, U, mu, eigVecs] = PCA(data_filtered, num);

end

