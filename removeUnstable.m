function data_stable = removeUnstable(data_interp)

% To retain the most stable features find the maximum distance (rounded to 
% the nearest pixel) traveled by each point between consecutive frames and 
% discard points with a distance exceeding the mode of the distribution.

maxVal = zeros(size(data_interp,1), 1);

% May be rounded up to nearest pixel
for i = 1:size(data_interp,1)
    dif = abs(diff((data_interp(i,:))));
    maxVal(i) = max(dif);
end

avg = mean(maxVal);

% Remove those feature points whose maxVal > avg
for i = 1:size(data_interp,1)
    if maxVal(i) > avg
        data_interp(i,:) = 0;
    end
end

data_interp(all(data_interp == 0, 2),:)= [];
size(data_interp,1);
data_stable = data_interp;
end
