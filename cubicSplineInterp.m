function data_interp = cubicSplineInterp(V, data, samplingRate)

% Increase sampling rate from frameRate to samplingRate Hz

frameRate = V.FrameRate;
numFr = V.NumberOfFrames;

data_interp = zeros(size(data,1), round(numFr*samplingRate/frameRate));

xx = linspace(1,numFr,round(numFr*samplingRate/frameRate));

for i = 1:size(data,1)
    data_interp(i,:) = spline(1:numFr,data(i,:),xx);
%     yinterp(i,:) = spline(1:numFr,y(i,:),xx);
%     plot(x(i,:),y(i,:),'o',xinterp,yinterp)
end

% figure, scatter(xx, data_interp(1,:))
end




