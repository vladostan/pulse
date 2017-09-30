function data_interp = cubicSplineInterp(V, data)

% Increase sampling rate from 30 to 250 Hz

% a = x(:,1);
% b = y(:,1);
% aa = linspace(min(a),max(a),222*250/30);
% bb = csapi(a,b,aa);
% plot(a,b,'o',aa,bb)

frameRate = V.FrameRate;
numFr = V.NumberOfFrames;

data_interp = zeros(size(data,1), round(numFr*250/frameRate));
% y_interp = zeros(size(y,1), round(numFr*250/30));

xx = linspace(1,numFr,round(numFr*250/frameRate));

for i = 1:size(data,1)
    data_interp(i,:) = spline(1:numFr,data(i,:),xx);
%     yinterp(i,:) = spline(1:numFr,y(i,:),xx);
%     plot(x(i,:),y(i,:),'o',xinterp,yinterp)
end

% figure, scatter(xx, data_interp(1,:))
% figure, scatter(xx, yinterp(1,:))
end




