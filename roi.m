function [forehead, nose] = roi(V)

% Detect face:
% faceDetector = vision.CascadeObjectDetector;
% bboxes = step(faceDetector, read(V,1));
% IFaces = insertObjectAnnotation(read(V,1), 'rectangle', bboxes, 'Face');
% figure, imshow(IFaces), title('Detected faces');

% Correct rectangle:
% bboxes = [bboxes(1)+bboxes(3)*0.2 bboxes(2)+bboxes(4)*0.05 bboxes(3)*0.52 bboxes(4)-bboxes(4)*0.15]

faceDetector = vision.CascadeObjectDetector;
% for i = 1:numFr
%     face = step(faceDetector, read(V,i));
%     face = [face(1)+face(3)*0.2 face(2)+face(4)*0.05 face(3)*0.52 face(4)-face(4)*0.15];
%     IFaces = insertObjectAnnotation(read(V,i), 'rectangle', face, '');
%     imshow(IFaces);
% end

% Remove eyes:
% Removing the subrectangle spanning 20 to 55 heightwise works well
% for i = 1:numFr
%     face = step(faceDetector, read(V,i));
%     face = [face(1)+face(3)*0.2 face(2)+face(4)*0.05 face(3)*0.52 face(4)-face(4)*0.15];
%     forehead = [face(1) face(2) face(3) face(4)*0.2];
%     nose = [face(1) face(2)+face(4)*0.55 face(3) face(4)*0.35];
%     IFaces = insertObjectAnnotation(read(V,i), 'rectangle', [forehead; nose], '');
%     imshow(IFaces);
% end

face = step(faceDetector, read(V,1));

% Balakrishnan
% face = [face(1)+face(3)*0.25 face(2) face(3)*0.5 face(4)-face(4)*0.10];
% forehead = [face(1) face(2) face(3) face(4)*0.2];
% nose = [face(1) face(2)+face(4)*0.55 face(3) face(4)*0.45];

% New
face = [face(1)+face(3)*0.25 face(2) face(3)*0.5 face(4)-face(4)*0.10];
forehead = [face(1) face(2) face(3) face(4)*0.25];
nose = [face(1) face(2)+face(4)*0.55 face(3) face(4)*0.45];

% IFaces = insertObjectAnnotation(read(V,1), 'rectangle', [forehead; nose], '');
% figure, imshow(IFaces);
end



