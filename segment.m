function digits = segment(img)

proj = sum(img, 1);    % Sum of white pixels per column
proj = movmean(proj, 10);  % Smooth to suppress noise
% figure; plot(proj);

% adaptive thresholding
th = mean(proj) + 0.5*std(proj);
mask = proj > th;
mask = imclose(mask, ones(1,20));  % bridge small gaps (20)
cols = bwlabel(mask);
numDigits = max(cols)

boxes = zeros(numDigits,4);
for i = 1:numDigits
    idx = find(cols==i);
    boxes(i,:) = [min(idx)-40,1, max(idx)-min(idx)+50, size(img,1)];
end

digits = cell(1,4);

for i=1:3
    if boxes(i,3)>0
        crop = imcrop(img, boxes(i,:));
        crop = imresize(crop, [40 40]);
        digits{i} = crop;
    else
        digits{i} = zeros(40, 40);
    end
end

% digits{4} will be empty in case of 3 digits

end