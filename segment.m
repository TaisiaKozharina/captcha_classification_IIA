function digits = segment(img)

proj = sum(img, 1);    % Sum of white pixels per column
proj = movmean(proj, 10);  % Smooth to suppress noise
figure; plot(proj);

th = mean(proj) + 0.5*std(proj);
mask = proj > th;
mask = imclose(mask, ones(1,20));  % bridge small gaps
figure; plot(mask);


cols = bwlabel(mask);
numDigits = max(cols)

centers   = zeros(1,numDigits);
widths    = zeros(1,numDigits);
strengths = zeros(1,numDigits);

for i = 1:numDigits
    idx = find(cols == i);
    centers(i)   = mean(idx);
    widths(i)    = numel(idx);
    strengths(i) = sum(proj(idx));
end

if numDigits > 4
    [~, order] = sort(strengths, 'descend');
    keep = sort(order(1:4));

    newMask = false(size(mask));
    for i = keep
        newMask(cols == i) = true;
    end

    mask = newMask;
end


if numDigits == 2
    [~, i] = max(widths);
    idx = find(cols == i);

    mid = round(mean(idx));

    mask(idx(idx <= mid)) = true;
    mask(idx(idx >  mid)) = true;

    % Force a gap at the split point
    mask(mid-2:mid+2) = false;
end

if numDigits == 1
    idx = find(cols == 1);

    w = numel(idx);
    cut1 = idx(round(w/3));
    cut2 = idx(round(2*w/3));

    mask(cut1-2:cut1+2) = false;
    mask(cut2-2:cut2+2) = false;
end

%mask = imclose(mask, ones(1,30));  % bridge small gaps (20)
%figure; plot(mask);
cols = bwlabel(mask);
numDigits = max(cols);



boxes = zeros(numDigits,4);
for i = 1:numDigits
    idx = find(cols==i);

    x1 = max(min(idx)-40, 1);
    w  = min(max(idx)-min(idx)+40, size(img,2)-x1);
    
    % display borders

    fprintf("Borders: min(idx) = %d, max(idx) = %d\n", min(idx),max(idx))
    fprintf("Borders: x_start -> %d, x_width -> %d\n", x1, w)

    %  boxes(i,:) = [min(idx)-45,1, max(idx)-min(idx)+50, size(img,1)];
    boxes(i,:) = [x1, 1, w, size(img,1)];
end

digits = cell(1,4);

for i=1:1:numDigits
    if boxes(i,3)>0
        crop = imcrop(img, boxes(i,:));
        crop = imresize(crop, [40 40]);
        digits{i} = crop;
    else
        digits{i} = zeros(40, 40);
    end
end