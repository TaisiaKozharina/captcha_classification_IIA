function boxes = splitBoundingBox(binImg, widthThreshold)

    binImg = logical(binImg);

    % Get coordinates of ALL non-zero pixels
    [rows, cols] = find(binImg);

    if isempty(rows)
        boxes = [];
        return;
    end

    % Global bounding box covering ALL foreground pixels
    xMin = min(cols);
    xMax = max(cols);
    yMin = min(rows);
    yMax = max(rows);

    w0 = xMax - xMin + 1;
    h0 = yMax - yMin + 1;


    % Decide number of parts
    if w0 > widthThreshold
        disp("4 digits based on threshold");
        nParts = 4;
    else
        disp("3 digits based on threshold");
        nParts = 3;
    end

    partW = w0 / nParts;

    % Create bounding boxes
    boxes = zeros(nParts, 4);
    for i = 1:nParts
        boxes(i,:) = [ ...
            xMin + (i-1)*partW, ...
            yMin, ...
            partW, ...
            h0 ];
    end
    
end

