function crops = splitBoundingBox(binImg, widthThreshold)

    binImg = logical(binImg);

    % Get coordinates of ALL non-zero pixels
    [rows, cols] = find(binImg);

    if isempty(rows)
        crops = {};
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

    [H, W] = size(binImg);

    % Create cropped images
    crops = cell(1, nParts);

    for i = 1:nParts
        % Horizontal limits
        x1 = round(xMin + (i-1)*partW);
        x2 = round(xMin + i*partW - 1);

        % Clamp to image bounds
        x1 = max(x1, 1);
        x2 = min(x2, W);

        % Vertical limits (global bbox)
        y1 = yMin;
        y2 = yMax;

        crops{i} = binImg(y1:y2, x1:x2);

end

