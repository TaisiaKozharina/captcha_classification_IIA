function meanImg = meanImage(folder)
    numImages = 1200;
    if nargin < 1 || isempty(folder)
        folder = pwd;
    end
    if nargin < 2 || isempty(numImages)
        numImages = 1200;
    end

    % collect files with common image extensions
    exts = {'*.png'};
    files = [];
    for k = 1:numel(exts)
        files = [files; dir(fullfile(folder, exts{k}))]; %#ok<AGROW>
    end

    if isempty(files)
        error('No image files found in folder: %s', folder);
    end

    % if requested more than available, use available count
    totalFiles = numel(files);
    if numImages > totalFiles
        warning('Requested %d images but only %d found. Using %d images.', ...
            numImages, totalFiles, totalFiles);
        numImages = totalFiles;
    end

    % read first image to get size and channels
    firstPath = fullfile(folder, files(1).name);
    img = im2double(imread(firstPath));
    acc = zeros(size(img));

    % accumulate images
    for i = 1:numImages
        pathi = fullfile(folder, files(i).name);
        Ii = im2double(imread(pathi));
        if ~isequal(size(Ii), size(acc))
            error('Image size mismatch: %s does not match first image size.', files(i).name);
        end
        acc = acc + Ii;
    end

    % compute mean
    meanImg = acc / numImages;
end