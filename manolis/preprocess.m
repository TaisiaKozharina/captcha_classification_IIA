function [out] = preprocess(img, bg)
   
% imds = imageDatastore(datasetDir, 'FileExtensions', {'.png'}, 'IncludeSubfolders', true);
%

subtracted = rgb2gray(img);

% subtracted = subtracted - uint8(bg);
bgGray = im2double(im2gray(bg));


sub = im2double(subtracted);
subtracted = sub - bgGray;

se = strel('disk', 30); % tune size!
bot = imbothat(subtracted, se); % enhances dark valleys

I_enh = subtracted - bot;
I_enh = mat2gray(I_enh);


% Median filter the enhanced image before binarization (tune window size)
I_med = medfilt2(I_enh, [5 5]);

% Then smooth and binarize
BW = ~imbinarize(imgaussfilt(I_med,1), 0.55);


eroded = imerode(BW,strel("disk",3));
closed = imdilate(eroded,strel("disk",3));


BW2 = bwareafilt(closed,[0 1550]);
cleaned = closed - BW2;

bw = cleaned;


%% --------- DESKEW USING POLYFIT ---------


% Get foreground pixel coordinates
[y, x] = find(bw);


% Safety check
if numel(x) > 50
    % Fit a line: y = a*x + b
    p = polyfit(x, y, 1);


    % Slope -> angle (in degrees)
    angle = atan(p(1)) * 180 / pi;


    % fprintf('Estimated skew angle: %.2f degrees\n', angle);


    % Rotate both binary and grayscale images
    bw_rot = imrotate(bw, angle, 'bilinear', 'crop');
else
    bw_rot = bw;
end


out = bw_rot;

end