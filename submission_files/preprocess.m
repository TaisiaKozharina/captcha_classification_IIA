function [out] = preprocess(img, bg)

    % Grayscale and get rid of background
    I0 = rgb2gray(img);   
    bgGray = im2double(im2gray(bg));
    I0 = im2double(I0) - bgGray;
    % figure;
    % imshow(I0); title("Grayscale minus background");
    

    % Enhance dark regions (the digits)
    se = strel('disk', 30);
    bot = imbothat(I0, se); % enhances dark valleys
    I1 = I0  - bot;
    I1 = mat2gray(I1);
    % figure;
    % imshow(I1); title("Dark regions enhanced");

    % Smmoth before binarizing
    I2 = medfilt2(I1, [5 5]); % Median filter the enhanced image before binarization (tune window size)
    I2 = imgaussfilt(I2,1); % smooth
    % figure;
    % imshow(I2); title("Median filtering + Gauss filtering");

    %Binarize
    I3 = ~imbinarize(I2, 0.55);
    % figure;
    % imshow(I3); title("Binarized");

    %Erode
    I4 = imerode(I3,strel("disk",3));
    % figure;
    % imshow(I4); title("Eroded");

    %Dilate
    I5 = imdilate(I4,strel("disk",3));
    % figure;
    % imshow(I5); title("Dilated");

    %Filter small objects
    small_objects = bwareafilt(I5,[0 1550]);
    I6 = I5 - small_objects;
    % figure;
    % imshow(I6); title("Dilated");

    %Rotate

    [y, x] = find(I6); %foreground coordinates

    % Safety check
    if numel(x) > 50
        
        p = polyfit(x, y, 1); % Fit a line: y = a*x + b
        angle = atan(p(1)) * 180 / pi;% Slope -> Estimated skew angle (in degrees)
        disp(angle)
        bw_rot = imrotate(I6, angle, 'bilinear', 'crop');
    else
        bw_rot = I6;
    end

    out = bw_rot;

end
