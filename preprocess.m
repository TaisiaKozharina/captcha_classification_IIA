function [out] = preprocess(img, bg, verbose)
   
% Grayscale & smooth
I0 = im2double(rgb2gray(img));
bgGray = im2double(im2gray(bg));
I1 = I0 - bgGray;
if verbose
    figure;
    imshow(I0); title('Grayscaled image - background');
end

% Enhance contrast and blur
se = strel('disk', 30);

top = imtophat(I1, se);      % enhances bright strokes
bot = imbothat(I1, se);      % enhances dark valleys

I2 = I1 + top - bot;
I2 = imgaussfilt(I2, 2);
I2 = imadjust(mat2gray(I2), stretchlim(I2,[0.001 0.995]));

if verbose
    figure;
    imshow(I2), title('Enhanced + blurred');
end


% binarize
level = graythresh(I2);
I3 = ~imbinarize(I2);
if verbose
    figure;
    imshow(I3); title('Binarized');
end

% Erosion
I4 = imerode(I3,strel("disk",2));
if verbose
figure;
imshow(I4); title("Eroded");
end


% Filter small objects
filter = bwareafilt(I4,[0 80]);
I5 = I4 - filter;
if verbose
    figure;
    imshow(I5); title("bwareafilt");
end


% Closing
I6 = imclose(I5,strel("disk",3));
if verbose
    figure;
    imshow(I6); title("Closing with disk");
end

% Filter small objects again
filter = bwareafilt(imbinarize(I6),[0 90]);
I7 = I6 - filter;
if verbose
    figure;
    imshow(I7); title("bwareafilt");
end

% Rotate
[y,x] = find(I7);
coords = [x y];

coeff = pca(coords);
angle = atan2(coeff(2,1), coeff(1,1));
angleDeg = rad2deg(angle);

I_rot = imbinarize(imrotate(I7, angleDeg, 'bilinear', 'crop'));
if verbose
    figure;
    imshow(I_rot); title("Rotated");
end

out = I_rot;

end