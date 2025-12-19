% figure;
% imshow(img);
% bg = imread('Data/Train/mean_image.png');
bg = meanImage('Data/Train');
% imshow(meanImage);
% tic;
% mdl = trainDigitClassifier('Data/Train', 'Data/Train/labels.txt', bg);
md1 = myclassifier('Data/Train', 'Data/Train/labels.txt', bg);
% elapsed1 = toc;

% tic
% for i = 1:1200
    % img = sprintf('Data/Train/captcha_%04d.png', i);
    digits = predictDigitClassifier(img);
% end
% elapsed2 = toc;

% fprintf('Elapsed time for training: %.4f s\n', elapsed1);
% fprintf('Elapsed time for test set: %.4f s\n', elapsed2);
% totalElapsed = elapsed1 + elapsed2;
% fprintf('Total elapsed time: %.4f s\n', totalElapsed);


