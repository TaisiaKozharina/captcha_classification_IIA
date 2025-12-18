% digits('G:\My Drive\Μεταπτυχιακο\Semester 1\IAML\Mini-Project\newImg\captcha_0001.png')
img = 'Data/Train/captcha_1020.png';
% figure;
% imshow(img);
% bg = imread('Data/Train/mean_image.png');
mdl = trainDigitClassifier('Data/Train', 'Data/Train/labels.txt', bg);
digits = predictDigitClassifier(img);