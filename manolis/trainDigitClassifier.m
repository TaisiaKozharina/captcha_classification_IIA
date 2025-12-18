function svm_model = trainDigitClassifier(imageDir, labelFile, bg)
% Trains an ECOC SVM digit classifier (3,4,5)

N = 1200; %size(labels,1)
% Read labels
labels = readmatrix(labelFile);

features = {};
digitLabels = []; % zeros(N,4);

hogParams = {'CellSize',[8 8]};

for i = 1:N

    fprintf("training on sample %d",i);

    sampleID = labels(i,1);
    imgName = sprintf('captcha_%04d.png', sampleID);
    imgPath = fullfile(imageDir, imgName);

    I = imread(imgPath);
    I_processed = preprocess(I, bg);
    digits = splitBoundingBox(I_processed);

    if sum(cellfun(@(d) nnz(d) > 0, digits)) ~= nonzeros(labels(i,2:5))
        fprintf("\nSkipping sample %d because number of labels dont match\n", i);
        continue;
    end


    % TODO: If first label will be zero - do something, because that zero
    % will be LAST (not first) in segmentation (digits array)
    first_zero = labels(i, 2)==0;
    % fprintf("\nInitialize first_zero as: %d\n",first_zero);

    % figure;
    % tiledlayout(1,4, 'Padding','compact', 'TileSpacing','compact');

    % Takes the 0 from 1st position to last
    if first_zero
        % Adjust labels if the first label is zero
        labels_adjusted = [labels(i, 3:5) 0];
    else
        labels_adjusted = labels(i, 2:5);
    end
    

    for k = 1:4

        %skip if digits{k} is zero - which will be the last digit
        %if there are only 3 after segmentation
        if nnz(digits{k}) == 0
            continue;
        end


        digitImg = digits{k};

        % Ensure size
        %digitImg = imresize(digitImg, [40 40]);

        % HOG feature
        feat = extractHOGFeatures(digitImg, hogParams{:});

        features{end+1,1} = feat;
        digitLabels(end+1,1) = labels_adjusted(k);

    end

    % fprintf("\nLabels for sample %d: ", i);
    % disp(digitLabels(end-4,end));

end

X = cell2mat(features);
Y = digitLabels(:);   % ensure column vector

fprintf('\nTotal digit samples: %d\n', numel(Y));

%% ---- Train / Test Split ----
rng(42); % for reproducibility
cv = cvpartition(Y, 'HoldOut', 0.4);  % 80% train, 20% test

Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));

Xtest  = X(test(cv), :);
Ytest  = Y(test(cv));

fprintf('Training samples: %d\n', numel(Ytrain));
fprintf('Testing samples: %d\n', numel(Ytest));



% ---- Train SVM (ECOC) ----
t = templateSVM( ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'BoxConstraint',1);

svm_model = fitcecoc(Xtrain, Ytrain, 'Learners', t);


%% ---- Evaluate Model ----
Ypred = predict(svm_model, Xtest);

% Build confusion matrix and exclude the first class (remove both row & column)
allClasses = unique([Ytest; Ypred]);
cm = confusionmat(Ytest, Ypred, 'Order', allClasses);

cm2 = cm(2:end, 2:end);        % skip first class
classes2 = allClasses(2:end);

figure;
confusionchart(cm2, classes2);
title('Digit Classification Confusion Matrix (excluding 1st class)');

testAccuracy = mean(Ypred == Ytest);
fprintf('Overall test accuracy: %.2f%%\n', testAccuracy * 100);

% Predict on the training set separately and compute training accuracy
Ypred_train = predict(svm_model, Xtrain);
trainAccuracy = mean(Ypred_train == Ytrain);
fprintf('Overall train accuracy: %.2f%%\n', trainAccuracy * 100);

classes = unique(Y);
numClasses = numel(classes);

precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1        = zeros(numClasses,1);

for i = 1:numClasses
    c = classes(i);

    TP = sum((Ypred == c) & (Ytest == c));
    FP = sum((Ypred == c) & (Ytest ~= c));
    FN = sum((Ypred ~= c) & (Ytest == c));

    precision(i) = TP / max(TP + FP, eps);
    recall(i)    = TP / max(TP + FN, eps);
    f1(i)        = 2 * precision(i) * recall(i) / ...
                   max(precision(i) + recall(i), eps);
end

%% ---- Display stats ----
T = table(classes, precision, recall, f1, ...
    'VariableNames', {'Digit','Precision','Recall','F1'});

disp(T);

% Save the trained model and relevant params for future prediction
saveFile = fullfile(pwd, 'svm_model.mat');
save(saveFile, 'svm_model', 'hogParams', 'bg');
fprintf('Saved SVM model to %s\n', saveFile);

disp('\nTraining complete.');
end