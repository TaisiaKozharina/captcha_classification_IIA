function svm_model = myclassifier(imageDir, labelFile, bg)
% Trains an ECOC SVM digit classifier and evaluates on a hold-out set

% Inputs:
% imageDir - directory of Training data
% labelFile - path to labels txt file
% bg - mean background image (provided to avoid recalculations)


% Validate inputs
if ~isfolder(imageDir)
    error('Image directory does not exist.');
end
if ~isfile(labelFile)
    error('Label file does not exist.');
end

N = 1200;
labels = readmatrix(labelFile);

features = {};
digitLabels = [];

hogParams = {'CellSize',[8 8]};

for i = 1:N
    fprintf("training on sample %d\n", i);

    sampleID = labels(i,1);
    imgName = sprintf('captcha_%04d.png', sampleID);
    imgPath = fullfile(imageDir, imgName);

    I = imread(imgPath);
    I_processed = preprocess(I, bg);
    digits = splitBoundingBox(I_processed);

    if sum(cellfun(@(d) nnz(d) > 0, digits)) ~= nonzeros(labels(i,2:5))
        fprintf("Skipping sample %d (label mismatch)\n", i);
        continue;
    end

    first_zero = labels(i,2) == 0;
    if first_zero
        labels_adjusted = [labels(i,3:5) 0];
    else
        labels_adjusted = labels(i,2:5);
    end

    for k = 1:4
        if nnz(digits{k}) == 0
            continue;
        end

        feat = extractHOGFeatures(digits{k}, hogParams{:});
        features{end+1,1} = feat;
        digitLabels(end+1,1) = labels_adjusted(k);
    end
end

X = cell2mat(features);
Y = digitLabels(:);

fprintf('\nTotal digit samples: %d\n', numel(Y));

%% ---- Train / Test Split ----
rng(42);
cv = cvpartition(Y, 'HoldOut', 0.4);

Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));
Xtest  = X(test(cv), :);
Ytest  = Y(test(cv));

%% ---- Train SVM (ECOC) ----
t = templateSVM( ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'BoxConstraint',1);

svm_model = fitcecoc(Xtrain, Ytrain, 'Learners', t);

%% ---- Evaluate ----
evaluate_classifier(svm_model, Xtest, Ytest);

% Save model
save('svm_model.mat', 'svm_model', 'hogParams', 'bg');
fprintf('Model saved.\n');
end
