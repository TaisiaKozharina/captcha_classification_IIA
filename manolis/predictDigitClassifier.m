function predictedDigits = predictDigitClassifier(imgPath)
% Predict digits from a captcha image using trained SVM model

% ---- Load trained model ----
modelFile = fullfile(pwd, 'svm_model.mat');
load(modelFile, 'svm_model', 'hogParams', 'bg');

% ---- Read and preprocess image ----
I = imread(imgPath);
I_processed = preprocess(I, bg);

% ---- Segment digits ----
digits = splitBoundingBox(I_processed);

predictedDigits = [];
validDigits = {};   % store digits that are actually predicted

% ---- Predict each digit ----
for k = 1:numel(digits)

    % Skip empty segments
    if nnz(digits{k}) == 0
        continue;
    end

    digitImg = digits{k};

    % HOG feature extraction
    feat = extractHOGFeatures(digitImg, hogParams{:});

    % Predict digit
    label = predict(svm_model, feat);

    predictedDigits(end+1) = label;
    validDigits{end+1} = digitImg; %#ok<AGROW>
end



%% ---- Visualization ----
figure;
tiledlayout(1, numel(validDigits), ...
    'Padding','compact', ...
    'TileSpacing','compact');

for i = 1:numel(validDigits)
    nexttile;
    imshow(validDigits{i});
    title(sprintf('Predicted: %d', predictedDigits(i)), ...
        'FontSize', 12, 'FontWeight', 'bold');
end

% ---- Handle leading zero case (optional) ----
if numel(predictedDigits) == 3
    predictedDigits = [0 predictedDigits];
end

sgtitle('Predicted Digits');

fprintf('Predicted digits: ');
disp(predictedDigits);

end
