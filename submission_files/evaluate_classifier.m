function predictedDigits = evaluate_classifier(imgPath)
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


% ---- Predict each digit ----
for k = 1:numel(digits)


    % Skip empty segments (same logic as training)
    if nnz(digits{k}) == 0
        continue;
    end


    digitImg = digits{k};


    % IMPORTANT: keep preprocessing identical to training
    % If you resize during training later, do it here too
    % digitImg = imresize(digitImg, [40 40]);


    % Extract HOG features
    feat = extractHOGFeatures(digitImg, hogParams{:});


    % Predict digit
    label = predict(svm_model, feat);


    predictedDigits(end+1) = label;
end


% ---- Handle leading zero case (optional) ----
% If your captcha may have a leading zero that was shifted during training,
% you may want to pad to 4 digits:
if numel(predictedDigits) == 3
    predictedDigits = [0 predictedDigits];
end


fprintf('Predicted digits: ');
disp(predictedDigits);


end



