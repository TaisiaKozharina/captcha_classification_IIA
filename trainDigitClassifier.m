function svm_model = trainDigitClassifier(imageDir, labelFile, bg)
% Trains an ECOC SVM digit classifier (3,4,5)

% Read labels
labels = readmatrix(labelFile);

features = {};
digitLabels = [];

hogParams = {'CellSize',[8 8]};

for i = 1:20%size(labels,1)

    fprintf("training on sample %d",i);

    sampleID = labels(i,1);
    imgName = sprintf('captcha_%04d.png', sampleID);
    imgPath = fullfile(imageDir, imgName);

    I = imread(imgPath);
    I_processed = preprocess(I, bg, false);
    digits = segment(I_processed);

    if sum(cellfun(@(d) nnz(d) > 0, digits)) ~= nonzeros(labels(i,2:5))
        fprint("Skipping sample $d because number of labels dont match", i);
        continue;
    end

    for k = 1:4
        lbl = labels(i, k+1);

        if lbl == 0
            continue;   % skip missing digits
        end

        digitImg = digits{k};

        % Ensure size
        %digitImg = imresize(digitImg, [40 40]);

        % HOG feature
        feat = extractHOGFeatures(digitImg, hogParams{:});

        features{end+1,1} = feat;
        digitLabels(end+1,1) = lbl;
    end
end

X = cell2mat(features);
Y = digitLabels;

% ---- Train SVM (ECOC) ----
t = templateSVM( ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'BoxConstraint',1);

svm_model = fitcecoc(X, Y, 'Learners', t);

disp('Training complete.');
end