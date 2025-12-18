function svm_model = trainDigitClassifier(imageDir, labelFile, bg)
% Trains an ECOC SVM digit classifier (3,4,5)

N = 100; %size(labels,1)
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
    I_processed = preprocess(I, bg, false);
    digits = segment(I_processed);

    if sum(cellfun(@(d) nnz(d) > 0, digits)) ~= nonzeros(labels(i,2:5))
        fprintf("\nSkipping sample %d because number of labels dont match\n", i);
        continue;
    end


    % TODO: If first label will be zero - do something, because that zero
    % will be LAST (not first) in segmentation (digits array)
    first_zero = labels(i, 2)==0;
    fprintf("\nInitialize first_zero as: %d\n",first_zero);

    figure;
    tiledlayout(1,4, 'Padding','compact', 'TileSpacing','compact');

    if first_zero
        % Adjust labels if the first label is zero
        labels_adjusted = [labels(i, 3:5) 0];
    else
        labels_adjusted = labels(i, 2:5);
    end
    
    for p = 1:4
        nexttile;
        imshow(digits{p});
        title(sprintf('Digit %d (True label: %d)', p, labels_adjusted(p)));
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
Y = digitLabels;
disp(Y);


% ---- Train SVM (ECOC) ----
t = templateSVM( ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'BoxConstraint',1);

svm_model = fitcecoc(X, Y, 'Learners', t);

disp('\nTraining complete.');
end