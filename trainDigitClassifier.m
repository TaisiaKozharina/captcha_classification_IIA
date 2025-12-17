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
        fprintf("Skipping sample $d because number of labels dont match", i);
        continue;
    end

    % TODO: If first label will be zero - do something, because that zero
    % will be LAST (not first) in segmentation (digits array)
    first_zero = labels(i, 1)==0;
    fprintf("Initialize first_zero as: %d\n",first_zero);


    for k = 1:4

        if first_zero && k==4 
            continue;
        end
        
        if first_zero
            %fprintf("First zero, so taking k+2: %d\n",k+2);
            lbl = labels(i, k+2);
        else
            %fprintf("First not zero, so taking k+1: %d\n",k+1);
            lbl = labels(i, k+1);
        end

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