function evaluate_classifier(svm_model, Xtest, Ytest)

% Inputs:
% svm_model - saved SVM model (if training run previously - saved as "svm_model.mat"
% Xtest - test set image array
% Ytest - true labels for test set (subset of labels.txt)

Ypred = predict(svm_model, Xtest);

accuracy = mean(Ypred == Ytest);
fprintf('Test accuracy: %.2f%%\n', accuracy * 100);

classes = unique(Ytest);
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

T = table(classes, precision, recall, f1, ...
    'VariableNames', {'Digit','Precision','Recall','F1'});
disp(T);

% Confusion matrix (skip first class)
allClasses = unique([Ytest; Ypred]);
cm = confusionmat(Ytest, Ypred, 'Order', allClasses);
cm2 = cm(2:end, 2:end);
classes2 = allClasses(2:end);

figure;
confusionchart(cm2, classes2);
title('Digit Classification Confusion Matrix');
end
