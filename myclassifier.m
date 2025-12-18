function preds = myclassifier(im, model, bg)

I = preprocess(im, bg, false);
digits = segment(I);

preds = [];

for i=1:4
    if nnz(digits{k}) == 0
        preds(i) = 0;
    else
        preds(i) = model.predict(digits(i));
    end
end
