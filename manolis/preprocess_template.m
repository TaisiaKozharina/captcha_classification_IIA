function skel= preprocess_template(img)

temp_5_skel = bwskel(img);

[rows, columns] = find(temp_5_skel);
row1 = min(rows);
row2 = max(rows);
col1 = min(columns);
col2 = max(columns);

skel = temp_5_skel(row1:row2, col1:col2); % Crop image.

% figure; imshow(skel); title("Skeleton of template");
end