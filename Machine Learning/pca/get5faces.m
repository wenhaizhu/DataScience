function get5faces(filepath)

img = load(filepath);
faceimage = img.image;
[firow, ficol] = size(faceimage);
vector = [];
for i=1:ficol
    vector = [vector; reshape(faceimage{i}',1, 2500)];
end
eigenvecs = pca_fun(vector, 5);

[row, col] = size(eigenvecs);
for i=1:col
    figure();
    final = reshape(eigenvecs(:,i)',50,50);
    imshow(final',[]);
end