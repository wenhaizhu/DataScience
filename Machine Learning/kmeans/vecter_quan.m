function vecter_quan(pic, k)
f=imread(pic);

[x,y,z]=size(f);
iminput = reshape(f, x*y, z);

[class, means] = kmeans(double(iminput), k);
for i = 1:x*y
    nthmeans = class(i,1);
    iminput(i,:) = means(nthmeans,:);
end
f = reshape(iminput, x, y, z);
figure;
imshow(f);
        