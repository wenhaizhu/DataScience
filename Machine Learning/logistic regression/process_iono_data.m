function data = process_iono_data(filename)
fid = fopen(filename);
iono_train_raw = textscan(fid,'%f%f%f%f64%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s','delimiter',',');
fclose(fid);
[iono_train_row, iono_train_col] = size(iono_train_raw);
iono_train = [];
for i = 1:iono_train_col-1
    temp = iono_train_raw(i);
    iono_train = [iono_train temp{1}];
end
iono_train_class_raw = iono_train_raw{iono_train_col};
iono_train_class = [];
[row, col] = size(iono_train_class_raw);
for i = 1:row
    if (iono_train_class_raw{i} == 'b')
        iono_train_class = [iono_train_class;1];
    elseif (iono_train_class_raw{i} == 'g')
        iono_train_class = [iono_train_class;0];
    end
end
iono_train = [iono_train iono_train_class];
data = iono_train;