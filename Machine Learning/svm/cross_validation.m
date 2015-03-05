function [time accus] = cross_validation(train_data, train_label, C)
[c_row, c_col] = size(C);
[data_row, data_col] = size(train_data);
part = data_row / 5;
time = [];
accus = [];
for i = 1:c_col
    c = C(1,i);
    t = cputime;
    accu = 0;
    for j = 0:4
        data = [train_data(1:j*part,:); train_data((j+1)*part+1:data_row,:)];
        label = [train_label(1:j*part,1); train_label((j+1)*part+1:data_row,1)];
        datatest = train_data(j*part+1:(j+1)*part,:);
        labeltest = train_label(j*part+1:(j+1)*part,1);
        [w b] = trainsvm(data,label, c);
        accu = accu + testsvm(datatest,labeltest,w,b);
    end
    e = cputime;
    meantime = (e - t)/ 5;
    meanaccu = accu / 5;
    time = [time meantime];
    accus = [accus meanaccu];
end
    