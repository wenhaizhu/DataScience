function CSCI567_hw3()
display('loading data ...');
load('./splice_train.mat');
[data_row, data_col] = size(data);
mean_data = mean(data);
std_data = std(data);
for i=1:data_row
    data(i,:) = (data(i,:) - mean_data)./std_data;
end
    
train_data = data;
train_label = label;

load('./splice_test.mat');
[data_row, data_col] = size(data);
for i=1:data_row
    data(i,:) = (data(i,:) - mean_data)./std_data;
end

test_data = data;
test_label = label;
display('data loaded');


display('shuffle the training data');
[data_row, data_col] = size(train_data);
p = randperm(data_row);
tempdata = [];
templabel = [];
for i = 1:size(p,2)
    index = p(1,i);
    tempdata = [tempdata;train_data(index,:)];
    templabel = [templabel;train_label(index,1)];
end

train_data = tempdata;
train_label = templabel;


display('for 4.3------------------');
display('computing the time and accuracy using cross validation');
C = [4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4 4^2 ];
[time accus] = cross_validation(train_data, train_label, C);
j = 1;
for i=-6:2
    str = strcat('for C = 4^', num2str(i), ': average time: %f, average accuracy: %f \n');
    fprintf(str, time(1, j), accus(1,j));
    j = j + 1;
end
        

[w, b] = trainsvm(train_data, train_label, 4^-3);
accu_curr = testsvm(test_data, test_label, w, b);
str = strcat('choose C = 4^-3, the test accuracy is: %f \n');
fprintf(str, accu_curr);

display('for 4.4------------------');
C = [4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4 4^2 ];
[c_row c_col] = size(C);
j=-6;
for i = 1:c_col
    c = C(1,i);
    fprintf('for C = 4^%d :\n',j);
    j=j+1;
    opts = sprintf('-q -t 0 -c %f -v 5',c);
    t = cputime;
    svmtrain(train_label, train_data, opts);
    e = (cputime - t) / 5;
    display(sprintf('average time = %.3fs\n', e));
end

display('for 4.5(a)------------------');
C = [4^-4 4^-3 4^-2 4^-1 4^0 4^1 4^2 4^3 4^4 4^5 4^6 4^7];
D = [1 2 3];
[c_row c_col] = size(C);
[d_row d_col] = size(D);
k = -4;
for i = 1:c_col
    for j = 1:d_col
        c = C(1,i);
        d = D(1,j);
        fprintf('for C = 4^%d and d = %d :\n',k,j);
        opts = sprintf('-q -t 1 -c %f -d %d -v 5',c,d);
        t = cputime;
        svmtrain(train_label, train_data, opts);
        e = (cputime - t) / 5;
        display(sprintf('average time = %.3fs\n', e));
    end
    k=k+1;
end

display('for 4.5(b)------------------');
C = [4^-4 4^-3 4^-2 4^-1 4^0 4^1 4^2 4^3 4^4 4^5 4^6 4^7];
gamma = [4^-7 4^-6 4^-5 4^-4 4^-3 4^-2 4^-1];
[c_row c_col] = size(C);
[g_row g_col] = size(gamma);
k = -4;
m = -7;
for i = 1:c_col
    for j = 1:g_col
        c = C(1,i);
        g = gamma(1,j);
        fprintf('for C = 4^%d and g = 4^%d :\n',k,m);
        m = m+1;
        opts = sprintf('-q -t 2 -c %f -g %f -v 5',c,g);
        t = cputime;
        svmtrain(train_label, train_data, opts);
        e = (cputime - t) / 5;
        display(sprintf('average time = %.3fs\n', e));
    end
    k=k+1;
    m=-7;
end

display('I choose RBF kernel with C = 4^1 and gamma = 4^-3: ');
model = svmtrain(train_label, train_data, sprintf('-q -t 2 -c %f -g %f',4^1,4^-3));
[predicted_label, accu, decision] = svmpredict(test_label, test_data, model);