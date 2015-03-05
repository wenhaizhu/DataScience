function accu = testsvm(test_data, test_label, w, b)
% Test linear SVM 
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  test_label: M*1 vector, each row as a label
%  w: feature vector 
%  b: bias term
%
% Output:
%  accu: test accuracy (between [0, 1])
%
% CSCI 576 2014 Fall, Homework 3

[data_row, data_col] = size(test_data);
count = 0;
for i=1:data_row
    y = test_data(i,:)*w+b;
    if y*test_label(i,1) > 0
        count = count + 1;
    end
end

accu = count/data_row;