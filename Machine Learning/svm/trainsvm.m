function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%
% CSCI 576 2014 Fall, Homework 3
[data_row, data_col] = size(train_data);
H = diag([ones(1,data_col) zeros(1, data_row+1)]);
f = [zeros(data_col+1, 1);C*ones(data_row,1)];
A = -1*[repmat(train_label,1,data_col).*train_data train_label diag(ones(1,data_row))];
bb = -1*ones(data_row,1);
opts = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
lb = [-inf(data_col+1,1); zeros(data_row,1)];
ub = [;inf(data_col+data_row+1,1)];
x0 = [;zeros(data_col+data_row+1,1)];
t = cputime;
x = quadprog(H,f,A,bb,[],[],lb,ub,x0,opts);
e = cputime - t;
w = x(1:data_col,1);
b = x(data_col+1,1);