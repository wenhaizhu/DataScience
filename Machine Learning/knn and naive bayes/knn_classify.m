function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CSCI 576 2014 Fall, Homework 1

[train_row, train_col] = size(train_data);
total_train = [train_data, train_label];
train_right = 0;
train_wrong = 0;
for i=1:train_row
    leftrow = total_train(i,:);
    total_distance = [];
    for j=1:train_row
        onerow = total_train(j,:);
        distance = 0;
        for kk=1:train_col
            distance = distance + (onerow(kk) - leftrow(kk)) * (onerow(kk) - leftrow(kk));
        end
        total_distance = [total_distance, [distance]];
    end
    [dis_sort_vector,ind] = sort(total_distance);
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    for m=2:k+1
        if total_train(ind(m),train_col+1) == 1
            c1 = c1 + 1;
        elseif total_train(ind(m),train_col+1) == 2
            c2 = c2 + 1;
        elseif total_train(ind(m),train_col+1) == 3
            c3 = c3 + 1;
        elseif total_train(ind(m),train_col+1) == 4
            c4 = c4 + 1;
        end
    end
    max = 0;
    flag = 0;
    if c1 > max
        max = c1;
        flag = 1;
    end
    if c2 > max
        max = c2;
        flag = 2;
    end
    if c3 > max
        max = c3;
        flag = 3;
    end    
    if c4 > max
        max = c4;
        flag = 4;
    end            
    if flag == total_train(i,train_col+1)
        train_right = train_right + 1;
    else 
        train_wrong = train_wrong + 1;
    end
end

train_accu = train_right / train_row;



[new_row, new_col] = size(new_data);
total_new = [new_data, new_label];
new_right = 0;
new_wrong = 0;
for i=1:new_row
    newrow = total_new(i,:);
    total_distance = [];
    for j=1:train_row
        onerow = total_train(j,:);
        distance = 0;
        for kk=1:train_col
            distance = distance + (onerow(kk) - newrow(kk)) * (onerow(kk) - newrow(kk));
        end
        total_distance = [total_distance, [distance]];
    end
    [dis_sort_vector,ind] = sort(total_distance);
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    for m=1:k
        if total_train(ind(m),train_col+1) == 1
            c1 = c1 + 1;
        elseif total_train(ind(m),train_col+1) == 2
            c2 = c2 + 1;
        elseif total_train(ind(m),train_col+1) == 3
            c3 = c3 + 1;
        elseif total_train(ind(m),train_col+1) == 4
            c4 = c4 + 1;
        end
    end
    max = 0;
    flag = 0;
    if c1 > max
        max = c1;
        flag = 1;
    end
    if c2 >max
        max = c2;
        flag = 2;
    end
    if c3 >max
        max = c3;
        flag = 3;
    end    
    if c4 >max
        max = c4;
        flag = 4;
    end            
    if flag == total_new(i,new_col+1)
        new_right = new_right + 1;
    else 
        new_wrong = new_wrong + 1;
    end
end

new_accu = new_right / new_row;




