function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CSCI 576 2014 Fall, Homework 1

[train_row, train_col] = size(train_data);

train_label_num = [];
train_label_num = hist(train_label, unique(train_label));
train_label_prob = train_label_num / train_row;

total_train = [];
total_train = [train_data, train_label];
prob_matrix = [];
for i=1:train_col
    for j=0:1
        temp = [];
        for k=1:4
            value = find(total_train(:,i)==j&total_train(:,train_col+1)==k);
            num = length(value) / train_label_num(1,k);
            if num==0
                num = 0.1;
            end
            temp = [temp,[num]];
        end
        prob_matrix = [prob_matrix;temp];
    end
end

train_right = 0;
train_wrong = 0;
    
for i=1:train_row
    max_prob = 0;
    flag = 0;
    for c=1:4
        prob = 1;
        for j=1:train_col
            if total_train(i,j)==0
                prob = prob * prob_matrix(2*j-1,c);
            else 
                prob = prob * prob_matrix(2*j,c);
            end
        end
        prob = prob * train_label_prob(c);
        if prob > max_prob
            max_prob = prob;
            flag = c;
        end
    end
    if flag == total_train(i,train_col+1)
        train_right = train_right + 1;
    else
        train_wrong = train_wrong + 1;
    end
end
    
train_accu = train_right / train_row;


    
new_right = 0;
new_wrong = 0;
[new_row, new_col] = size(new_data); 
total_new = [];
total_new = [new_data, new_label];
for i=1:new_row
    max_prob = 0;
    flag = 0;
    for c=1:4
        prob = 1;
        for j=1:new_col
            if total_new(i,j)==0
                prob = prob * prob_matrix(2*j-1,c);
            else 
                prob = prob * prob_matrix(2*j,c);
            end
        end
        prob = prob * train_label_prob(c);
        if prob > max_prob
            max_prob = prob;
            flag = c;
        end
    end
    if flag == total_new(i,new_col+1)
        new_right = new_right + 1;
    else
        new_wrong = new_wrong + 1;
    end
end
    
new_accu = new_right / new_row;

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
     