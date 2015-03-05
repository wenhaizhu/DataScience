function [train_accu, valid_accu, test_accu]=logistic_regression(train_data, train_label, test_data, test_label, valid_data, valid_label)


train_b = mnrfit(train_data, train_label,'model','nominal');
train_prob = mnrval(train_b,train_data);
[train_row, train_col] = size(train_prob);
train_right = 0;
for i=1:train_row
    [x, y] = max(train_prob(i,:));
    if y == train_label(i)
        train_right = train_right + 1;
    end
end
train_accu = train_right / train_row;


valid_prob = mnrval(train_b,valid_data);
[valid_row, valid_col] = size(valid_prob);
valid_right = 0;
for i=1:valid_row
    [x, y] = max(valid_prob(i,:));
    if y == valid_label(i)
        valid_right = valid_right + 1;
    end
end
valid_accu = valid_right / valid_row;


test_prob = mnrval(train_b,test_data);
[test_row, test_col] = size(test_prob);
test_right = 0;
for i=1:test_row
    [x, y] = max(test_prob(i,:));
    if y == test_label(i)
        test_right = test_right + 1;
    end
end
test_accu = test_right / test_row;