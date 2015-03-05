function [gini, cross]=decision_tree(train_data, train_label, test_data, test_label, valid_data, valid_label)

gini=[];
[train_row, train_col] = size(train_data);
[valid_row, valid_col] = size(valid_data);
[test_row, test_col] = size(test_data);

for i=1:10
    temp=[];
    tree=ClassificationTree.fit(train_data,train_label,'SplitCriterion','gdi','MinLeaf',i,'Prune','off');
    y = predict(tree,train_data);
    accu1 = length(y(y==train_label)) / train_row;
    y = predict(tree,valid_data);
    accu2 = length(y(y==valid_label)) / valid_row;
    y = predict(tree,test_data);
    accu3 = length(y(y==test_label)) / test_row;
    temp = [temp, [accu1, accu2, accu3]];
    gini = [gini;temp];
end


cross=[];
for i=1:10
    temp=[];
    tree=ClassificationTree.fit(train_data,train_label,'SplitCriterion','deviance','MinLeaf',i,'Prune','off');
    y = predict(tree,train_data);
    accu1 = length(y(y==train_label)) / train_row;
    y = predict(tree,valid_data);
    accu2 = length(y(y==valid_label)) / valid_row;
    y = predict(tree,test_data);
    accu3 = length(y(y==test_label)) / test_row;
    temp = [temp, [accu1, accu2, accu3]];
    cross = [cross;temp];
end