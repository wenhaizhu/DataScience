function decision_boundary(train_data, train_label, k)


test_data=[];
for r=1:100
    for c=1:100
        test_data = [test_data;[r/100, c/100]];
    end
end


[train_row, train_col] = size(train_data);
total_train = [train_data, train_label];
[test_row, test_col] = size(test_data);
result_label = [];
for i=1:test_row
    newrow = test_data(i,:);
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
    for m=1:k
        if total_train(ind(m),3) == 1
            c1 = c1 + 1;
        elseif total_train(ind(m),3) == -1
            c2 = c2 + 1;
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
        flag = -1;
    end
    result_label = [result_label;[flag]];
end


test_total = [test_data, result_label];
index1 = find(test_total(:,3)==1);
index2 = find(test_total(:,3)==-1);
scatter(test_total(index1,1),test_total(index1,2));

hold on
c='red';
scatter(test_total(index2,1),test_total(index2,2),c);
