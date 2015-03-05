function [class, means, funcvalue] = kmeans2(input, k, iteration)
[input_row input_col] = size(input);
means = [];
rnum = ceil(unifrnd(1,input_row,1,k));
for i=1:k
    means = [means; input(rnum(i),:)];
end
class = zeros(input_row, 1);
distance = zeros(input_row, k);
funcvalue = zeros(iteration,1);
for it=1:iteration
    cur_funcvalue = 0;
    for i = 1:k
        means_matrix = repmat(means(i,:), input_row, 1);
        tmp_matrix = (input - means_matrix).*(input - means_matrix);
        distance(:,i) = sum(tmp_matrix, 2);
    end
    [nouse, class] = min(distance, [], 2);
    funcvalue(it,1) = sum(min(distance'));
    
    data = [input class];
    means = [];
    for i=1:k
        cur_mean = mean(data(data(:,end)==i,:));
        means = [means; cur_mean(:,1:input_col)];
    end
    
end