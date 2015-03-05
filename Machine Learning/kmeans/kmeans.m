function [class, means] = kmeans(input, k)
[input_row input_col] = size(input);
means = [];
rnum = ceil(unifrnd(1,input_row,1,k));
for i=1:k
    means = [means; input(rnum(i),:)];
end
former_means = zeros(k,input_col);
class = zeros(input_row, 1);
distance = zeros(input_row, k);
while 1
    if isequal(former_means, means) 
        break;
    else 
        former_means = means;
    end
    for i = 1:k
        means_matrix = repmat(means(i,:), input_row, 1);
        tmp_matrix = (input - means_matrix).*(input - means_matrix);
        distance(:,i) = sum(tmp_matrix, 2);
    end
    [nouse, class] = min(distance, [], 2);
    
    data = [input class];
    means = [];
    for i=1:k
        cur_mean = mean(data(data(:,end)==i,:));
        means = [means; cur_mean(:,1:input_col)];
    end
end

            