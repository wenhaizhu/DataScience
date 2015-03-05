function [error_fun, w] = gradient_descent(yita, data, step, b)
[row, col] = size(data);
features = data(:,1:col-1);
w = zeros(1, col-1);
cross_entropy = [];
for s = 1:step
    inner_w = zeros(1, col-1);
    inner_b = 0;
    for i = 1:row
        x = features(i,:);
        y = data(i,col);
        wx = b + sum(w.*x);
        delta = 1 / (1 + exp(-wx));
        if delta < exp(-16)
            delta = exp(-16);
        end
        inner_w = inner_w + (delta - y) * x;
        inner_b = inner_b + (delta - y);
    end
    w = w - yita * inner_w;
    b = b - yita * inner_b;

    ce_sum = 0;
    for i = 1:row
        x = features(i,:);
        y = data(i,col);
        wx = b + sum(w.*x);
        delta = 1 / (1 + exp(-wx));
        delta1 = 1 - delta;
        if delta < exp(-16)
            delta = exp(-16);
        end
        if delta1 < exp(-16)
            delta1 = exp(-16);
        end
        ce_sum = ce_sum + y * log(delta) + (1-y) * log(delta1);
    end
    ce_sum = -ce_sum;
    cross_entropy = [cross_entropy ce_sum];
end
error_fun = cross_entropy;
