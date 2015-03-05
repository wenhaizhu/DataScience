function error_fun = gradient_descent_regular_fortest(lambda, yita, data, b, w)
[row, col] = size(data);
features = data(:,1:col-1);


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
ce_sum = -ce_sum + lambda * sqrt(sum(w.*w));

error_fun = ce_sum;