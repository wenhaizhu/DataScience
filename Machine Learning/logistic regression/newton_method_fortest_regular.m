function ce_sum = newton_method_fortest_regular(lambda, data, w, b)

[row, col] = size(data);
features = data(:,1:col-1);
class = data(:,col);

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