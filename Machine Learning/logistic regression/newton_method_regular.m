function [cross_entropy, w, b] = newton_method_regular(lambda, step, data, w, b)

[row, col] = size(data);
features = data(:,1:col-1);
class = data(:,col);
cross_entropy = [];
for s=1:step
    
    inner_b1 = 0;
    inner_b2 = 0;
    for j = 1:row
        x = features(j,:);
        y = data(j,col);
        wx = b + sum(w.*x);
        delta = 1 / (1 + exp(-wx));
        delta1 = 1- delta;
        if delta < exp(-16)
            delta = exp(-16);
        end
        if delta1 < exp(-16)
            delta1 = exp(-16);
        end
        inner_b1 = inner_b1 + (delta - y);
        inner_b2 = inner_b2 + delta * delta1;
    end
    
    
    delta = 1 ./ (1 + exp(-features * transpose(w) - b));
    trans_x = transpose(features);
    inner = pinv(trans_x * delta * transpose(1 - delta) * features + 2*lambda) *( trans_x * (delta - class) + 2*lambda * transpose(w));
    w = w - transpose(inner);
    
    b = b - inner_b1/inner_b2;
    
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