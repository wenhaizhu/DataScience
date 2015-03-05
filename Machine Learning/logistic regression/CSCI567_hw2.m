function temp = CSCI567_hw2()

disp('The program will run about 10 - 15 mins. ');
disp('Processing ionosphere data');
iono_train_path = './hw2_data/ionosphere/ionosphere_train.dat';
iono_test_path = './hw2_data/ionosphere/ionosphere_test.dat';
iono_train = process_iono_data(iono_train_path);
iono_test = process_iono_data(iono_test_path);

disp('Processing email dictionary - vocab.dat');
dic_path = './hw2_data/spam/vocab.dat';
dic_fid = fopen(dic_path);
dic = textscan(dic_fid,'%s','Delimiter','\n');
dic = transpose(dic{1});
[dic_row, dic_col] = size(dic);
fclose(dic_fid);

disp('Processing spam & ham data');
train_spam_path = './hw2_data/spam/train/spam/';
train_ham_path = './hw2_data/spam/train/ham/';
test_spam_path = './hw2_data/spam/test/spam/';
test_ham_path = './hw2_data/spam/test/ham/';
spam_train = [];
ham_train = [];
spamham_train = [];
spam_test = [];
ham_test = [];
spamham_test = [];
[spam_train, ham_train, spamham_train] = process_spam_data(train_spam_path, train_ham_path, dic);
[spam_test, ham_test, spamham_test] = process_spam_data(test_spam_path, test_ham_path, dic);

disp('5 (1) Top 3 words--------------------------');
[spamham_train_row, spam_ham_col] = size(spamham_train);
spamham_train_features = spamham_train(:,1:spam_ham_col-1);
spamham_train_features_sum = sum(spamham_train_features);
[train_features_sum_vector,ind] = sort(spamham_train_features_sum,'descend');
fprintf('{');
for i = 1:3
    index = ind(i);
    word = dic(index);
    count = train_features_sum_vector(i);
    fprintf('<%s: %d>',word{1},count(1,1));
    if i == 1 || i == 2
        fprintf(',');
    end
end
fprintf('} \n');
disp('end--------------------------');

disp('5(3) Gradient descent without regularization-----------------');
step = 50;
steparray = [1:50];
b = 0.1;
yita = [0.001, 0.01, 0.05, 0.1, 0.5];
[yita_row, yita_col] = size(yita);
ef_iono_array = [];
ef_spam_array = [];
w_iono_array = [];
w_spam_array = [];
for i = 1:yita_col
    [ef_iono, w_iono] = gradient_descent(yita(1,i), iono_train, step, b);
    [ef_spam, w_spam] = gradient_descent(yita(1,i), spamham_train, step, b);
    ef_iono_array = [ef_iono_array;ef_iono];
    ef_spam_array = [ef_spam_array;ef_spam];
    w_iono_array = [w_iono_array; w_iono];
    w_spam_array = [w_spam_array;w_spam];
end

figure();
plot(steparray, ef_iono_array(1,:), steparray, ef_iono_array(2,:), '--', steparray, ef_iono_array(3,:), '-.', steparray, ef_iono_array(4,:), '-+', steparray, ef_iono_array(5,:), '-*');
title('Ionosphere Training Dataset (without regularization)');
xlabel('steps T');
ylabel('cross entropy function value');
legend('stepsize=0.001', 'stepsize=0.01', 'stepsize=0.05', 'stepsize=0.1', 'stepsize=0.5');

figure();
plot(steparray, ef_spam_array(1,:),  steparray, ef_spam_array(2,:), '--', steparray, ef_spam_array(3,:), '-.', steparray, ef_spam_array(4,:), '-+', steparray, ef_spam_array(5,:), '-*');
title('Emailspam Training Dataset (without regularization)');
xlabel('steps T');
ylabel('cross entropy function value');
legend('stepsize=0.001', 'stepsize=0.01', 'stepsize=0.05', 'stepsize=0.1', 'stepsize=0.5');

l2w_i = [];
l2w_s = [];
for i = 1:yita_col
    w_i = w_iono_array(i, :);
    l2w_i = [l2w_i sqrt(sum(w_i.*w_i))];
    w_s = w_spam_array(i, :);
    l2w_s = [l2w_s sqrt(sum(w_s.*w_s))];
end
disp('for L2 norm(without regularization:');
disp('For Ionosphere:');
disp(l2w_i);
disp('For EmailSpam:');
disp(l2w_s);
disp('end--------------------------');

disp('5 (4) Gradient descent with regularization-----------------');

ef_iono_array_r = [];
ef_spam_array_r = [];
for i = 1:yita_col
    [ef_iono_r, w_iono_r, bt] = gradient_descent_regular(0.1, yita(1,i), iono_train, step, b);
    [ef_spam_r, w_spam_r, bt] = gradient_descent_regular(0.1, yita(1,i), spamham_train, step, b);
    ef_iono_array_r = [ef_iono_array_r;ef_iono_r];
    ef_spam_array_r = [ef_spam_array_r;ef_spam_r];
end

figure();
plot(steparray, ef_iono_array_r(1,:), steparray, ef_iono_array_r(2,:), '--',steparray, ef_iono_array_r(3,:), '-.',steparray, ef_iono_array_r(4,:), '-+',steparray, ef_iono_array_r(5,:),'-*');
title('Ionosphere Training Dataset (with regularization, lambda=0.1)');
xlabel('steps T');
ylabel('cross entropy function value');
legend('stepsize=0.001', 'stepsize=0.01', 'stepsize=0.05', 'stepsize=0.1', 'stepsize=0.5');

figure();
plot(steparray, ef_spam_array_r(1,:), steparray, ef_spam_array_r(2,:), '--',steparray, ef_spam_array_r(3,:), '-.',steparray, ef_spam_array_r(4,:), '-+',steparray, ef_spam_array_r(5,:),'-*');
title('Emailspam Training Dataset (with regularization, lambda=0.1)');
xlabel('steps T');
ylabel('cross entropy function value');
legend('stepsize=0.001', 'stepsize=0.01', 'stepsize=0.05', 'stepsize=0.1', 'stepsize=0.5');

l2w_i_r = [];
l2w_s_r = [];
lambda = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
[lambda_row, lambda_col] = size(lambda);
for i = 1:lambda_col
    [ef_iono_r, w_iono_r, bt] = gradient_descent_regular(lambda(1,i), yita(1,2), iono_train, step, b);
    [ef_spam_r, w_spam_r, bt] = gradient_descent_regular(lambda(1,i), yita(1,2), spamham_train, step, b);
    l2w_i_r = [l2w_i_r sqrt(sum(w_iono_r.*w_iono_r))];
    l2w_s_r = [l2w_s_r sqrt(sum(w_spam_r.*w_spam_r))];
end
disp('for L2 norm(without regularization:');
disp('For Ionosphere:');
disp(l2w_i_r);
disp('For EmailSpam:');
disp(l2w_s_r);

train_iono_lambda_yita = [];
train_spam_lambda_yita = [];
test_iono_lambda_yita = [];
test_spam_lambda_yita = [];
for i = 1:lambda_col
    l2w_i_r = [];
    l2w_s_r = [];
    l2w_i_r2 = [];
    l2w_s_r2 = [];
    for j = 1:yita_col
        [ef_iono_r, w_iono_r, b_iono_r] = gradient_descent_regular(lambda(1,i), yita(1,j), iono_train, step, b);
        [ef_spam_r, w_spam_r, b_spam_r] = gradient_descent_regular(lambda(1,i), yita(1,j), spamham_train, step, b);
        l2w_i_r = [l2w_i_r ef_iono_r(1, size(ef_iono_r,2))];
        l2w_s_r = [l2w_s_r ef_spam_r(1, size(ef_spam_r,2))];
        ef_iono_test = gradient_descent_regular_fortest(lambda(1,i), yita(1,j), iono_test, b_iono_r, w_iono_r);
        ef_spam_test = gradient_descent_regular_fortest(lambda(1,i), yita(1,j), spamham_test, b_spam_r, w_spam_r);
        l2w_i_r2 = [l2w_i_r2 ef_iono_test];
        l2w_s_r2 = [l2w_s_r2 ef_spam_test];
    end
    train_iono_lambda_yita = [train_iono_lambda_yita; l2w_i_r];
    train_spam_lambda_yita = [train_spam_lambda_yita; l2w_s_r];
    test_iono_lambda_yita = [test_iono_lambda_yita; l2w_i_r2];
    test_spam_lambda_yita = [test_spam_lambda_yita; l2w_s_r2];
end

figure();
plot(lambda, train_iono_lambda_yita(:,1), '--', lambda, test_iono_lambda_yita(:,1));
title('Ionosphere Dataset, with stepsize = 0.001');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_iono_lambda_yita(:,2), '--', lambda, test_iono_lambda_yita(:,2));
title('Ionosphere Dataset, with stepsize = 0.01');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_iono_lambda_yita(:,3), '--', lambda, test_iono_lambda_yita(:,3));
title('Ionosphere Dataset, with stepsize = 0.05');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_iono_lambda_yita(:,4), '--', lambda, test_iono_lambda_yita(:,4));
title('Ionosphere Dataset, with stepsize = 0.1');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_iono_lambda_yita(:,5), '--', lambda, test_iono_lambda_yita(:,5));
title('Ionosphere Dataset, with stepsize = 0.5');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

%------

figure();
plot(lambda, train_spam_lambda_yita(:,1), '--', lambda, test_spam_lambda_yita(:,1));
title('Emailspam Dataset, with stepsize = 0.001');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_spam_lambda_yita(:,2), '--', lambda, test_spam_lambda_yita(:,2));
title('Emailspam Dataset, with stepsize = 0.01');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_spam_lambda_yita(:,3), '--', lambda, test_spam_lambda_yita(:,3));
title('Emailspam Dataset, with stepsize = 0.05');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_spam_lambda_yita(:,4), '--', lambda, test_spam_lambda_yita(:,4));
title('Emailspam Dataset, with stepsize = 0.1');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

figure();
plot(lambda, train_spam_lambda_yita(:,5), '--', lambda, test_spam_lambda_yita(:,5));
title('Emailspam Dataset, with stepsize = 0.5');
xlabel('regularization coecient');
ylabel('cross entropy function value');
legend('Train dataset', 'Test dataset');

disp('end--------------------------');


disp('5 (6) Newton Method (without regularization------------)');
[error_fun, iono_w_nm, iono_b_nm] = gradient_descent_regular(0.05, 0.01, iono_train, 5, b);
[error_fun, spam_w_nm, spam_b_nm] = gradient_descent_regular(0.05, 0.01, spamham_train, 5, b);

[ef_iono_nm, w_iono_nm, b_iono_nm] = newton_method(50, iono_train, iono_w_nm, iono_b_nm);
[ef_spam_nm, w_spam_nm, b_spam_nm] = newton_method(50, spamham_train, spam_w_nm, spam_b_nm);

figure();
plot(steparray, ef_iono_nm);
title('Ionosphere Training Dataset (without regularization)');
xlabel('steps T');
ylabel('cross entropy function value');

figure();
plot(steparray, ef_spam_nm);
title('Emailspam Training Dataset (without regularization)');
xlabel('steps T');
ylabel('cross entropy function value');

fprintf('L2 norm for ionosphere data: %f \n', sqrt(sum(w_iono_nm.*w_iono_nm)));
fprintf('L2 norm for emailspam data: %f \n', sqrt(sum(w_spam_nm.*w_spam_nm)));

iono_ce_nm = newton_method_fortest(iono_test, w_iono_nm, b_iono_nm);
spam_ce_nm = newton_method_fortest(spamham_test, w_spam_nm, b_spam_nm);
fprintf('cross entropy value for ionosphere test data: %f \n', iono_ce_nm);
fprintf('cross entropy value for emailspam test data: %f \n', spam_ce_nm);


disp('end--------------------------');

    
disp('5 (7) Newton Method (with regularization)------------');
disp('This part needs a little more time. Please be patient. Thank you!');
ef_iono_nm_array = [];
ef_spam_nm_array = [];
w_iono_nm_array = [];
w_spam_nm_array = [];
for i=1:lambda_col
    [ef_iono_nm, w_iono_nm, b_iono_nm] = newton_method_regular(lambda(1, i), 50, iono_train, iono_w_nm, iono_b_nm);
    [ef_spam_nm, w_spam_nm, b_spam_nm] = newton_method_regular(lambda(1, i), 50, spamham_train, spam_w_nm, spam_b_nm);
    ef_iono_nm_array = [ef_iono_nm_array; ef_iono_nm];
    ef_spam_nm_array = [ef_spam_nm_array; ef_spam_nm];
    w_iono_nm_array = [w_iono_nm_array;w_iono_nm];
    w_spam_nm_array = [w_spam_nm_array; w_spam_nm];
end

figure();
plot(steparray, ef_iono_nm_array(1,:), steparray, ef_iono_nm_array(2,:), '--', steparray, ef_iono_nm_array(3,:), '-.', steparray,  ef_iono_nm_array(4,:), '-x',steparray, ef_iono_nm_array(5,:),':',steparray, ef_iono_nm_array(6,:), '-o',steparray, ef_iono_nm_array(7,:), '-+', steparray, ef_iono_nm_array(8,:), '-*', steparray, ef_iono_nm_array(9,:),'-^',  steparray, ef_iono_nm_array(10,:),'-s',steparray, ef_iono_nm_array(11,:), '-d');
title('Ionosphere Training Dataset (with regularization)');
xlabel('steps T');
ylabel('cross entropy function value');

figure();
plot(steparray, ef_spam_nm_array(1,:), steparray, ef_spam_nm_array(2,:), '--', steparray, ef_spam_nm_array(3,:), '-.', steparray,  ef_spam_nm_array(4,:), '-x',steparray, ef_spam_nm_array(5,:),':',steparray,ef_spam_nm_array(6,:), '-o',steparray, ef_spam_nm_array(7,:), '-+', steparray, ef_spam_nm_array(8,:), '-*', steparray, ef_spam_nm_array(9,:),'-^',  steparray, ef_spam_nm_array(10,:),'-s',steparray, ef_spam_nm_array(11,:), '-d');
title('Emailspam Training Dataset (with regularization)');
xlabel('steps T');
ylabel('cross entropy function value');

disp('L2 norm for ionosphere data: \n');
for i = 1:size(w_iono_nm_array,1)
    fprintf('%f ', sqrt(sum(w_iono_nm_array(i,:).*w_iono_nm_array(i,:))));
end
disp('L2 norm for emailspam data:  \n');
for i = 1:size(w_spam_nm_array,1)
    fprintf('%f ', sqrt(sum(w_spam_nm_array(i,:).*w_spam_nm_array(i,:))));
end

iono_ce_nm_array = [];
spam_ce_nm_array = [];
for i=1:lambda_col
    iono_ce_nm = newton_method_fortest_regular(lambda(1,i),iono_test, w_iono_nm_array(i,:), b_iono_nm);
    spam_ce_nm = newton_method_fortest_regular(lambda(1,i),spamham_test, w_spam_nm_array(i,:), b_spam_nm);
    iono_ce_nm_array = [iono_ce_nm_array iono_ce_nm];
    spam_ce_nm_array = [spam_ce_nm_array spam_ce_nm];
end

disp('cross entropy value for ionosphere test data:');
disp(iono_ce_nm_array);
disp('cross entropy value for emailspam test data:');
disp(spam_ce_nm_array);

disp('end--------------------------');







