function [spamdata, hamdata, totaldata] = process_spam_data(spampath, hampath, dic)

train_spam_features = spam_feature_vector(spampath, dic);
[train_spam_features_row, train_spam_features_col] = size(train_spam_features);
train_ham_features = spam_feature_vector(hampath, dic);
[train_ham_features_row, train_ham_features_col] = size(train_ham_features);
spamham_train_features = [train_spam_features;train_ham_features];

train_spam_class = ones(train_spam_features_row, 1);
train_ham_class = zeros(train_ham_features_row, 1);
spamham_train_class = [train_spam_class; train_ham_class];
spamham_train = [spamham_train_features spamham_train_class];

spamdata = [train_spam_features train_spam_class];
hamdata = [train_ham_features train_ham_class];
totaldata = spamham_train;