function eigenvecs = pca_fun(X, d)

% Implementation of PCA
% input:
%   X - N*D data matrix, each row as a data sample
%   d - target dimensionality, d <= D
% output:
%   eigenvecs: D*d matrix
%
% usage:
%   eigenvecs = pca_fun(X, d);
%   projection = X*eigenvecs;
%
% CSCI 576 2014 Fall, Homework 5

xmean = mean(X, 1);
X = X - repmat(xmean, [size(X, 1) 1]);
cov = (1 / size(X, 1)) * (X' * X);
[eigenvecs, eigenvalue] = eig(cov);
[eigenvalue, ind] = sort(diag(eigenvalue), 'descend');
eigenvecs = eigenvecs(:,ind(1:d));