function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = rows(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
    z = theta'*X'(:,i);
    J = J - y(i,1)*log(sigmoid(z)) - (1 - y(i,1))*log(1 - sigmoid(z));
    grad = grad + (sigmoid(z) - y(i,1))*X'(:,i);
endfor

for i = 2:n
    J = J + lambda/2*theta(i, 1)**2;
endfor

for i = 2:rows(theta)
    grad(i,1) = grad(i,1) + lambda*theta(i,1);
endfor

J = J/m;
grad = grad./m

% =============================================================

end
