function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%Linear Regularized Cost Function -----------------------------------------
J_sqerr = (1/(2*m))*sum(((X*theta)-y).^2);        %X is [ones(m, 1) X] so (12x2) * theta(2x1) = (12x1)
J_reg = (lambda/(2*m))*sum(theta(2:size(theta),:).^2);  %ommitting the theta0 term

J = J_sqerr + J_reg;

%Gradient -----------------------------------------------------------------

% grad0 = (1/m)*sum(sum(((X*theta(2:size(theta),:)) - y).*X(:,1)));
% grad_rest = (1/m)*sum(sum(((X*theta - y).*X(:,2:size(X,2)))) + ((lambda/m)*(theta(2:size(theta),:))));
% grad = ((1/m)*sum(((X*theta) - y).*X)) + ((lambda/m)*theta(2:size(theta),:));   %excluding theta0 for the regularization term
% grad = [grad0; grad_rest];

theta0 = theta;
theta0(1) = 0;          %setting first term of theta to zero for regularization term. 

grad = ((1/m)*sum(((X*theta)-y).*X)) + ((lambda/m)*theta0');






% =========================================================================

grad = grad(:);

end
