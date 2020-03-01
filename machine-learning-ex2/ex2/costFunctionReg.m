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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_temp1 = (lambda/(2*m)).*(theta(2:size(theta),:).^2);
theta_temp2 = [theta(1) ; theta_temp1];
% theta_temp3 = sum(theta_temp2);

delta = sum(-y.*log(sigmoid(sum(theta'.*X,2))) - (1-y).*log(1-sigmoid(sum(theta'.*X,2))),2);
J = (1/m)*sum(delta) + sum(theta_temp2(2:size(theta_temp2)));


temp0 = sigmoid(sum(theta'.*X,2))-y;
grad_0 = (1/m)*sum(temp0.*X(:,1));

temp_rest = sigmoid(sum(theta'.*X,2))-y;
grad_rest1 = (1/m)*sum(temp_rest.*X(:, 2:size(X,2))); 
grad_rest2 = ((lambda/m).*theta(2:size(theta),:));
grad_rest = grad_rest1' + grad_rest2;

grad = [grad_0 ;grad_rest];



% temporary = sigmoid(sum(theta'.*X,2))-y;
% grad = (1/m)*sum(temporary.*X);
% =============================================================

end
