function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% a = [];
% for i = 1:3
%     if i == 1
%         a(i) = X;
%     else 
%         a(i) = sigmoid(a(i-1)*Theta1);
    
%Forward Propogation-------------------------------------------------------
a1 = [ones(m,1) X];                     %(5000x401)

z2 = a1*Theta1';                        %a1*theta1. Trans to match dims.(5000x401)*(401x25) = (5000x25)
a2 = [ones(size(z2,1),1) sigmoid(z2)];  %(5000x26)

z3 = a2*Theta2';                        %(5000x26)*(26x10) = (5000x10)
a3 = sigmoid(z3);

htheta = a3;

% y_new = zeros(num_labels, size(y,1));   %(10x5000)
% for i = 1:size(y,1)                     %the important bit. setting y_new to be vectors containing 0 and 1 where y label points. 
%     y_new(y(i),i) = 1;
% end

%mapping vector y to matrix
eye_matrix = eye(num_labels);             %from discussion thread
y_matrix = eye_matrix(y,:);

%Unregularized Cost Function-----------------------------------------------
%help from discussion thread. USE .* WITH LOG AND DOUBLE SUM
J = (1/m)*(sum(sum((-y_matrix.*log(htheta)) - ((1-y_matrix).*log(1-htheta)))));

% delta = sum(-y_new.*log(htheta) - (1-y_new).*log(1-htheta));
% J = (1/m)*sum(delta);



%Regularization Terms------------------------------------------------------

theta1_r = Theta1(:,2:end);
theta2_r = Theta2(:,2:end);

reg1_temp = sum(sum(theta1_r.^2));
reg2_temp = sum(sum(theta2_r.^2));

Reg = (lambda/(2*m))*(reg1_temp + reg2_temp);


%Regularized Cost Function-------------------------------------------------


J = J+Reg;

%Back Propogation ---------------------------------------------------------

d3 = a3 - y_matrix;

%using theta2_r to exclude the first column
%the hidden layer bias has no effect on the input layer
d2 = (d3*theta2_r).*(sigmoidGradient(z2));      %(5000x10)*(10x25) = (5000x25)

Delta1 = d2'*a1;      %(25x5000)*(5000*401) = (25x401)
Delta2 = d3'*a2;

%Gradient without Regularization-------------------------------------------
Theta1_grad = (1/m)*(Delta1);
Theta2_grad = (1/m)*(Delta2);

%Gradient with Regularization----------------------------------------------
Theta1(:,1) = 0; 
Theta1 = (lambda/m)*(Theta1);

Theta2(:,1) = 0;
Theta2 = (lambda/m)*(Theta2);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% for t=1:m
%     
%     
% end








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
