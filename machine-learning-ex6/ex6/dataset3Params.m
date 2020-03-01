function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list =  [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list =  [0.01 0.03 0.1 0.3 1 3 10 30];

%initializing
error = zeros(length(C_list),length(sigma_list));
result = zeros(length(C_list)*length(sigma_list),3);
row = 1;

%2 loops. C fixed and run all sigma. go to next C.%save output in result. 
%increment row# for next loop and save value in new row. 
for i=1:length(C_list)
    for j = 1:length(sigma_list)
        model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j))); 
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));
        
        result(row,:) = [error(i,j) C_list(i) sigma_list(j)];
        row = row+1;
    end
end

%find the min value and index
[min_error_value min_error_index] = min(result(:,1));
% [result_r,result_c] = find(result==min_error);

%return C and sigma corresponding to that index
C = result(min_error_index,2);
sigma = result(min_error_index,3);

 
% % Sorting prediction_error in ascending order
% sorted_result = sortrows(result, 1);
%   
% % C and sigma corresponding to min(prediction_error)
% C = sorted_result(1,2);
% sigma = sorted_result(1,3);
% =========================================================================

end
