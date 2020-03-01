function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%




%initialize variable to store distance(norm) of point from centroid. 
%Compare against each centroid. Thus, size will be [total # Centroids, 1] 
temp_dist = zeros(size(centroids,1),1);

%for all points, find distance to each centroid and save in temp_dist
%find which one is minimum. that one will be the closesnt centroid to point
%save the index to idx(that point)

for j=1:size(X,1)
    for k=1:K
        temp_dist(k,:) = sqrt(sum((X(j,:)-centroids(k,:)).^2));
    end
%had a variable here to store value, Matlab suggested to use ~ since value
%is unused. 
    [~,idx(j)] = min(temp_dist);
end







% =============================================================

end

