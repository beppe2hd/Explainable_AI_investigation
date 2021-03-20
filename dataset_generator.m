% Generation of training and validation dataset for Explainable AI tests
% The script generates and save in two (training and validation) CSV files a set of features vector made up by
% gaussian random variables plus the class contribution on specific features

file_train_name = 'mytrain.csv';
file_val_name = 'myval.csv';

num_of_element_train = 10000; % number of element for training
num_of_element_val = 500; % number of element for validation

feature_0_contribution_class_0 = 50; %features amplitude on class 0
feature_0_contribution_class_1 = -50; %features amplitude on class 0

vector_size = 290; % features vector size
noise_factor = 10; % noise amplitude

class_0_features_start = 1; % start index for features 0
class_0_features_end = 45; % end index for features 0
class_0_features_size = class_0_features_end - class_0_features_start +1; %width of features influencing class 0

class_1_features_start = 156; % start index for features 1
class_1_features_end = 156+44; % end index for features 1
class_1_features_size = class_1_features_end - class_1_features_start +1; %width of features influencing class 1

% Train matrix 

train_mat = noise_factor * rand(num_of_element_train,vector_size+1); % matrix generation

class_0 = feature_0_contribution_class_0 * ones(num_of_element_train/2,class_0_features_size); %features 0 contribuion 
class_1 = feature_0_contribution_class_1 * ones(num_of_element_train/2,class_1_features_size); %features 1 contribuion 

gt_class_0 = zeros(num_of_element_train/2,1); % ground truth for features 0
gt_class_1 = ones(num_of_element_train/2,1); % ground truth for features 1


% The features contribution are added to the main matrix

train_mat(1:num_of_element_train/2,class_0_features_start:class_0_features_end) = ...
    train_mat(1:num_of_element_train/2,class_0_features_start:class_0_features_end) + class_0;

train_mat(num_of_element_train/2+1:end,class_1_features_start:class_1_features_end) = ...
    train_mat(num_of_element_train/2+1:end,class_1_features_start:class_1_features_end) + class_1;

train_mat(1:num_of_element_train/2,end) = gt_class_0;

train_mat(num_of_element_train/2+1:end,end) = gt_class_1;

csvwrite(file_train_name, train_mat)


% Validation matrix

val_mat = noise_factor * rand(num_of_element_val,vector_size+1); % matrix generatione

class_0 = feature_0_contribution_class_0 * ones(num_of_element_val/2,class_0_features_size); %features 0 contribuion 
class_1 = feature_0_contribution_class_1 * ones(num_of_element_val/2,class_1_features_size); %features 1 contribuion 

gt_class_0 = zeros(num_of_element_val/2,1); % ground truth for features 0
gt_class_1 = ones(num_of_element_val/2,1); % ground truth for features 1

% The features contribution are added to the main matrix

val_mat(1:num_of_element_val/2,class_0_features_start:class_0_features_end) = ...
    val_mat(1:num_of_element_val/2,class_0_features_start:class_0_features_end)+class_0;

val_mat(num_of_element_val/2+1:end,class_1_features_start:class_1_features_end) = ...
    val_mat(num_of_element_val/2+1:end,class_1_features_start:class_1_features_end)+class_1;

val_mat(1:num_of_element_val/2,end) = gt_class_0;

val_mat(num_of_element_val/2+1:end,end) = gt_class_1;

csvwrite(file_val_name, val_mat)


% The haed will be printed and manually added to the csv

for i = 1:vector_size
    fprintf('"A%d",',i)
end
fprintf('"Class"\n')