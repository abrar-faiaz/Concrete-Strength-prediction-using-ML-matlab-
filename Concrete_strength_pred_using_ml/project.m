%% DATAFRAME DESCRIPTION 

df=readtable("/Users/user/Documents/projects/Concrete Strength prediction using ML(matlab)/concrete_data.csv");
%disp(df(1:5,:))
%summary(df)

%{
Variable Description:
cement              - measured in kg in a m3 mixture
slag                - measured in kg in a m3 mixture
flyash              - measured in kg in a m3 mixture
water               - measured in kg in a m3 mixture
superplasticizer    - measured in kg in a m3 mixture
coarse_agg          - measured in kg in a m3 mixture
fine_agg            - measured in kg in a m3 mixture
age                 - day (1~365)
strength            - measured in kg in a m3 mixture
%}

%% DATA DISTRIBUTION(CHECKING FOR OUTLIERS
%An outlier is a data point that is significantly different from the rest 
% of the data.

%Lower Outlier = Q1 - (1.5 * IQR) Higher Outlier = Q3 + (1.5 * IQR)

%{
figure('Position', [0, 0, 1000, 800]); 
features = df.Properties.VariableNames;

n = 0;
for i = 1:length(features)
    n = n + 1;
    
    % adjusted spacing
    subplot(4, 2, n);
    sub_pos = get(gca, 'Position');
    sub_pos(3) = sub_pos(3) * 0.8;
    set(gca, 'Position', sub_pos);
    
    % boxplot
    boxplot(df.(features{i}));
    title(features{i});
end
sgtitle('Boxplots of Features'); 
%}

% So, There are a few outliers in slag, water, superplasticizer, fine_agg 
% and age.
%% Shape of Dataframe
%{
% size (shape) of the table
[numRows, numCols] = size(df);

% Display the shape of the table
disp(['Number of Rows: ', num2str(numRows)]);
disp(['Number of Columns: ', num2str(numCols)]);
%}

%% Removing vs Replacing Outlier from df.
%% Removing
%{
df2 = rmoutliers(df);
%{
[numRows, numCols] = size(df2);
disp(['Number of Rows: ', num2str(numRows)]);
disp(['Number of Columns: ', num2str(numCols)]);
%}
% So,We lost almost 800 rows if we go with removing
%}

%% Replacing the outlier with linear interpolation
df3= filloutliers(df,'center');
[numRows, numCols] = size(df3);
%{
% Display the shape of the table
disp(['Number of Rows: ', num2str(numRows)]);
disp(['Number of Columns: ', num2str(numCols)]);
1030/258
%}

%% Correlation plot  
%{
corrplot(df)
%}
%Cement, Superplasticizer, Age Have a positive effect on the Concrete's
% Strength, while, Water has a negative effect.

%% Correlation with strength  
%{

corrMatrix = corr(df{:,:});

figure('Position', [100, 100, 1200, 400]);

bar(corrMatrix(:, strcmp(df.Properties.VariableNames, 'Strength')));
xticks(1:numel(df.Properties.VariableNames)); % Setting x-axis tick positions
xticklabels(df.Properties.VariableNames);     % Setting x-axis tick labels
xtickangle(45);                    % Rotating x-axis labels for better visibility

title('Correlation of Strength to other features');
ylabel('Correlation with Strength');
xlabel('Features');

grid on;   
box on;    
axis tight; 
set(gca, 'TickLabelInterpreter', 'none'); 
%}

%% Correlation Heatmap
%{
features = df.Properties.VariableNames;

corrMatrix = corr(df{:, features});

figure('Position', [100, 100, 800, 800]);

% Create a heatmap
heatmap = imagesc(corrMatrix);
colorbar; 

textColors = repmat(corrMatrix(:) > 0, 1, 3); 
textStrings = num2str(corrMatrix(:), '%.2f'); 
textStrings = strtrim(cellstr(textStrings));   
[x, y] = meshgrid(1:numel(features));  
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'FontSize', 12);

set(hStrings, {'Color'}, num2cell(textColors, 2));

xticks(1:numel(features));
yticks(1:numel(features));

xticklabels(features);
yticklabels(features);

xtickangle(45);

title('Correlation Heatmap');

axis equal tight;
box on;
%}

%% Train Test set Split
%{

Y = df.Strength;
X = df(:, ~strcmp(df.Properties.VariableNames, 'Strength'));

rng(7);

testSize = 0.3;

numTest = round(testSize * height(df));

indices = randperm(height(df));

testIndices = indices(1:numTest);
trainIndices = indices(numTest+1:end);

X_train = X(trainIndices, :);
X_test = X(testIndices, :);
Y_train = Y(trainIndices);
Y_test = Y(testIndices);

fprintf('X_train size: %d x %d\n', size(X_train));
fprintf('X_test size: %d x %d\n', size(X_test));
fprintf('Y_train size: %d x %d\n', size(Y_train));
fprintf('Y_test size: %d x %d\n', size(Y_test));
%}

%% 1.Linear regression 
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

mdl = fitlm(X_train_matrix, Y_train_vector);

X_test_matrix = table2array(X_test);

Y_pred = predict(mdl, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_linreg = 1 - (SSE / SST);

mae_linreg = mean(abs(Y_pred - Y_test));

mse_linreg = mean((Y_pred - Y_test).^2);

fprintf('R squared linreg: %.4f\n', r_squared_linreg);
fprintf('MAE linreg: %.4f\n', mae_linreg);
fprintf('MSE linreg: %.4f\n', mse_linreg);

%}

%% 2.Ridge_Regression
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

lambda = 0.01; 
mdl = fitrlinear(X_train_matrix, Y_train_vector, 'Regularization', 'ridge', 'Lambda', lambda);

X_test_matrix = table2array(X_test);

Y_pred_Ridge_Regression = predict(mdl, X_test_matrix);

SSE = sum((Y_test - Y_pred_Ridge_Regression).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_Ridge_Regression = 1 - (SSE / SST);

mae_Ridge_Regression = mean(abs(Y_pred_Ridge_Regression - Y_pred_Ridge_Regression));

mse_Ridge_Regression = mean((Y_pred_Ridge_Regression - Y_pred_Ridge_Regression).^2);

fprintf('R squared_Ridge_Regression: %.4f\n', r_squared_Ridge_Regression);
fprintf('MAE_Ridge_Regression: %.4f\n', mae_Ridge_Regression);
fprintf('MSE_Ridge_Regression: %.4f\n', mse_Ridge_Regression);
%}

%% 3.Lasso Regression 

%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

lambda = 0.01; % Adjust the regularization parameter as needed
mdl = fitrlinear(X_train_matrix, Y_train_vector, 'Learner', 'leastsquares', 'Lambda', lambda);

X_test_matrix = table2array(X_test);

Y_pred = predict(mdl, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_Lasso_Regression = 1 - (SSE / SST);

mae_Lasso_Regression = mean(abs(Y_pred - Y_test));

mse_Lasso_Regression = mean((Y_pred - Y_test).^2);

fprintf('R squared_Lasso_Regression: %.4f\n', r_squared_Lasso_Regression);
fprintf('MAE_Lasso_Regression: %.4f\n', mae_Lasso_Regression);
fprintf('MSE_Lasso_Regression: %.4f\n', mse_Lasso_Regression);
%}

%% 4.Support Vector Regression (SVR):
%{

X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

svrModel = fitrsvm(X_train_matrix, Y_train_vector);

X_test_matrix = table2array(X_test);

Y_pred = predict(svrModel, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_SVR = 1 - (SSE / SST);

mae_SVR = mean(abs(Y_pred - Y_test));

mse_SVR = mean((Y_pred - Y_test).^2);

fprintf('R squared_SVR: %.4f\n', r_squared_SVR);
fprintf('MAE_SVR: %.4f\n', mae_SVR);
fprintf('MSE_SVR: %.4f\n', mse_SVR);
%}

%% 5.Regression Trees
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

tree_model = fitrtree(X_train_matrix, Y_train_vector);

X_test_matrix = table2array(X_test);

Y_pred = predict(tree_model, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_Regression_Trees = 1 - (SSE / SST);

mae_Regression_Trees = mean(abs(Y_pred - Y_test));

mse_Regression_Trees = mean((Y_pred - Y_test).^2);

fprintf('R squared_Regression_Trees: %.4f\n', r_squared_Regression_Trees);
fprintf('MAE_Regression_Trees: %.4f\n', mae_Regression_Trees);
fprintf('MSE_Regression_Trees: %.4f\n', mse_Regression_Trees);
% R squared: 0.8618
% MAE: 4.7750
% MSE: 41.8548
%}

%% 6.Ensemble Methods:
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

numTrees = 100; % Number of trees in the ensemble
mdl = fitrensemble(X_train_matrix, Y_train_vector, 'Method', 'LSBoost', 'NumLearningCycles', numTrees);

% Convert X_test to a matrix
X_test_matrix = table2array(X_test);

% Predict using the model
Y_pred = predict(mdl, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_Ensemble_Methods = 1 - (SSE / SST);

mae_Ensemble_Methods = mean(abs(Y_pred - Y_test));

mse_Ensemble_Methods = mean((Y_pred - Y_test).^2);

fprintf('R squared_Ensemble_Methods: %.4f\n', r_squared_Ensemble_Methods);
fprintf('MAE_Ensemble_Methods: %.4f\n', mae_Ensemble_Methods);
fprintf('MSE_Ensemble_Methods: %.4f\n', mse_Ensemble_Methods);
%R squared: 0.8944
%MAE: 3.8828
%MSE: 31.9774
%}

%% 7.Gaussian Process Regression
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

mdl = fitrgp(X_train_matrix, Y_train_vector);

X_test_matrix = table2array(X_test);

Y_pred = predict(mdl, X_test_matrix);

SSE = sum((Y_test - Y_pred).^2);
SST = sum((Y_test - mean(Y_test)).^2);
r_squared_GPR = 1 - (SSE / SST);

mae_GPR = mean(abs(Y_pred - Y_test));

mse_GPR = mean((Y_pred - Y_test).^2);

fprintf('R squared_GPR: %.4f\n', r_squared_GPR);
fprintf('MAE_GPR: %.4f\n', mae_GPR);
fprintf('MSE_GPR: %.4f\n', mse_GPR);
%}

%% Compare the rsquared value of different model
%{
modelNames = {
    'Linear Regression', 
    'Ridge Regression', 
    'Lasso Regression', 
    'SVR', 
    'Regression Trees', 
    'Ensemble Methods', 
    'Gaussian Process Regression'
};

rSquaredValues = [
    r_squared_linreg, 
    r_squared_Ridge_Regression, 
    r_squared_Lasso_Regression, 
    r_squared_SVR, 
    r_squared_Regression_Trees, 
    r_squared_Ensemble_Methods, 
    r_squared_GPR
];

figure;
bar(rSquaredValues);
xticks(1:numel(modelNames));
xticklabels(modelNames);
xtickangle(45);
ylabel('R-squared Value');
title('Comparison of R-squared Values for Different ML Models');
%}


%% Prediction error plot fro GRP
%{
X_train_matrix = table2array(X_train);
Y_train_vector = Y_train;

mdl = fitrgp(X_train_matrix, Y_train_vector);

X_test_matrix = table2array(X_test);

Y_pred = predict(mdl, X_test_matrix);
residuals = Y_test - Y_pred;

figure;
scatter(Y_pred, residuals);
xlabel('Predicted Values');
ylabel('Residuals');
title('Residual Plot');
grid on;

hold on;
plot(get(gca, 'xlim'), [0 0], 'k--');

residual_std = std(residuals);
fprintf('Standard Deviation of Residuals: %.4f\n', residual_std);
%}

%% %% Taking input from user to predict:
%{
input_cement = input('Enter Cement value: ');
input_blast_furnace_slag = input('Enter Blast Furnace Slag value: ');
input_fly_ash = input('Enter Fly Ash value: ');
input_water = input('Enter Water value: ');
input_superplasticizer = input('Enter Superplasticizer value: ');
input_coarse_aggregate = input('Enter Coarse Aggregate value: ');
input_fine_aggregate = input('Enter Fine Aggregate value: ');
input_age = input('Enter Age value: ');

user_input = table(input_cement, input_blast_furnace_slag, input_fly_ash, input_water, ...
                   input_superplasticizer, input_coarse_aggregate, input_fine_aggregate, input_age);

% Predict the strength using the trained GPR model
user_input_matrix = table2array(user_input);
predicted_strength = predict(mdl, user_input_matrix);

fprintf('Predicted Strength: %.4f\n', predicted_strength);
%}
