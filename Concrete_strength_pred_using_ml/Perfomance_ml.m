df=readtable("/Users/user/Documents/projects/Concrete Strength prediction using ML(matlab)/concrete_data.csv");
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