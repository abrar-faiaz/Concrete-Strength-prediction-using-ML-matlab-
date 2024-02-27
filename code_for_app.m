df=readtable("/Users/user/Documents/projects/Concrete Strength prediction using ML(matlab)/concrete_data.csv");
%% Correlation plot  

corrplot(df)

%Cement, Superplasticizer, Age Have a positive effect on the Concrete's
% Strength, while, Water has a negative effect.
%% Correlation Heatmap

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

%% Train Test set Split


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

%% 7.Gaussian Process Regression

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

%% Taking input from user to predict:

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

user_input_matrix = table2array(user_input);
predicted_strength = predict(mdl, user_input_matrix);

fprintf('Predicted Strength: %.4f\n', predicted_strength);
