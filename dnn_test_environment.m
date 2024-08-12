%% Description
% Test environment for trained models

close all;
clear; 
clc;

%% settings
sysParams = params_system();
tSpan = 0:0.002:10;
% tSpan = 0:0.005:10;
tRMSE = floor(length(tSpan)/2); % time steps not in rmse calculation
tForceStop = 1;
ctrlParams = params_control();

% modelFile = "best_dnn_models.mat";
% modelFile2 = "best_dnn_models2.mat";
modelFile = "best_dnn_models_3.mat";
% modelFile = "best_dnn_models_4.mat";
maxEpochs = 50;
F1Min = max(5,sysParams.fc_max);
Fmax = 17;

%% Test 1
net = load(modelFile).model_6_256_200;
% net = load(modelFile).model_10_256_600;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 1", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 2
net = load(modelFile).model_7_128_200;
% net = load(modelFile).model_12_128_400;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 2", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 3
net = load(modelFile).model_8_128_200;
% net = load(modelFile).model_12_128_600;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 3", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 4
net = load(modelFile).model_8_256_200;
% net = load(modelFile).model_8_128_600;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 4", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 5
net = load(modelFile).model_8_32_200;
% net = load(modelFile).model_8_256_600;
ctrlParams.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best model 5", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlParams.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% root square error function
function rse = root_square_err(indices, x, xp)
    % root square error of prediction and reference
    numPoints = length(indices);
    x_size = size(xp);
    errs = zeros(x_size(2), numPoints);
    for i = 1 : numPoints
        for j = 1:x_size(2)
            errs(j, i) = x(indices(i), j) - xp(indices(i), j);
        end
    end
    rse = sqrt(errs.^2);
end