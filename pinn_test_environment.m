%% Description
% Test environment for trained models

close all;
clear; 
clc;

%% settings
params = parameters();
tSpan = 0:0.01:10;
tRMSE = 500; % time steps not in rmse calculation
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = length(ds.samples);
modelFile = "best_PINN_models.mat";
maxEpochs = 50;
F1Min = max(20,params(10));
Fmax = 5;

%% Test 1
net = load(modelFile).best_val_loss.trainedNet;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,1:4);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),4);
for i = 1:length(tp)
    xp(i,1:4) = extractdata(predict(net,dlarray([x0,tp(i)-t0]','CB')));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best PINN model", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)
