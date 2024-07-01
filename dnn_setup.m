function [xTrain,yTrain,layers,options] = dnn_setup(params)

% settings
ds = load('trainingData.mat');
numSamples = params.numSamples;
maxEpochs = 50;

% generate data
% Feature data: 6-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot,q1ddot,q2ddot]
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:); % q1,q2,q1_dot,q2_dot
    for tInit = initTimes
        initIdx = find(t >= tInit,1,'first');
        x0 = x(:,initIdx); % Initial state 
        t0 = t(initIdx); % Start time
        for j = initIdx+1:length(t)
            xTrain = [xTrain,[x0; t(j)-t0]];
            yTrain = [yTrain,x(:,j)];
        end
    end
end
%disp([num2str(length(xTrain)),' samples are generated for training.'])
xTrain = xTrain';
yTrain = yTrain';

% Split test and validation data
xdata = xTrain;
ydata = yTrain;
training_percent = 0.9;
size = length(xTrain);
indices = randperm(size);
num_train = round(size*training_percent);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);
xTrain = xdata(train_indices,:);
yTrain = ydata(train_indices,:);
xVal = xdata(test_indices,:);
yVal = ydata(test_indices,:);

% make dnn and train 
numLayers = params.numLayers;
numNeurons = params.numNeurons;
dropoutProb = params.dropoutProb;
numStates = 6; % 6-dim states in the first second
layers = featureInputLayer(numStates+1);
for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        reluLayer
        dropoutLayer(dropoutProb)]; 
end
layers = [
    layers
    fullyConnectedLayer(numStates)
    myRegressionLayer("mse")];
layers = layerGraph(layers);

% Create options
miniBatchSize = 200; % params.miniBatchSize;
InitialLearnRate = 1e-3; % params.InitialLearnRate;
LearnRateDropFactor = 0.2; % params.LearnRateDropFactor;
options = trainingOptions("adam",MaxEpochs=maxEpochs,Verbose=false,Plots="training-progress",...
    InitialLearnRate=InitialLearnRate,LearnRateSchedule="piecewise",LearnRateDropFactor=LearnRateDropFactor,...
    LearnRateDropPeriod=10,ValidationData={xVal,yVal},MiniBatchSize=miniBatchSize);

end