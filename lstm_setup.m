function [dsTrain,layers,options] = lstm_setup(params)

% settings
ds = load('trainingData.mat');
numSamples = params.numSamples;
maxEpochs = 50;
seqSteps = params.seqSteps;

% preprocess data for training
% Refer to the Help "Import Data into Deep Network Designer / Sequences and time series" 
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
states = {};
times = [];
labels = [];
for i=1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:);
    for tInit = initTimes
        initIdx = find(t >= tInit, 1, 'first');
        startIdx = initIdx-seqSteps+1;
        t0 = t(initIdx);
        x0 = [t(startIdx:initIdx);x(:,startIdx:initIdx)];
        for j=initIdx+1:length(t)
            states{end+1} = x0;
            times = [times,t(j)-t0];
            labels = [labels,x(:,j)];
        end
    end
end
%disp([num2str(length(times)),' samples are generated for training.'])
states = reshape(states,[],1);
times = times';
labels = labels';

% Split test and validation data
training_percent = 0.90;
size = length(times);
indices = randperm(size);
num_train = round(size*training_percent);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);
xTrain = cell(length(train_indices),1);
for id = 1:length(train_indices)
    xTrain{id} = states{train_indices(id)};
end
xVal = cell(length(test_indices),1);
for id = 1:length(test_indices)
    xVal{id} = states{test_indices(id)};
end
tTrain = times(train_indices);
yTrain = labels(train_indices,:);
tVal = times(test_indices);
yVal = labels(test_indices,:);

% combine a datastore for training
miniBatchSize = 200;
dsState = arrayDatastore(xTrain,'OutputType',"same",'ReadSize',miniBatchSize);
dsTime = arrayDatastore(tTrain,'ReadSize',miniBatchSize);
dsLabel = arrayDatastore(yTrain,'ReadSize',miniBatchSize);
dsTrain = combine(dsState, dsTime, dsLabel);

dsState = arrayDatastore(xVal,'OutputType',"same",'ReadSize',miniBatchSize);
dsTime = arrayDatastore(tVal,'ReadSize',miniBatchSize);
dsLabel = arrayDatastore(yVal,'ReadSize',miniBatchSize);
dsVal = combine(dsState, dsTime, dsLabel);

% make dnn and train 
numLayers = params.numLayers;
numNeurons = params.numNeurons;
dropoutProb = params.dropoutProb;
numStates = 6; % 6-dim states in the first second
layers = [
    sequenceInputLayer(numStates+1)
    lstmLayer(32,OutputMode="last")
    concatenationLayer(1,2,Name="cat")];
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
layers = addLayers(layers,[...
    featureInputLayer(1,Name="time")]);
layers = connectLayers(layers,"time","cat/in2");

% Create options
InitialLearnRate = 1e-3; % params.InitialLearnRate;
LearnRateDropFactor = 0.2; % params.LearnRateDropFactor;
options = trainingOptions("adam",MaxEpochs=maxEpochs,Verbose=false,Plots="training-progress",...
    InitialLearnRate=InitialLearnRate,LearnRateSchedule="piecewise",LearnRateDropFactor=LearnRateDropFactor,...
    LearnRateDropPeriod=10,ValidationData=dsVal,MiniBatchSize=miniBatchSize);

end