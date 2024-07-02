function output = pinn_training(params,monitor)

%initialize output
output.trainedNet = [];
output.executionEnvironment = "auto";

% settings
ds = load('trainingData.mat');
numSamples = params.numSamples;
maxEpochs = 60;

% generate data
% Feature data: 4-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot]
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:7,:); % q1,q2,q1_dot,q2_dot
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
% Split test and validation data
xdata = xTrain;
ydata = yTrain;
training_percent = 0.9;
ln = length(xTrain);
indices = randperm(ln);
num_train = round(ln*training_percent);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);
xTrain = xdata(:,train_indices);
yTrain = ydata(:,train_indices);
xVal = xdata(:,test_indices);
yVal = ydata(:,test_indices);
validationFrequency = 50;

% make dnn and train 
numStates = 4; % q1,q2,q1dot,q2dot
% make dnn and train 
numLayers = params.numLayers;
numNeurons = params.numNeurons;
dropoutProb = params.dropoutProb;
layers = featureInputLayer(numStates+1);
for i = 1:numLayers
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        eluLayer %reluLayer
        dropoutLayer(dropoutProb)]; 
end
layers = [
    layers
    fullyConnectedLayer(numStates)];
layers = layerGraph(layers);

% convert the layer array to a dlnetwork object
net = dlnetwork(layers);
% disp(net); 
% plot(net)

% training options
monitor.Metrics = ["Loss"  "ValidationLoss" "TestAccuracy"];
monitor.Info = ["LearnRate","IterationPerEpoch","MaximumIteration","Epoch","Iteration"];
monitor.XLabel = "Epoch";

% using stochastic gradient decent
miniBatchSize = 200;
learnRate = 0.001;
learnRatedrop = 0.2;
learnRateDropPeriod = 10;
dataSize = size(yTrain,2);
numBatches = floor(dataSize/miniBatchSize);
numIterations = maxEpochs * numBatches;

momentum = 0.9; % for sgdmupdate
velocity = [];  % for sgdmupdate
averageGrad = [];
averageSqGrad = [];
iter = 0;
epoch = 0;
while epoch < maxEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Shuffle data.
    idx = randperm(dataSize);
    xTrain = xTrain(:,idx);
    yTrain = yTrain(:,idx);
    for j=1:numBatches
        iter  = iter + 1;
        startIdx = (j-1)*miniBatchSize+1;
        endIdx = min(j*miniBatchSize, dataSize);
        xBatch = xTrain(:,startIdx:endIdx);
        yBatch = yTrain(:,startIdx:endIdx); 
        X = gpuArray(dlarray(xBatch,"CB"));
        T = gpuArray(dlarray(yBatch,"CB"));
        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(@modelLoss,net,X,T);

        % Update the network parameters using the SGDM optimizer.
        % [net,velocity] = sgdmupdate(net,gradients,vel,learnRate,momentum);

        % Update the network parameters using the ADAM optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iter,learnRate);

        recordMetrics(monitor,iter,Loss=loss);
        if iter == 1 || mod(iter,validationFrequency) == 0
            xVal = gpuArray(dlarray(xVal,"CB"));
            yVal = gpuArray(dlarray(yVal,"CB"));
            [lossValidation,~] = dlfeval(@modelLoss,net,xVal,yVal);

            recordMetrics(monitor,iter, ...
                ValidationLoss=lossValidation);
        end

        if mod(iter,maxEpochs) == 0
            updateInfo(monitor,LearnRate=learnRate,Epoch=epoch,Iteration=iter,MaximumIteration=numIterations,IterationPerEpoch=numBatches);
            monitor.Progress = 100*iter/numIterations;
        end
    end
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate*learnRatedrop;
    end
    output.trainedNet = net;
end

recordMetrics(monitor,iter,...
        TestAccuracy=pinn_model_eval(net))

% loss function
function [loss, gradients, state] = modelLoss(net,X,T)
    % make prediction
    [Y, state] = forward(net,X);
    dataLoss = l2loss(Y,T);
   
    % compute gradients using automatic differentiation
    % predict
    q1 = Y(1,:);
    q2 = Y(2,:);
    q1d = Y(3,:);
    q2d = Y(4,:);
    % q1X = dlgradient(sum(q1,'all'), X);
    % q2X = dlgradient(sum(q2,'all'), X);
    q1dX = dlgradient(sum(q1d,'all'), X);
    q2dX = dlgradient(sum(q2d,'all'), X);
    % q1d = q1X(5,:);
    % q2d = q2X(5,:); 
    q1dd = q1dX(5,:);
    q2dd = q2dX(5,:);

    % target
    q1T = T(1,:);
    q2T = T(2,:);
    q1dT = T(3,:);
    q2dT = T(4,:);
    % q1X = dlgradient(sum(q1,'all'), X);
    % q2X = dlgradient(sum(q2,'all'), X);
    q1dXT = dlgradient(sum(q1dT,'all'), X);
    q2dXT = dlgradient(sum(q2dT,'all'), X);
    % q1d = q1X(5,:);
    % q2d = q2X(5,:); 
    q1ddT = q1dXT(5,:);
    q2ddT = q2dXT(5,:);

    f = physics_law([q1;q2],[q1d;q2d],[q1dd;q2dd]);
    fT = physics_law([q1T;q2T],[q1dT;q2dT],[q1ddT;q2ddT]);
    %zeroTarget = zeros(size(f),"like",f);
    physicLoss = l2loss(f,fT);
    
    % total loss
    ctrlOptions = control_options();
    loss = (1.0-ctrlOptions.alpha)*dataLoss + ctrlOptions.alpha*physicLoss;
    gradients = dlgradient(loss, net.Learnables);
end

end