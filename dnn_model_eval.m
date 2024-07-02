function metricOutput = dnn_model_eval(trialInfo)

% set task type
params = parameters();
seqSteps = 10;
tForceStop = 1;% time stop force
tSpan = 0:0.01:10; % simulation time span
ctrlOptions = control_options();
disp("initialize parameters");

modelType = "dnn"; % "dnn", "pinn", "lstm"
net = trialInfo.trainedNetwork;
predInterval = 3;
F1Min = max(20,params(10));

% Prediction Accuracy evluation
% evaluate the model with specified forces, and time steps
numCase = 50;
numTime = 100;
refTime = linspace(1,10,numTime);
maxForces = linspace(0.5,15,numCase);
errs = zeros(6*numCase,numTime);
for i = 1:numCase
    % reference
    ctrlOptions.fMax = [F1Min+maxForces(i);0];
    y = sdpm_simulation(tSpan, ctrlOptions);
    t = y(:,1);
    x = y(:,4:9);
    xp = predict_motion(net,modelType,t,x,predInterval,seqSteps,tForceStop);
    % test points
    tTestIndices = zeros(1,numTime);
    for k = 1:numTime
        indices = find(t<=refTime(k));
        tTestIndices(1,k) = indices(end);
    end
    rmseErr = root_square_err(tTestIndices,x(:,1:6),xp(:,1:6));
    idx = 6*(i-1);
    errs(idx+1,:) = rmseErr(1,:);
    errs(idx+2,:) = rmseErr(2,:);
    errs(idx+3,:) = rmseErr(3,:);
    errs(idx+4,:) = rmseErr(4,:);
    errs(idx+5,:) = rmseErr(5,:);
    errs(idx+6,:) = rmseErr(6,:);
end
metricOutput = mean(errs,'all');
 
end

% supporting functions
function xp = predict_motion(net,type,t,x,predInterval,seqSteps,tForceStop)
    % prediction
    numTime = length(t);
    initIdx = find(t >= tForceStop,1,'first');
    xp = zeros(numTime,6);
    xp(1:initIdx,:) = x(1:initIdx,:);
    switch type
        case "dnn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                if (t(i)-t0) > predInterval
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        case "lstm"
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
            t0 = t(initIdx);
            for i = initIdx+1:numTime          
                if (t(i)-t0) >= predInterval
                    initIdx = i-1;
                    startIdx = initIdx-seqSteps+1;
                    x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
                    t0 = t(initIdx);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        case "pinn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                if (t(i)-t0 > predInterval)
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        otherwise
            disp("unsupport type model");
    end
end

function xp = predict_step_state(net,type,xInit,tPred)
    xp = zeros(1,6);
    switch type
        case "dnn"
            xp = predict(net,[xInit,tPred]);
        case "lstm"
            dsState = arrayDatastore(xInit,'OutputType',"same",'ReadSize',128);
            dsTime = arrayDatastore(tPred,'ReadSize',128);
            dsTest = combine(dsState, dsTime);
            xp = predict(net,dsTest);
        case "pinn"
            xInit = dlarray([xInit(1:4),tPred]','CB');
            xp(1:4) = extractdata(predict(net,xInit));
        otherwise 
            disp("unsupport model type")
    end
end
