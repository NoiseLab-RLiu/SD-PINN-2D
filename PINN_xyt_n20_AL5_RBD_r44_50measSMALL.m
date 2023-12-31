%% Data Preparation
load('Waves_ObsAttenAlp5.mat');
% Normalize and add noise
rng(0);
Xn=W30_2(6:15,6:15,533:598);
Xn = 2*((Xn-min(Xn(:)))/(max(Xn(:))-min(Xn(:)))-0.5);
noise_perc = 0.2;
noise = (std(Xn(:))*noise_perc)*randn(size(Xn)); 
Xn = Xn+noise;
% Define the range of ROI
x_lb = 1;
y_lb = 1;
t_lb = 1;
x_range = 9;
y_range = 9;
t_range = 65;
x_onecol = x_lb*dx: dx: x_lb*dx+x_range*dx; 
y_onerow = y_lb*dy: dy: y_lb*dy+y_range*dy;
t_oneslice = t_lb*dt: dt: t_lb*dt+t_range*dt;
Usel = Xn(x_lb:x_lb+x_range, y_lb:y_lb+y_range, t_lb:t_lb+t_range);
% Add masks to the measurements
coor_ind_ori = 1:length(y_onerow)*length(x_onecol);
coor_ind_add = [];
for i=1:length(coor_ind_ori)
    % record the index for the entries located on the right, bottom
    % boundaries and the diagonal (RBD)
    if(mod(i,size(Usel,2))==0 || i > size(Usel,2)*(size(Usel,1)-1) || mod(i-1,size(Usel,2)+1)==0)
        coor_ind_add = [coor_ind_add,coor_ind_ori(i)];
        coor_ind_ori(i) = -1;
    end
end
coor_ind_from = coor_ind_ori;
coor_ind_from(coor_ind_from == -1) = []; % Do not remove the measurements on RBD randomly
rand_perc = 0.5; % percentage of kept locations with measurements (not considering locations on RBD)
Smpl = randsample(coor_ind_from, floor(length(coor_ind_from)*rand_perc));
Smpl = [Smpl,coor_ind_add]; % Locations with measurements
[Smpl,sind] = sort(Smpl,'ascend');
% The row and column index of the locations with available measurements
M1 = mod(Smpl-1,size(Usel,2))+1;%x
M2 = floor((Smpl-1)/size(Usel,2))+1;%y
M = [M1',M2'];
% X0, Y0, T0 are vectorized indices with available measurements, can be
% used for loss_u. U0 is the available measurements at X0,Y0,T0
X1 = repmat(x_onecol,1,length(y_onerow));
X0 = repmat(X1(Smpl),1,length(t_oneslice)); %vectorize
Y1 = repelem(y_onerow, length(x_onecol));
Y0 = repmat(Y1(Smpl),1,length(t_oneslice));
T0 = repelem(t_oneslice,length(Smpl));
U0 = [];
for t = t_lb:t_lb+t_range
    U0 =  [U0, Xn(sub2ind(size(Xn), M(:, 1)', M(:, 2)', repmat(t, [1, size(M,1)])))];
end
% X,Y,T are vectorized indices for all points in the ROI, can be used for loss_f. 
T = repelem(t_oneslice,length(x_onecol)*length(y_onerow));
Y = repmat(Y1,1,length(t_oneslice));
X = repmat(X1,1,length(t_oneslice));

%% Network Preparation
ds = arrayDatastore([X' Y' T']);
% Hyperparameters
numLayers = 5;
numNeurons = 50;
sz = [numNeurons 3];
numIn = numNeurons;
% Initialization
parameters = struct;
parameters.fc1.Weights = initializeHe(sz,3);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);
% Low rank assumption
rnk = 5;
rnkAlpha = 5;

mu = 0;
sigma = 0.1;
parameters.Umat1 = initializeGaussian([length(x_onecol),rnk],mu,sigma);%([length(x_onecol),rnk]);
parameters.Vmat1 = initializeGaussian([rnk,length(x_onecol)],mu,sigma);
parameters.Umat2 = initializeGaussian([length(x_onecol),rnkAlpha],mu,sigma);%([length(x_onecol),rnk]);
parameters.Vmat2 = initializeGaussian([rnkAlpha,length(x_onecol)],mu,sigma);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end
parameters.("fc" + numLayers).Weights = initializeHe([1,numNeurons],numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);

%% Training configuration
numEpochs = 600;
miniBatchSize = length(X0);

executionEnvironment = "auto";

initialLearnRate = 1e-3;
decayRate = 2.5e-4;

Weight.lossF = 1;
Weight.lossU = 10;
Weight.lossS = 10;
Weight.lossB = 10;

lenT = 11; 
mbs = size(Usel,1)*size(Usel,2)*lenT;
mbq = minibatchqueue(ds, ...
    MiniBatchSize=mbs, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);

X0 = dlarray(X0,"CB");
Y0 = dlarray(Y0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(2*(U0-min(U0(:)))*(1/(max(U0(:))-min(U0(:))))-1);

averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelLoss_2D_atten);
%% Specify locations of given PDE coefficients
% Known entries are located in the right, bottom boundaries and the
% diagonal. This is the row and column index of the locations within
% coor_ind_add, and is hard-coded here to be less confusing. For a different coor_ind_add,
% change the addon manualy.
addon = zeros(28,2);
for i=1:10 % diagonal
    addon(i,:) = i;
end
for i=11:19 % bottom
    addon(i,1) = 10;
    addon(i,2) = i-10;
end
for i=20:28 % right
    addon(i,1) = i-19;
    addon(i,2) = 10;
end

%% Start Training
iteration = 0;
RECORD = cell(numEpochs,2);
LOSS = zeros(numEpochs,5);
C = c(6:15,6:15);
ALPHA = alpha(6:15,6:15);
numParams = 2; % 2 PDE parameters -c^2 and alpha to be recovered.
for epoch = 1:numEpochs
    tic
    reset(mbq);
    iteration = iteration + 1;
    learningRate = initialLearnRate / (1+decayRate*iteration);
    while hasdata(mbq)
        XYT = next(mbq);
        X = XYT(1,:);
        Y = XYT(2,:);
        T = XYT(3,:);

       [loss,lossF,lossU,lossS,lossB,LAM1,LAM2,gradients] = dlfeval(accfun,parameters,X,Y,T,X0,Y0,T0,U0,lenT,C,ALPHA,Weight,addon,numParams);
        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);    
    end
    if(mod(epoch,100)==0)
        RECORD{epoch,1} = LAM1;
        RECORD{epoch,2} = LAM2;
    end
    loss = double(gather(extractdata(loss)));
    LOSS(epoch,1) = loss;
    LOSS(epoch,2) = lossF;
    LOSS(epoch,3) = lossU;
    LOSS(epoch,4) = lossS;
    LOSS(epoch,5) = lossB;

    fprintf('epoch %d: loss = %e  %e   %e\n', epoch, loss, lossF);
    if(mod(epoch,500)==0 || epoch==5)
        fname = strcat('Saved/',strcat(num2str(epoch),'STD20_RBD_r44SMALL.mat'));
        save(fname);
    end
    toc
end

%%
figure
subplot(2,2,1)
imagesc(-C.^2)
axis square
colorbar
title('True -c^2')
subplot(2,2,2)
imagesc(extractdata(gather(LAM1)))
axis square
colorbar
title('Recovered -c^2')
subplot(2,2,3)
imagesc(ALPHA)
axis square
colorbar
title('True alpha')
subplot(2,2,4)
imagesc(extractdata(gather(LAM2)))
axis square
colorbar
title('Recovered alpha')
