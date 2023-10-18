%% Resample
load('Waves_NoAtten.mat');

rng(0);
Xn=W30_2(:,:,:);
figure
imagesc(alpha)
axis square
colorbar

Xmin = min(Xn(:));
Xmax = max(Xn(:));
Xn = 2*((Xn-Xmin)/(Xmax-Xmin)-0.5);
noise_perc = 0;
noise = (std(Xn(:))*noise_perc)*randn(size(Xn)); 
Xn = Xn+noise;

x_lb = 1;
y_lb = 1;
t_lb = 2;

x_range = 29;
y_range = 29;
t_range = 197;

x_onecol = x_lb*dx: dx: x_lb*dx+x_range*dx; 
y_onerow = y_lb*dy: dy: y_lb*dy+y_range*dy;
t_oneslice = t_lb*dt: dt: t_lb*dt+t_range*dt;

Usel = Xn(x_lb:x_lb+x_range, y_lb:y_lb+y_range, t_lb:t_lb+t_range);

% vectorize
U0 = [];
for i_t=1:size(Usel,3)
    for i_y=1:size(Usel,2)
        for i_x=1:size(Usel,1)
            U0 = [U0, Usel(i_x,i_y,i_t)];
        end
    end
end

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

% X0, Y0, T0 are vectorized indices with available measurements, can be
% used for loss_u.
X1 = repmat(x_onecol,1,length(y_onerow));
X0 = repmat(X1(Smpl),1,length(t_oneslice)); %vectorize
Y0=[];
for i=1:length(y_onerow)
    Y0 = [Y0, y_onerow(i)*ones(1,length(x_onecol))];
end
Y0 = repmat(Y0(Smpl),1,length(t_oneslice));
T0 = [];
for i = 1:length(t_oneslice)
    T0 = [T0, t_oneslice(i)*ones(1,length(Smpl))];
end

% X,Y,T are vectorized indices for all points in the ROI, can be used for loss_f. 
t_intoneslice = t_oneslice; % In our settings, if a sensor works well at one time step, it works well across all the time.
T = [];
for i = 1:length(t_intoneslice)
    T = [T, t_intoneslice(i)*ones(1,length(x_onecol)*length(y_onerow))];
end
Y=[];
for i=1:length(y_onerow)
    Y = [Y, y_onerow(i)*ones(1,length(x_onecol))];
end
Y = repmat(Y,1,length(t_intoneslice));
X = repmat(repmat(x_onecol,1,length(y_onerow)),1,length(t_intoneslice)); %vectorize

% The row and column index of the locations with available measurements
M1 = mod(Smpl-1,size(Usel,2))+1;%x
M2 = floor((Smpl-1)/size(Usel,2))+1;%y
M = [M1',M2'];

U0 = [];
for t=t_lb:t_lb+t_range
    for i=1:size(M,1)
        U0 = [U0, Xn(M(i,1),M(i,2),t)];
    end
end

Urec = nan(size(Usel,1),size(Usel,2),size(Usel,3));
for i_t=1:size(Usel,3)
    for i_m=1:length(M)
        Urec(M(i_m,1),M(i_m,2),i_t) = U0((i_t-1)*length(M)+i_m);
    end
end

figure
for i=1:size(Usel,3)
    Data_Array = Urec(:,:,i);
    imAlpha=ones(size(Data_Array));
    imAlpha(isnan(Data_Array))=0;
    imagesc(Data_Array,'AlphaData',imAlpha);
    set(gca,'color',0*[1 1 1]);
    axis square
    colorbar
    caxis([-1 1])
    title(num2str(i))
    pause(.2)
end
%% 
ds = arrayDatastore([X' Y' T']);
%% Network parameters
numLayers = 5;
numNeurons = 200;
sz = [numNeurons 3];
numIn = numNeurons;
%% Network initialization
parameters = struct;

parameters.fc1.Weights = initializeHe(sz,3);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

rnk = 5;
mu = 0;
sigma = 0.1;

parameters.UL1 = initializeGaussian([length(x_onecol),rnk],mu,sigma);%([length(x_onecol),rnk]);
parameters.SVR1 = initializeGaussian([rnk,length(x_onecol)],mu,sigma);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

parameters.("fc" + numLayers).Weights = initializeHe([1,numNeurons],numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
%% Configure training parameters
numEpochs = 5000;
miniBatchSize = length(X0);

executionEnvironment = "auto";

initialLearnRate = 1e-3;
decayRate = 2.5e-4;

Weight.lossF = 1;
Weight.lossU = 10;
Weight.lossS = 10;
Weight.lossB = 10;
%% Configure a minibatch
lenT = 11; 
mbs = size(Usel,1)*size(Usel,2)*lenT;
mbq = minibatchqueue(ds, ...
    MiniBatchSize=mbs, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);
%%
X0 = dlarray(X0,"CB");
Y0 = dlarray(Y0,"CB");
T0 = dlarray(T0,"CB");
% Normalization
ampAdj = 1/(max(U0(:))-min(U0(:)));
biasAdj = min(U0(:));
U0 = 2*(U0-biasAdj)*ampAdj-1;

U0 = dlarray(U0);

averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelLoss_2D_no_atten);
%%
C = c;
iteration = 0;
RECORD = cell(numEpochs,4);
LOSS = zeros(numEpochs,5);
LOSS_in = zeros(6,5);

% Known entries are located in the right, bottom boundaries and the
% diagonal. This is the row and column index of the locations within
% coor_ind_add, and is hard-coded here to be less confusing. For a different coor_ind_add,
% change the addon manualy.
addon = zeros(88,2);
for i=1:30 % diagonal
    addon(i,:) = i;
end
for i=31:59 % bottom
    addon(i,1) = 30;
    addon(i,2) = i-30;
end
for i=60:88 % right
    addon(i,1) = i-59;
    addon(i,2) = 30;
end

%%
numParams = 1; % Only 1 PDE parameter -c^2 to be recovered.
for epoch = 1:numEpochs
    tic

    reset(mbq);
    ep_in = 0;
    iteration = iteration + 1;
    learningRate = initialLearnRate / (1+decayRate*iteration);
    while hasdata(mbq)
        ep_in = ep_in+1
       
        XYT = next(mbq);
        X = XYT(1,:);
        Y = XYT(2,:);
        T = XYT(3,:);

       [loss,lossF,lossU,lossS,lossB,LAM1,gradients,vU,PDEterms] = dlfeval(accfun,parameters,X,Y,T,X0,Y0,T0,U0,lenT,C,Weight,addon,numParams);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);

        RECORD{epoch,1} = LAM1;
        RECORD{epoch,2} = vU;
        
        LOSS_in(ep_in,1) = double(gather(extractdata(loss)));
        lossF_in = double(gather(extractdata(lossF)));
        LOSS_in(ep_in,2) = lossF_in;
        lossU_in = double(gather(extractdata(lossU)));
        LOSS_in(ep_in,3) = lossU_in;
        lossS_in = double(gather(extractdata(lossS)));
        LOSS_in(ep_in,4) = lossS_in;
        lossB_in = double(gather(extractdata(lossB)));
        LOSS_in(ep_in,5) = lossB_in;
    end
    epoch
    if(mod(epoch,100)==0)
        RECORD{epoch,3} = gradients;
        RECORD{epoch,4} = PDEterms;
    end
    loss = double(gather(extractdata(loss)));

    LOSS(epoch,1) = loss;
    lossF = double(gather(extractdata(lossF)));
    LOSS(epoch,2) = lossF;
    lossU = double(gather(extractdata(lossU)));
    LOSS(epoch,3) = lossU;
    lossS = double(gather(extractdata(lossS)));
    LOSS(epoch,4) = lossS;
    lossB = double(gather(extractdata(lossB)));
    LOSS(epoch,5) = lossB;
    LOSS(epoch,1:5)
    if(mod(epoch,500)==0 || epoch==5)
        fname = strcat('Saved/',strcat(num2str(epoch),'STD0_RBD_r5.mat'));
        save(fname);
    end
    toc
end
