%% Resample
load('Waves_ObsAttenAlp5.mat');

rng(0);
Xn=W30_2(:,:,301:500);
figure
imagesc(alpha)
axis square
colorbar

Xmin = min(Xn(:));
Xmax = max(Xn(:));
Xn = 2*((Xn-Xmin)/(Xmax-Xmin)-0.5);
noise = (std(Xn(:))*0.2)*randn(size(Xn)); 
Xn = Xn+noise;
x_lb = 1*dx;
y_lb = 1*dy;
t_lb = 2*dt;

x_onecol = x_lb: dx: x_lb+29*dx; 
y_onerow = y_lb:dy:y_lb+29*dy;
t_oneslice = t_lb: dt: t_lb+197*dt;

%t_intoneslice = 0.02:0.005:1.995;
t_intoneslice = t_lb:dt:t_lb+197*dt;
T = [];
for i = 1:length(t_intoneslice)
    T = [T, t_intoneslice(i)*ones(1,length(x_onecol)*length(y_onerow))];
end

Usel = Xn(:,:, 2:1:2+197);

U0 = [];
for k=1:198
    for j=1:30
        for i=1:30
            U0 = [U0, Usel(i,j,k)];
        end
    end
end

% test if U0 is correct
U0_rec = zeros(30,30,198);
for k=1:198
    for j=1:30
        for i=1:30
            U0_rec(i,j,k) = U0(900*(k-1)+30*(j-1)+i);
        end
    end
end
U_diff = abs(Usel-U0_rec);
diffsum = sum(U_diff(:));
% end test

coor_ind_ori = 1:length(y_onerow)*length(x_onecol);
coor_ind_add = [];
for i=1:length(coor_ind_ori)
    if(mod(i,30)==0 || i>870 || mod(i-1,31)==0)
        coor_ind_add = [coor_ind_add,coor_ind_ori(i)];
        coor_ind_ori(i) = -1;
    end
end
coor_ind_from = coor_ind_ori;
coor_ind_from(coor_ind_from == -1) = [];

Smpl = randsample(coor_ind_from, floor(length(coor_ind_from)*0.5));
Smpl = [Smpl,coor_ind_add];


[Smpl,sind] = sort(Smpl,'ascend');

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

t_intoneslice = t_oneslice;%0.01:0.005:1.985;
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

M1 = mod(Smpl-1,30)+1;%x
M2 = floor((Smpl-1)/30)+1;%y
M = [M1',M2'];


U0 = [];
for t=2:199
    for i=1:size(M,1)
        U0 = [U0, Xn(M(i,1),M(i,2),t)];
    end
end

Urec = nan(30,30,198);
for i=1:198
    for j=1:length(M)
        Urec(M(j,1),M(j,2),i) = U0((i-1)*length(M)+j);
    end
end

figure
for i=1:198
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
%%
numLayers = 5;
numNeurons = 200;
sz = [numNeurons 3];
numIn = numNeurons;
%%
C = c;
ALPHA = alpha;

parameters = struct;

parameters.fc1.Weights = initializeHe(sz,3);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

rnk = 5;
rnkAlpha = 5
mu = 0;
sigma = 0.1;

parameters.UL1 = initializeGaussian([length(x_onecol),rnk],mu,sigma);%([length(x_onecol),rnk]);
parameters.SVR1 = initializeGaussian([rnk,length(x_onecol)],mu,sigma);

parameters.UL2 = initializeGaussian([length(x_onecol),rnkAlpha],mu,sigma);%([length(x_onecol),rnk]);
parameters.SVR2 = initializeGaussian([rnkAlpha,length(x_onecol)],mu,sigma);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

parameters.("fc" + numLayers).Weights = initializeHe([1,numNeurons],numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
%
%%
numEpochs = 5000;
miniBatchSize = length(X0);

executionEnvironment = "auto";

initialLearnRate = 1e-3;
decayRate = 2.5e-4;
%%
mbq = minibatchqueue(ds, ...
    MiniBatchSize=900*11, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);
%%
X0 = dlarray(X0,"CB");
Y0 = dlarray(Y0,"CB");
T0 = dlarray(T0,"CB");

ampAdj = 1/(max(U0(:))-min(U0(:)));
biasAdj = min(U0(:));
U0 = 2*(U0-biasAdj)*ampAdj-1;

U0 = dlarray(U0);

averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelLoss_2D_atten_var_wuSVD_noLossb);
%%
iteration = 0;
RECORD = cell(numEpochs,5);
LOSS = zeros(numEpochs,5);
LOSS_in = zeros(6,5);
lenT = 10; % in fact this is lenT-1

% Known entries are located in the right, bottom boundaries and the
% diagonal
addon = zeros(88,2);
for i=1:30
    addon(i,:) = i;
end
for i=31:59
    addon(i,1) = 30;
    addon(i,2) = i-30;
end
for i=60:88
    addon(i,1) = i-59;
    addon(i,2) = 30;
end

%%
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

       [loss,lossF,lossU,loss_s,loss_b,LAM1,LAM2,gradients,vU,PDEterms] = dlfeval(accfun,parameters,X,Y,T,X0,Y0,T0,U0,lenT,C,ALPHA,10,addon);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);

        RECORD{epoch,1} = LAM1;
        RECORD{epoch,2} = LAM2;
        RECORD{epoch,4} = vU;
        
        LOSS_in(ep_in,1) = double(gather(extractdata(loss)));
        lossF_in = double(gather(extractdata(lossF)));
        LOSS_in(ep_in,2) = lossF_in;
        lossU_in = double(gather(extractdata(lossU)));
        LOSS_in(ep_in,3) = lossU_in;
        loss_s_in = double(gather(extractdata(loss_s)));
        LOSS_in(ep_in,4) = loss_s_in;
        loss_b_in = double(gather(extractdata(loss_b)));
        LOSS_in(ep_in,5) = loss_b_in;
    end
    epoch
    if(mod(epoch,100)==0)
        RECORD{epoch,3} = gradients;
        RECORD{epoch,5} = PDEterms;
    end
    % Plot training progress.
    loss = double(gather(extractdata(loss)));

    LOSS(epoch,1) = loss;
    lossF = double(gather(extractdata(lossF)));
    LOSS(epoch,2) = lossF;
    lossU = double(gather(extractdata(lossU)));
    LOSS(epoch,3) = lossU;
    loss_s = double(gather(extractdata(loss_s)));
    LOSS(epoch,4) = loss_s;
    loss_b = double(gather(extractdata(loss_b)));
    LOSS(epoch,5) = loss_b;
    %[loss, lossU]
    LOSS(epoch,1:5)
    if(mod(epoch,500)==0 || epoch==5)
        fname = strcat('Saved/',strcat(num2str(epoch),'STD20_RBD_r55.mat'));
        save(fname);
    end
    toc
end