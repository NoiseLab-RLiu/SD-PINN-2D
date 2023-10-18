function [loss,lossF,lossU,lossS,lossB,LAM1,gradients, vU, PDEterms] = modelLoss_2D_no_atten(parameters,X,Y, T,X0, Y0, T0,U0,lenT,C,Weight,addon,numParams)
% % Make predictions with the initial conditions.
U = model_2D(parameters,X,Y,T,numParams);
vU = var(U(:));


% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,Y,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Uy = gradientsU{2};
Ut = gradientsU{3};

%Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
Uyy = dlgradient(sum(Uy,"all"),Y,EnableHigherDerivatives=true);
Utt = dlgradient(sum(Ut,"all"),T,EnableHigherDerivatives=true);

PDEterms = cell(6,1);
PDEterms{1} = Ux;
PDEterms{2} = Uy;
PDEterms{3} = Ut;
PDEterms{4} = Uxx;
PDEterms{5} = Uyy;
PDEterms{6} = Utt;

LAM1 = parameters.UL1*parameters.SVR1;
f = Utt +repmat(reshape(LAM1,1,size(LAM1,1)*size(LAM1,2)),1,lenT).*(Uxx+Uyy);

zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

lossS = 0; % loss for the sign

for i=1:size(LAM1,1)
    for j=1:size(LAM1,2)
        lossS = lossS + 1*relu(LAM1(i,j));
    end
end

lossB = dlarray(0);
if(~isempty(addon))
    for i=1:size(addon,1)
        coor = addon(i,:);
        lossB = lossB+mse(LAM1(coor(1),coor(2)), -C(coor(1),coor(2))^2,'DataFormat','CB');
    end
end
% Compute lossU
U0Pred = model_2D(parameters,X0,Y0,T0,numParams);
lossU = mse(U0Pred, U0);

%Combine losses.
loss = Weight.lossF*lossF + Weight.lossU*lossU + Weight.lossS*lossS + Weight.lossB*lossB;
gradients = dlgradient(loss,parameters);

end
