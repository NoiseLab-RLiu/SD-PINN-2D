function [loss,lossF,lossU,lossS,lossB,LAM1,LAM2,gradients] = modelLoss_2D_attenBowl(parameters,XYT,X0Y0T0,U0,lenT,C,alpha, Weight,addon,numParams)
% % Make predictions with the initial conditions.
U = model_2D(parameters,XYT,numParams);
% Calculate derivatives with respect to X and T.
X = XYT(1,:);
Y = XYT(2,:);
T = XYT(3,:);
gradientsU = dlgradient(sum(U,"all"),{X,Y,T},EnableHigherDerivatives=true);
Ut = gradientsU{3};
%Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(gradientsU{1},"all"),X,EnableHigherDerivatives=true);
Uyy = dlgradient(sum(gradientsU{2},"all"),Y,EnableHigherDerivatives=true);
Utt = dlgradient(sum(Ut,"all"),T,EnableHigherDerivatives=true);

% Calculate los
LAM1 = parameters.Umat1*parameters.Vmat1;
LAM2 = parameters.Umat2*parameters.Vmat2;
[numR, numC] = size(LAM1);
f = Utt +repmat(reshape(LAM1,1,numR*numC),1,lenT).*(Uxx+Uyy)+repmat(reshape(LAM2,1,numR*numC),1,lenT).*Ut; 

zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

lossS = sum(sum(relu(LAM1+1)+relu(-(LAM1+4))))+sum(sum(relu(-LAM2)+relu(LAM2-5)));

linearIndices = sub2ind(size(LAM1), addon(:, 1), addon(:, 2)); % Convert subscript indices to linear indices
lossB = dlarray(sum( ( LAM1(linearIndices) + C(linearIndices).^2 ) .^2) + sum( ( LAM2(linearIndices) - alpha(linearIndices) ) .^2 ) );

% Calculate lossU. 
U0Pred = model_2D(parameters,X0Y0T0,numParams);
lossU = mse(U0Pred, U0);

%Combine losses.
loss = Weight.lossF*lossF + Weight.lossU*lossU + Weight.lossS*lossS + Weight.lossB*lossB;
gradients = dlgradient(loss,parameters);
end
