function [loss,lossF,lossU,lossS,lossB,LAM1,LAM2,gradients] = modelLoss_2D_atten(parameters,X,Y,T,X0,Y0,T0,U0,lenT,C,alpha, Weight,addon,numParams)
% % Make predictions with the initial conditions.
U = model_2D(parameters,X,Y,T,numParams);
% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,Y,T},EnableHigherDerivatives=true);
Ut = gradientsU{3};
%Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(gradientsU{1},"all"),X,EnableHigherDerivatives=true);
Uyy = dlgradient(sum(gradientsU{2},"all"),Y,EnableHigherDerivatives=true);
Utt = dlgradient(sum(Ut,"all"),T,EnableHigherDerivatives=true);

% Calculate los
LAM1 = parameters.Umat1*parameters.Vmat1;
LAM2 = parameters.Umat2*parameters.Vmat2;
f = Utt +repmat(reshape(LAM1,1,size(LAM1,1)*size(LAM1,2)),1,lenT).*(Uxx+Uyy)+repmat(reshape(LAM2,1,size(LAM2,1)*size(LAM2,2)),1,lenT).*Ut; 

zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

lossS = sum(sum(relu(LAM1)))+sum(sum(relu(-LAM2))); % loss for the sign

linearIndices = sub2ind(size(LAM1), addon(:, 1), addon(:, 2)); % Convert subscript indices to linear indices
lossB = dlarray(sum( ( LAM1(linearIndices) + C(linearIndices).^2 ) .^2) + sum( ( LAM2(linearIndices) - alpha(linearIndices) ) .^2 ) );

% Calculate lossU. 
U0Pred = model_2D(parameters,X0,Y0,T0,numParams);
lossU = mse(U0Pred, U0);

%Combine losses.
loss = Weight.lossF*lossF + Weight.lossU*lossU + Weight.lossS*lossS + Weight.lossB*lossB;
gradients = dlgradient(loss,parameters);
end
