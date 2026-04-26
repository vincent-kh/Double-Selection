% This Version: Sep 4, 2017. @copyright Guanhao Feng, Stefano Giglio and Dacheng Xiu

% this is the OLS estimation function for price risk

% This code uses the panel of portfolio returns rt to estimate risk price
% for the factors gt by controling ht
% using the Feng, Giglio and Xiu (2017) approach.

% Input:
% Ri, nxT stock excess returns
% gt, dxT testing factors
% ht, pxT control factors

% output:
% output for the No Selection OLS Estimation
% lambdag_ols, se_ols, and lambda_ols

function result = PriceRisk_OLS3(Ri, gt, ht)

% data information
[n,T] = size(Ri);
d = size(gt,1);
p = size(ht,1);

cov_h = NaN(n,p);
for nn = 1:n
    temp = nancov([Ri(nn,:)' ht']);
    cov_h(nn,:) = temp(1,2:end);
end
cov_g = NaN(n,d);
for nn = 1:n
    temp = nancov([Ri(nn,:)' gt']);
    cov_g(nn,:) = temp(1,2:end);
end

ER = nanmean(Ri,2);

% for later use
M0 = eye(n) - ones(n,1)*inv(ones(n,1)'*ones(n,1))*ones(n,1)';

%% For no selection OLS
% no selection estimate
X = [cov_g, cov_h];
X_zero = [ones(n,1),cov_g, cov_h];
lambda_full_zero = inv(X_zero'*X_zero)*(X_zero'*ER);
lambda_full = inv(X'*M0*X)*(X'*M0*ER);
lambdag_ols = lambda_full(1:d);
clear X

nomissing = find(sum(isnan([ht;gt]),1)==0);
Lnm = length(nomissing);

% calculate avar_ols
zthat3 = NaN(d,Lnm);
for i = 1:d
    M_mdl = eye(Lnm) - ht(:,nomissing)'*inv(ht(:,nomissing)*ht(:,nomissing)')*ht(:,nomissing);
    zthat3(i,:) = M_mdl*gt(i,nomissing)';
    clear M_mdl
end
Sigmazhat3 = zthat3*zthat3'/Lnm;

vt = [gt(:,nomissing);ht(:,nomissing)];
temp3  = zeros(d,d);
i = 0;
for l = nomissing
    i = i+1;
    mt = 1-lambda_full'*[gt(:,l);ht(:,l)];
    temp3 = temp3 + mt^2*(inv(Sigmazhat3)*zthat3(:,i)*zthat3(:,i)'*inv(Sigmazhat3));
end

avar_lambdag3 = diag(temp3)/Lnm;
se3 = sqrt(avar_lambdag3/Lnm);
clear temp3

% scaled lambda
V_bar = vt - mean(vt,2)*ones(1,Lnm);
var_v = V_bar*V_bar'/Lnm;
lambda_ols = diag(var_v).*lambda_full;
%clear X vt V_bar var_v lambda_full

%% output

% output for OLS, No Selection
result.lambdag_ols = lambdag_ols;
result.se_ols = se3;
result.lambda_ols = lambda_ols;
result.lambda_ols_zero = lambda_full_zero;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new avar estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
avar = zeros(p+d,p+d);
meanBt = zeros(p+d,1);

sigmavhat = var_v;
vtbar = V_bar;
Gammahat = sigmavhat*lambda_full;

for t=1:Lnm

    Bt   = inv(sigmavhat)*vtbar(:,t)*vtbar(:,t)'*inv(sigmavhat)*Gammahat;
    At   = inv(sigmavhat)*vtbar(:,t)*vtbar(:,t)'*inv(sigmavhat)*(Gammahat'*inv(sigmavhat)*vtbar(:,t));
    avar =  avar - 2*At/Lnm + Bt*Bt'/Lnm;
    meanBt = meanBt + Bt/Lnm;
end

avarhatLFM = diag(inv(sigmavhat) + avar - (meanBt)*meanBt')/(Lnm);%diag(inv(sigmavhat))/(T);%

result.se_new = sqrt(avarhatLFM);

end
