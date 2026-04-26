% This Version: Jan 4, 2019. 
% @copyright Guanhao Feng, Stefano Giglio and Dacheng Xiu

% We implement our program via the matlab version of glmnet 
% <http://web.stanford.edu/~hastie/glmnet_matlab/> on Matlab 2014b. 

%%
% The main function for Double-Selection implementation
% no cross-validation for 1st and 2nd selections
%
function result = DS(Ri, gt, ht, tune1, tune2, alpha, seednum)

% dim of g is 1!

if isempty(alpha)
    alpha = 1; % default is lasso
end

% data information
n = size(Ri,1);
p = size(ht,1);
d = size(gt,1);

tmp1  = nancov([gt',Ri']);
cov_g = tmp1((d+1):end,1:d);
tmp2  = nancov([ht',Ri']);
cov_h = tmp2((p+1):end,1:p);

ER    = nanmean(Ri,2);

beta = NaN(n,p);
for i = 1:p
  beta(:,i) = cov_h(:,i)/nanvar(ht(i,:));
end
penalty = mean(beta.^2,1);
penalty = penalty./mean(penalty); % normalize the level

lambda0 = exp(linspace(0,-35,100));

% 1st selection in cross-sectional regression
opts1 = struct('standardize',false,'lambda',exp(-tune1),'alpha',alpha);
model1 = glmnet(cov_h*(diag(penalty)), ER,'gaussian',opts1);
model1_est = glmnetCoef(model1);
sel1 = find(model1_est(2:(p+1))~=0)';
err1 = mean((ER - [ones(n,1),cov_h*(diag(penalty))]*model1_est).^2);

% 2nd selection
sel2 = [];
err2 = NaN(d,1);
opts2 = struct('standardize',false,'lambda',exp(-tune2),'alhpa',alpha);
for i = 1:d
    model2 = glmnet(cov_h*(diag(penalty)),cov_g(:,i),'gaussian',opts2);
    model2_est = glmnetCoef(model2);
    sel2 = [sel2; find(model2_est(2:end)~=0)];
    err2(i) = mean((cov_g(:,i) - [ones(n,1),cov_h*(diag(penalty))]*model2_est).^2);
end
sel2 = unique(sel2)';

% 3rd selection for avar zt
sel3 = [];
for i = 1:d
    TSCVout =  TSCV(Ri, gt(i,:), ht, lambda0, 10, 1, alpha, seednum);
    sel3 = [sel3; TSCVout.sel3_1se];
end
sel3 = unique(sel3);

% post-selection estimation and inference
dsout = infer(Ri, gt, ht, sel1, sel2, sel3);
ssout = infer(Ri, gt, ht, sel1, [], sel3);

% output for Double Selection
result.lambdag_ds = dsout.lambdag;
result.se_ds = dsout.se;
result.gamma_ds = dsout.gamma;

% output for Single Selection
result.lambdag_ss = ssout.lambdag;
result.se_ss = ssout.se;
result.gamma_ss = ssout.gamma;

% % selection results
result.sel1 = sel1;
result.sel2 = sel2;
result.sel3 = sel3;
select1 = union(sel1,sel2);
result.select = select1;
result.err1 = err1;
result.err2 = err2;

end


%%
% the function for estimation and inference
%
function output = infer(Ri, gt, ht, sel1, sel2, sel3)

n = size(Ri,1);
p     = size(ht,1);
d = size(gt,1);

tmp1  = nancov([gt',Ri']);
cov_g = tmp1((d+1):end,1:d);
tmp2  = nancov([ht',Ri']);
cov_h = tmp2((p+1):end,1:p);

ER    = mean(Ri,2);
%
M0 = eye(n) - ones(n,1)*inv(ones(n,1)'*ones(n,1))*ones(n,1)';

nomissing = find(sum(isnan([ht;gt]),1)==0);
Lnm = length(nomissing);
select = union(sel1,sel2);


X = [cov_g, cov_h(:,select)];
lambda_full = inv(X'*M0*X)*(X'*M0*ER);
lambdag = lambda_full(1:d);
clear X

% For double selection inference: AVAR
zthat = NaN(d,Lnm);
for i = 1:d
    M_mdl = eye(Lnm) - ht(sel3,nomissing)'*inv(ht(sel3,nomissing)*ht(sel3,nomissing)')*ht(sel3,nomissing);
    zthat(i,:) = M_mdl*gt(i,nomissing)';
    clear M_mdl
end
Sigmazhat = zthat*zthat'/Lnm;

temp2  =  zeros(d,d);
ii = 0;
for l = nomissing
    ii = ii+1;
    mt = 1-lambda_full'*[gt(1:d,l);ht(select,l)];
    temp2 = temp2 + mt^2*(inv(Sigmazhat)*zthat(:,ii)*zthat(:,ii)'*inv(Sigmazhat));
end

avar_lambdag = diag(temp2)/Lnm;
se = sqrt(avar_lambdag/Lnm);
clear temp2

% scaled lambda for DS
vt = [gt(:,nomissing);ht(select,nomissing)];
V_bar = vt - mean(vt,2)*ones(1,Lnm);
var_v = V_bar*V_bar'/Lnm;
gamma = diag(var_v).*lambda_full;
clear X vt V_bar var_v lambda_full

output.lambdag = lambdag;
output.se = se;
output.gamma = gamma;

end


%%
% The function for cross-validation over time
% only for 3rd selection in the DS function
%

function output = TSCV(Ri, gt, ht, lambda, Kfld, Jrep,alpha,seednum)

if isempty(seednum)
    seednum = 101;
end

% data information
[p,T] = size(ht);

L = length(lambda);

cvm3 = NaN(L,Kfld,Jrep);
cvm33 = [];

nomissing = (sum(isnan([ht;gt]),1)==0)';

for j = 1:Jrep

    rng(seednum+j)
    indices = crossvalind('Kfold',T,Kfld);

    for k = 1:Kfld

        % divide the train and test samples
        test = (indices == k);
        train = (indices ~= k);

        ht_train = ht(:,train & nomissing);
        gt_train = gt(:,train & nomissing);

        ht_test = ht(:,test);
        gt_test = gt(:,test);

        opts3 = struct('intr',false,'standardize',true,'lambda',lambda,'alpha',alpha);
        model3 = glmnet(ht_train', gt_train','gaussian',opts3);

        gt_pred = ht_test'*model3.beta;

        LL3 = length(model3.lambda);

        cvm3(1:LL3,k,j) = nanmean((repmat(gt_test',1,LL3) - gt_pred).^2,1)';
    end

    cvm33 = [cvm33, cvm3(:,:,j)];
end

cv_sd3 = std(cvm33')/sqrt(Kfld*Jrep);
cvm333 = mean(cvm33,2);
[~,l_sel3] = min(cvm333);

cvm33ub = cvm333(l_sel3) + cv_sd3(l_sel3);
l3_1se = find(cvm333(1:l_sel3) >= cvm33ub, 1,'last');
if isempty(l3_1se)
    l3_1se = l_sel3;
end

% to reestimate the model with all data
% refit the model

opts33 = struct('intr',false,'standardize',true,'lambda',lambda([l3_1se l_sel3]),'alpha',alpha);
model3 = glmnet(ht(:,nomissing)', gt(nomissing)','gaussian',opts33);

sel3 = find(model3.beta(:,2) ~= 0);
output.sel3 = sel3;
output.lambda3 = lambda(l_sel3);


sel3_1se = find(model3.beta(:,1) ~= 0);
output.sel3_1se = sel3_1se;
output.lambda3_1se = lambda(l3_1se);

end
