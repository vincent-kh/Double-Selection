% This Version: Jan 4, 2019.
% @copyright Guanhao Feng, Stefano Giglio and Dacheng Xiu

% We implement our program via the matlab version of glmnet 
% <http://web.stanford.edu/~hastie/glmnet_matlab/> on Matlab 2014b. 

%%
% The main function for Double-Selection implementation with cross-validation
%

function result = DS_TSCV(Ri, gt, ht, alpha, NumLambda, seednum, Kfld)

if isempty(alpha)
    alpha = 1; % default is lasso
end

if isempty(NumLambda)
    NumLambda  = 100;
end

if isempty(seednum)
    rng(100);
else
    rng(seednum);
end

lambda = exp(linspace(0,-35, NumLambda));

% dim of gt
d = size(gt,1);
sel2 = [];
sel3 = [];

% 1st and 2nd selection
for i = 1:d
    TSCVout =  TSCV(Ri, gt(i,:), ht, lambda, Kfld, 1, alpha, seednum);
    sel1 = TSCVout.sel1;
    sel2 = [sel2; TSCVout.sel2];
    sel3 = [sel3; TSCVout.sel3];
end
sel1 = unique(sel1);
sel2 = unique(sel2);
sel3 = unique(sel3);
result.cvm111 = TSCVout.cvm111;
result.cvm222 = TSCVout.cvm222;

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
select1 = unique([sel1;sel2]);
result.select = select1;
result.tune1 = TSCVout.lambda1;
result.tune2 = TSCVout.lambda2;
result.tune3 = TSCVout.lambda3;

end


%%
% The function for cross-validation over time
%

function output = TSCV(Ri, gt, ht, lambda, Kfld, Jrep,alpha,seednum)

if isempty(seednum)
    seednum = 100;
end

% data information
[p,T] = size(ht);

n = size(Ri,1);
L = length(lambda);

tmp3 = nancov([ht; Ri]');
Ch0   = tmp3((p+1):end,1:p);
tmp3b = nancov([gt; Ri]');
Cg0 = tmp3b(2:end,1);
ER0 = nanmean(Ri,2);

beta = NaN(n,p);
for i = 1:p
    beta(:,i) = Ch0(:,i)/nanvar(ht(i,:));
end
penalty = mean(beta.^2,1);
penalty = penalty./mean(penalty); % normalize the level

cvm1 = NaN(L,Kfld,Jrep);
cvm2 = NaN(L,Kfld,Jrep);
cvm3 = NaN(L,Kfld,Jrep);

cvm11 = [];
cvm22 = [];
cvm33 = [];

nomissing = (sum(isnan([ht;gt]),1)==0)';

for j = 1:Jrep
    
    rng(seednum+j)
    indices = crossvalind('Kfold',T,Kfld);
    
    for k = 1:Kfld
        
        % divide the train and test samples
        test = (indices == k);
        train = (indices ~= k);
        
        Ri_train = Ri(:,train);
        ht_train = ht(:,train);
        gt_train = gt(:,train);
        tmp1 = nancov([ht_train; Ri_train]');
        Ch_train = tmp1((p+1):end,1:p);
        tmp1b = nancov([gt_train; Ri_train]');
        Cg_train = tmp1b(2:end,1);
        ER_train = nanmean(Ri_train,2);
        
        ht_train = ht(:,train & nomissing);
        gt_train = gt(:,train & nomissing);
        
        Ri_test = Ri(:,test);
        ht_test = ht(:,test);
        gt_test = gt(:,test);
        tmp2 = nancov([ht_test; Ri_test]');
        Ch_test = tmp2((p+1):end,1:p);
        tmp2b = nancov([gt_test; Ri_test]');
        Cg_test = tmp2b(2:end,1);
        ER_test = nanmean(Ri_test,2);
        
        opts1 = struct('standardize',false,'lambda',lambda,'alpha',alpha);
        model1 = glmnet(Ch_train*(diag(penalty)), ER_train,'gaussian',opts1);
        Estimate1 = glmnetCoef(model1);
        
        opts2 = struct('standardize',false,'lambda',lambda,'alpha',alpha);
        model2 = glmnet(Ch_train*(diag(penalty)), Cg_train,'gaussian',opts2);
        Estimate2 = glmnetCoef(model2);
        
        opts3 = struct('intr',false,'standardize',true,'lambda',lambda,'alpha',alpha);
        model3 = glmnet(ht_train', gt_train','gaussian',opts3);
        
        ER_pred = [ones(n,1),Ch_test*(diag(penalty))]*Estimate1;
        
        Cg_pred = [ones(n,1),Ch_test*(diag(penalty))]*Estimate2;
        
        gt_pred = ht_test'*model3.beta;
        
        LL1 = length(model1.lambda);
        LL2 = length(model2.lambda);
        LL3 = length(model3.lambda);
        
        cvm1(1:LL1,k,j) = mean((repmat(ER_test,1,LL1) - ER_pred).^2,1)';
        cvm2(1:LL2,k,j) = mean((repmat(Cg_test,1,LL2) - Cg_pred).^2,1)';
        cvm3(1:LL3,k,j) = nanmean((repmat(gt_test',1,LL3) - gt_pred).^2,1)';
    end
    
    cvm11 = [cvm11, cvm1(:,:,j)];
    cvm22 = [cvm22, cvm2(:,:,j)];
    cvm33 = [cvm33, cvm3(:,:,j)];
end

cv_sd1 = std(cvm11',1)/sqrt(Kfld*Jrep);
cv_sd2 = std(cvm22',1)/sqrt(Kfld*Jrep);
cv_sd3 = std(cvm33',1)/sqrt(Kfld*Jrep);

cvm111 = mean(cvm11,2);
cvm222 = mean(cvm22,2);
cvm333 = mean(cvm33,2);

[~,l_sel1] = min(cvm111);
[~,l_sel2] = min(cvm222);
[~,l_sel3] = min(cvm333);

if(l_sel1 == 1)
    l1_tmp = find(cvm111 == cvm111(l_sel1), 1, 'last');
    l_sel1 = l1_tmp;
    l1_1se = l1_tmp;
else
    cvm11ub = cvm111(l_sel1) + cv_sd1(l_sel1);
    l1_1se = find(cvm111(1:l_sel1) >= cvm11ub, 1, 'last');
    if isempty(l1_1se)
        l1_1se = l_sel1;
    end
end

if(l_sel2 == 1)
    l2_tmp = find(cvm222 == cvm222(l_sel2), 1, 'last');
    l_sel2 = l2_tmp;
    l2_1se = l2_tmp;
else
    cvm22ub = cvm222(l_sel2) + cv_sd2(l_sel2);
    l2_1se = find(cvm222(1:l_sel2) >= cvm22ub, 1,'last');
    if isempty(l2_1se)
        l2_1se = l_sel2;
    end
end

if(l_sel3 == 1)
    l3_tmp = find(cvm333 == cvm333(l_sel3), 1, 'last');
    l_sel3 = l3_tmp;
    l3_1se = l3_tmp;
else
    cvm33ub = cvm333(l_sel3) + cv_sd3(l_sel3);
    l3_1se = find(cvm333(1:l_sel3) >= cvm33ub, 1,'last');
    if isempty(l3_1se)
        l3_1se = l_sel3;
    end
end

% to reestimate the model with all data

% refit the model
opts11 = struct('standardize',false,'lambda',lambda([l1_1se l_sel1]),'alpha',alpha);
model1 = glmnet(Ch0*(diag(penalty)), ER0,'gaussian',opts11);
opts22 = struct('standardize',false,'lambda',lambda([l2_1se l_sel2]),'alpha',alpha);
model2 = glmnet(Ch0*(diag(penalty)), Cg0,'gaussian',opts22);

opts33 = struct('intr',false,'standardize',true,'lambda',lambda([l3_1se l_sel3]),'alpha',alpha);
model3 = glmnet(ht(:,nomissing)', gt(nomissing)','gaussian',opts33);

sel1 = find(model1.beta(:,2) ~= 0);
sel2 = find(model2.beta(:,2) ~= 0);
sel3 = find(model3.beta(:,2) ~= 0);
output.sel1 = sel1;
output.sel2 = sel2;
output.sel3 = sel3;
output.lambda1 = lambda(l_sel1);
output.lambda2 = lambda(l_sel2);
output.lambda3 = lambda(l_sel3);

sel1_1se = find(model1.beta(:,1) ~= 0);
sel2_1se = find(model2.beta(:,1) ~= 0);
sel3_1se = find(model3.beta(:,1) ~= 0);
output.sel1_1se = sel1_1se;
output.sel2_1se = sel2_1se;
output.sel3_1se = sel3_1se;
output.lambda1_1se = lambda(l1_1se);
output.lambda2_1se = lambda(l2_1se);
output.lambda3_1se = lambda(l3_1se);

output.cvm111 = cvm111;
output.cvm222 = cvm222;

end


%%
% the function for estimation and inference
%

function output = infer(Ri, gt, ht, sel1, sel2, sel3)

n = size(Ri,1);
p = size(ht,1);
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
select = unique([sel1;sel2]);


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

temp2  =  0;
ii = 0;
for l = nomissing
    ii = ii+1;
    mt = 1-lambda_full'*[gt(:,l);ht(select,l)];
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
