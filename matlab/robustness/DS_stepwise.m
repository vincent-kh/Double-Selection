% This Version: Jan 4, 2019. 
% @copyright Guanhao Feng, Stefano Giglio and Dacheng Xiu


%%
% The main function for Double-Selection implementation
% this is the forward stepwise version
%
function result = DS_stepwise(Ri, gt, ht, start)

% data information
n = size(Ri,1);
p = size(ht,1);
d = size(gt,1);

% determine the starting model
kk = length(start);
term = zeros(kk+1, p);
for i = 1:kk
    term(i+1,start(i)) = 1;
end

tmp1  = nancov([gt',Ri']);
cov_g = tmp1((d+1):end,1:d);
tmp2  = nancov([ht',Ri']);
cov_h = tmp2((p+1):end,1:p);

ER    = nanmean(Ri,2);

% create variable names for later use
var_name = cell(p,1);
for i = 1:p
    var_name{i} = ['x',num2str(i)];
end

% 1st selection in cross-sectional regression
model1 = stepwiselm(cov_h,ER,term,'Upper','linear','Verbose',0,...
    'Criterion','BIC');
sel1 = find(ismember(var_name,model1.CoefficientNames));
%sel1 = unique([sel1;start']);

% 2nd selection
sel2 = [];
for i = 1:d
    model2 = stepwiselm(cov_h,cov_g(:,i),term,'Upper','linear','Verbose',0,...
    'Criterion','BIC');
    sel2 = [sel2; find(ismember(var_name,model2.CoefficientNames))];
end
sel2 = unique(sel2)';

% 3rd selection for avar zt
sel3 = [];
for i = 1:d
    model3 = stepwiselm(ht',gt',term,'Upper','linear','Verbose',0,...
    'Criterion','BIC');
    sel3 = [sel3; find(ismember(var_name,model3.CoefficientNames))];
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

