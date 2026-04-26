% Robustness

clear; clc;
close all;

addpath('../glmnet_matlab')
addpath('../functions')
addpath('../data')

% fix the random seed for any additional CV
seed_num = 100;

% factor
allfactors = csvread('factors.csv',1,0);
date = allfactors(:,1);
rf = allfactors(:,2);
factors = allfactors(:,3:end);

L = length(date);
P = size(factors,2);

% test portfolios
port_5x5 = csvread('port_5x5.csv',0,1);
port_5x5 = port_5x5 - rf*ones(1,size(port_5x5,2)); % excess return

port_3x2 = csvread('port_3x2.csv',0,1);
port_3x2 = port_3x2 - rf*ones(1,size(port_3x2,2));

port_202 = csvread('port202.csv',0,1)/100;
port_202 = port_202 - rf*ones(1,size(port_202,2));

% other information
summary = readtable('summary.csv');
factorname = summary.Row;
factorname_full = summary.Descpription;
year_pub = summary.Year;
year_end = summary.Year_end;

port_5x5_id = readtable('port_5x5_id.csv');
port_3x2_id = readtable('port_3x2_id.csv');

mkt_ind = find(ismember(factorname,'MktRf'));
smb_ind = find(ismember(factorname,'SMB'));
hml_ind = find(ismember(factorname,'HML'));
mom_ind = find(ismember(factorname,'UMD'));

%% form a smaller set of portfolios for bivariate sorted porfolios
kk = 10; % minimun number of stocks in a portfolio

include_3x2 = find(port_3x2_id.min_stk6>=kk)';
port_3x2b = [];

for i = 1:P
    if ismember(i,include_3x2)
        port_3x2b = [port_3x2b port_3x2(:,(i*6-5):(i*6))];
    end
end

include_5x5 = find(port_5x5_id.min_stk>=kk)';
port_5x5b = [];

for i = 1:P
    if ismember(i,include_5x5)
        port_5x5b = [port_5x5b port_5x5(:,(i*25-24):(i*25))];
    end
end

% load tune_center for 200 randome seeds selected by cross-validations
% For each randome seed in 1:200, we run a cross-validation and find the
% best tuning parameter.
% Then we take the average for the 200 selected tuning parameters at the
% log scale as tune center, because we plot heat maps on log(lambda).
load tune_robustness.mat

%% Let's run results

% choose control factors before 2012
ControlFactor = factors(:,year_pub < 2012);

% test factors since 2012
TestList = find(year_pub >= 2012);
TestFactor = factors(:,TestList);

result = [];

% test factor individually
for j = 1:length(TestList)
    
    disp(j)
    
    gt = TestFactor(:,j)'; % test factor
    ht = ControlFactor'; % control factor
    
    % use the average tuning parameter
    model_ds = DS(port_3x2b', gt, ht,-log(tune_center(j,1)),-log(tune_center(j,2)),...
        1,seed_num);
    tstat_ds = model_ds.lambdag_ds/model_ds.se_ds;
    lambda_ds = model_ds.gamma_ds(1);
    
    % use the average tuning parameter for PCA estimation
    ht_pca = ht'*pca(ht');
    ht_pca = ht_pca/diag(nanstd(ht_pca));
    model_pca = DS(port_3x2b', gt, ht_pca',-log(tune_center_pca(j,1)),...
        -log(tune_center_pca(j,2)),1,seed_num);
    tstat_pca = model_pca.lambdag_ds/model_pca.se_ds;
    lambda_pca = model_pca.gamma_ds(1);
    
    % use the average tuning parameter for portfolio 202
    model_202 = DS(port_202', gt, ht,-log(tune_center_202(j,1)),...
        -log(tune_center_202(j,2)),1,seed_num);
    tstat_202 = model_202.lambdag_ds/model_202.se_ds;
    lambda_202 = model_202.gamma_ds(1);
    
    % use the average tuning parameter for port 5x5
    model_ds25 = DS(port_5x5b', gt, ht,-log(tune_center_25(j,1)),...
        -log(tune_center_25(j,2)),1,seed_num);
    tstat_ds25 = model_ds25.lambdag_ds/model_ds25.se_ds;
    lambda_ds25 = model_ds25.gamma_ds(1);
    
    % use the average tuning parameter for elastic net selection
    % alpha = 0.5 weight between lasso and ridge
    alpha = 0.5;
    model_glmnet = DS(port_3x2b', gt, ht,-log(tune_center_enet(j,1)),...
        -log(tune_center_enet(j,2)),alpha,seed_num);
    tstat_glmnet = model_glmnet.lambdag_ds/model_glmnet.se_ds;
    lambda_glmnet = model_glmnet.gamma_ds(1);
    
    % combime the results in a table
    temp = table(lambda_ds,tstat_ds,lambda_glmnet,tstat_glmnet,lambda_ds25,tstat_ds25,...
        lambda_202,tstat_202,lambda_pca,tstat_pca);
    
    result = [result;temp];
end

% extract outputs for Table 3
lambda_ds = result.lambda_ds*10000; % bp
tstat_ds = result.tstat_ds;

lambda_ds25 = result.lambda_ds25*10000; % bp
tstat_ds25 = result.tstat_ds25;

lambda_pca = result.lambda_pca*10000; % bp
tstat_pca = result.tstat_pca;

lambda_glmnet = result.lambda_glmnet*10000; % bp
tstat_glmnet = result.tstat_glmnet;

lambda_202 = result.lambda_202*10000; % bp
tstat_202 = result.tstat_202;

% factor names for those factors since 2012
factornames = factorname_full(TestList);

result2 = table(TestList,factornames,lambda_ds,tstat_ds,...
    lambda_ds25,tstat_ds25,lambda_202,tstat_202,...
    lambda_glmnet,tstat_glmnet,lambda_pca,tstat_pca);
% display the table
disp(result2)

% output robustness table as a CSV file
writetable(result2, 'robustness.csv')
