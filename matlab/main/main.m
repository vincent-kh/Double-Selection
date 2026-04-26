% Main results

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
port_3x2 = csvread('port_3x2.csv',0,1);
port_3x2 = port_3x2 - rf*ones(1,size(port_3x2,2)); % excess return

% other information
summary = readtable('summary.csv');
factorname = summary.Row;
factorname_full = summary.Descpription;
year_pub = summary.Year;
year_end = summary.Year_end;

port_3x2_id = readtable('port_3x2_id.csv');

mkt_ind = find(ismember(factorname,'MktRf'));
smb_ind = find(ismember(factorname,'SMB'));
hml_ind = find(ismember(factorname,'HML'));

%% form a smaller set of portfolios for bivariate sorted porfolios
kk = 10; % minimun number of stocks in a portfolio

include_3x2 = find(port_3x2_id.min_stk6>=kk)';
port_3x2b = [];

for i = 1:P
    if ismember(i,include_3x2)
        port_3x2b = [port_3x2b port_3x2(:,(i*6-5):(i*6))];
    end
end

Ri = port_3x2b; % test asset

% load tune_center for 200 randome seeds selected by cross-validations
% For each randome seed in 1:200, we run a cross-validation and find the
% best tuning parameter.
% Then we take the average for the 200 selected tuning parameters at the
% log scale as tune center, because we plot heat maps on log(lambda).
load tune_main.mat

% choose control factors before 2012
ContrlList = find(year_pub < 2012);
ControlFactor = factors(:,ContrlList);
FF3 = factors(:,[mkt_ind,smb_ind,hml_ind])';

% test factors since 2012
TestList = find(year_pub >= 2012);
TestFactor = factors(:,TestList);

result = [];

% test factor individually
for j = 1:length(TestList)

    disp(j)

    gt = TestFactor(:,j)'; % test factor
    ht = ControlFactor'; % control factor

    % use the average tuning parameter from 200 randome seeds
    model_ds = DS(Ri', gt, ht,-log(tune_center(j,1)),-log(tune_center(j,2)),1,seed_num);
    tstat_ds = model_ds.lambdag_ds/model_ds.se_ds;
    lambda_ds = model_ds.gamma_ds(1);

    % Single-Selection results, replace with a huge tune2
    model_ss = DS(Ri', gt, ht,-log(tune_center(j,1)),-log(1),1,seed_num);
    tstat_ss = model_ss.lambdag_ds/model_ss.se_ds;
    lambda_ss = model_ss.gamma_ds(1);

    % controlling everything, no selection, OLS
    model_ols = PriceRisk_OLS(Ri', gt, ht);
    tstat_ols = model_ols.lambdag_ols/model_ols.se_ols;
    lambda_ols = model_ols.lambda_ols(1);

    % only control FF3 by OLS
    model_FF3 = PriceRisk_OLS(Ri', gt, FF3);
    tstat_FF3 = model_FF3.lambdag_ols/model_FF3.se_ols;
    lambda_FF3 = model_FF3.lambda_ols(1);

    % time series average
    avg = nanmean(gt);
    tstat_avg = avg/nanstd(gt)*sqrt(sum(~isnan(gt)));

    % combine the results in a table
    temp = table(tstat_ds, lambda_ds,tstat_ss,lambda_ss,avg,tstat_avg,...
        lambda_ols,tstat_ols,lambda_FF3,tstat_FF3);

    result = [result;temp];
end

disp(factorname_full(model_ds.sel1))

% extract outputs
lambda_ds = result.lambda_ds*10000; % bp
tstat_ds = result.tstat_ds;

lambda_ss = result.lambda_ss*10000; % bp
tstat_ss = result.tstat_ss;

lambda_FF3 = result.lambda_FF3*10000; % bp
tstat_FF3 = result.tstat_FF3;

avg = result.avg*10000; % bp
tstat_avg = result.tstat_avg;

lambda_ols = result.lambda_ols*10000; % bp
tstat_ols = result.tstat_ols;

% factor names for those factors since 2012
factornames = factorname_full(TestList);

result = table(TestList,factornames,lambda_ds,tstat_ds,lambda_ss,tstat_ss,lambda_FF3,...
    tstat_FF3,lambda_ols,tstat_ols,avg,tstat_avg);

% display the table
disp(result)

% output Table as a CSV file
writetable(result, 'main.csv')
