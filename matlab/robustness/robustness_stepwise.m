% This main program provides stepwise robustness results 
% pretty slow

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
port_3x2 = port_3x2 - rf*ones(1,size(port_3x2,2));

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
umd_ind = find(ismember(factorname,'UMD'));
FF4 = [mkt_ind,smb_ind,hml_ind,umd_ind];

%% form a smaller set of portfolios for bivariate sorted porfolios
kk = 10; % minimun number of stocks in a portfolio

include_3x2 = find(port_3x2_id.min_stk6>=kk)';
port_3x2b = [];

for i = 1:P
    if ismember(i,include_3x2)
        port_3x2b = [port_3x2b port_3x2(:,(i*6-5):(i*6))];
    end
end

% choose control factors before 2012
ControlFactor = factors(:,year_pub < 2012);

% test factors since 2012
TestList = find(year_pub >= 2012);
TestFactor = factors(:,TestList);

lambda_fs = NaN(length(TestList),1);
tstat_fs = NaN(length(TestList),1);

% test factor individually
parfor j = 1:length(TestList)
    
    disp(j)
    
    gt = TestFactor(:,j)'; % test factor
    ht = ControlFactor'; % control factor
    
    % forward stepwise (this one is pretty slow)
    model_fs = DS_stepwise(port_3x2b', gt, ht, FF4);
    tstat_fs(j) = model_fs.lambdag_ds/model_fs.se_ds;
    lambda_fs(j) = model_fs.gamma_ds(1);

end

% bp
lambda_fs = lambda_fs*10000; 

% factor names for those factors since 2012
factornames = factorname_full(TestList);
result2 = table(TestList,factornames,lambda_fs,tstat_fs);

disp(result2)

% output Table as a CSV file
writetable(result2, 'robustness_stepwise.csv')
