% This main program provides Figure 1 in the paper

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

% load those 200 randome seeds selected by cross-validations
% For each randome seed in 1:200, we run a cross-validation and find the
% best tuning parameter.
load tune_figure1.mat

%% save the 1st selection for the 1st factor
% the 1st selections are the same for all factors

% choose control factors before 2012
ContrlList = find(year_pub < 2012);
ControlFactor = factors(:,ContrlList);

% test factors since 2012
TestList = find(year_pub >= 2012);
TestFactor = factors(:,TestList);

sel11 = zeros(200,135);

for k = 1:200
    
    disp(k)
    
    gt = TestFactor(:,1)'; % test factor
    ht = ControlFactor'; % control factor
    
    model_ds = DS(Ri', gt, ht,-log(tune_sel_150(k,1)),-log(tune_sel_150(k,2)),...
        1,seed_num);
    
    sel11(k, model_ds.sel1) = 1;
    
end

%% Figure 1 for factor selection rates

fig1 = figure;
h2 = bar(1:135, mean(sel11), 0.3);
xlim([-1,136]) % make the range larger
xlabel('Factor ID')
ylabel('Selection rate')
set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);
set(gca,'fontsize',10)

saveas(fig1,'Figure1','pdf');
