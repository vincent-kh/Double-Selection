% This program is used to import data

clear; clc;
close all;

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

port_5x5_seq = csvread('port_5x5_seq.csv',0,1);
port_5x5_seq = port_5x5_seq - rf*ones(1,size(port_5x5_seq,2));

port_3x2_seq = csvread('port_3x2_seq.csv',0,1);
port_3x2_seq = port_3x2_seq - rf*ones(1,size(port_3x2_seq,2));

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
