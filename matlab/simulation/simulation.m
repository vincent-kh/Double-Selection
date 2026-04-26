% This is the program to provide the simulation figure in the appendix and
% an exmaple for the asymptotic performance

% Part I: CLT plots
% Part II: exmaple for the asymptotic performance


clear; clc;
close all;

% choose p, n an T for the plots
p = 100;
n = 300;
T = 480;

simName = strcat('_T_',num2str(T),'_n_',num2str(n),'_p_',num2str(p));
load(strcat('TS_CMA_Simu',simName))


%% Part I: CLT plots

nbin = 50; lw=1;
binCtrs = linspace(-6,6,nbin);
binWidth=binCtrs(2)-binCtrs(1);

fig11 = figure;

% double selection

subplot(3,2,1)

counts=hist(lambdag_ds_std(:,1),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('DS: Uselful','FontSize',11)

subplot(3,2,3)
counts=hist(lambdag_ds_std(:,2),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('DS: Redundant','FontSize',11)

subplot(3,2,5)
counts=hist(lambdag_ds_std(:,3),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('DS: Useless','FontSize',11)

% single selection 

subplot(3,2,2)

counts=hist(lambdag_ss_std(:,1),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('SS: Uselful','FontSize',11)

subplot(3,2,4)
counts=hist(lambdag_ss_std(:,2),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('SS: Redundant','FontSize',11)

subplot(3,2,6)
counts=hist(lambdag_ss_std(:,3),binCtrs);
prob = counts / (K * binWidth);
h2=bar(binCtrs,prob,'hist');

set(h2,'FaceColor',[.6 .6 .6]);
set(h2,'EdgeColor',[.6 .6 .6]);

xgrid = linspace(-4,4,81);
pdfReal=pdf('Normal',-4:0.1:4,0,1);
line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);
xlim([-5,5]);
ylim([0,0.5]);
title('SS: Useless','FontSize',11)

saveas(fig11,'Figure11','epsc')

%% Part II: exmaple for the asymptotic performance

% true values for lambdas
lambdag = [lambdag(1);0;0]; 

% MC bias for symbol
disp('MC bias')
disp({'useful','redundant','useless'})
disp(mean(lambdag_ds)-lambdag')
disp(mean(lambdag_ss)-lambdag')
disp(mean(lambdag_ns)-lambdag')


% MC RMSE for symbol
disp('MC RMSE')
disp({'useful','redundant','useless'})
disp(sqrt(mean((lambdag_ds-ones(K,1)*lambdag').^2)))
disp(sqrt(mean((lambdag_ss-ones(K,1)*lambdag').^2)))
disp(sqrt(mean((lambdag_ns-ones(K,1)*lambdag').^2)))



