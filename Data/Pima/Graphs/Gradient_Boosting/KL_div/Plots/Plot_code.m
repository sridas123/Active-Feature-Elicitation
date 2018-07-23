color1 = [1, 0, 0];
color2 = [0 1 1];
color4 = [0, 1,0];
color3 = [0, 0, 1];

fullPath = 'D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\Recall.csv';
KRecall = csvread(fullPath);
%disp(KRecall);
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
AFE = KRecall(:,1);	
RND = KRecall(:,2);
Obs_US = KRecall(:,3);
UnObs_US = KRecall(:,4);

%set(groot,'defaultAxesColorOrder',[color1;color3;color4;'green','black']);
fig2 = figure;
graph = plot(X,AFE,'-h', X,RND,'-p', X,Obs_US, '-s', X,UnObs_US, '-v','MarkerSize',15);

%title('UW','FontSize',22);
xlabel('Iterations','FontSize',20);
ylabel('Recall','FontSize',20);
legend({'AFE+KL','RND','USObs','USAll'},'Location','north', 'Orientation', 'Horizontal', 'FontSize', 16, 'FontName', 'ArialNarrow');
set(graph,'LineWidth',3);
ylim([0.5 1]);
xlim([0,15]);

fig2.CurrentAxes.FontSize=20;
yL = get(gca,'YLim');
%fig2.PaperUnits = 'inches';
fig2.Position = [100 100 600 500];
%line([8 8],yL,'Color','k', 'LineWidth', 3);
print('D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\Recall','-dpng','-r200');



fullPath = 'D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\F1.csv';
KF1 = csvread(fullPath);
%disp(KF1);
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
AFE = KF1(:,1);	
RND = KF1(:,2);
Obs_US = KF1(:,3);
UnObs_US = KF1(:,4);

%set(groot,'defaultAxesColorOrder',[color1;color3;color4;'green','black']);
fig2 = figure;
graph = plot(X,AFE,'-h', X,RND,'-p', X,Obs_US, '-s', X,UnObs_US, '-v','MarkerSize',15);

%title('UW','FontSize',22);
xlabel('Iterations','FontSize',20);
ylabel('F1 Score','FontSize',20);
legend({'AFE+KL','RND','USObs','USAll'},'Location','north', 'Orientation', 'Horizontal', 'FontSize', 16, 'FontName', 'ArialNarrow');
set(graph,'LineWidth',3);
ylim([0.5 1]);
xlim([0,15]);

fig2.CurrentAxes.FontSize=20;
yL = get(gca,'YLim');
%fig2.PaperUnits = 'inches';
fig2.Position = [100 100 600 500];
%line([8 8],yL,'Color','k', 'LineWidth', 3);
print('D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\F1Score','-dpng','-r200');


fullPath = 'D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\gmean.csv';
Kgm = csvread(fullPath);
%disp(Kgm);
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
AFE = Kgm(:,1);	
RND = Kgm(:,2);
Obs_US = Kgm(:,3);
UnObs_US = Kgm(:,4);

%set(groot,'defaultAxesColorOrder',[color1;color3;color4;'green','black']);
fig2 = figure;
graph = plot(X,AFE,'-h', X,RND,'-p', X,Obs_US, '-s', X,UnObs_US, '-v','MarkerSize',15);

%title('UW','FontSize',22);
xlabel('Iterations','FontSize',20);
ylabel('Geometric Mean','FontSize',20);
legend({'AFE+KL','RND','USObs','USAll'},'Location','north', 'Orientation', 'Horizontal', 'FontSize', 16, 'FontName', 'ArialNarrow');
set(graph,'LineWidth',3);
ylim([0.5 1]);
xlim([0,15]);

fig2.CurrentAxes.FontSize=20;
yL = get(gca,'YLim');
%fig2.PaperUnits = 'inches';
fig2.Position = [100 100 600 500];
%line([8 8],yL,'Color','k', 'LineWidth', 3);
print('D:\Grad Studies\SRL\Active_Feature_acquisition_IJCAI_journal_exp\Data\Pima\Graphs\Gradient_Boosting\KL_div\Plots\GMean','-dpng','-r200');


% 
% free_xaxis=[0,100,200,300,400,500];
% free_pgplanner1=[0,0.702,0.708,0.716,0.72,0.726]; %All-Min F3
% free_pgplanner2=[0,0.506,0.85,0.469,0.85,0.848]; %RW F3
% free_pgplanner3=[0,0.9,0.896,0.898,0.9,0.896]; %FS F3
% free_pgplanner4=[0,0.846,0.87,0.866,0.892,0.788]; %SS F3
% free_pgplanner5=[0,0.896,0.896,0.894,0.888,0.89]; %FP F3
% free_pgplanner6=[0,0.898,0.886,0.838,0.862,0.886]; %MACCS F3
% 
% 
% 
% %set(groot,'defaultAxesColorOrder',[color1;color3;color4;'green','black']);
% fig3 = figure;
% graph = plot(free_xaxis,free_pgplanner1,'-o', free_xaxis,free_pgplanner2,'-p', free_xaxis,free_pgplanner3, '-s',free_xaxis,free_pgplanner4, '--x',free_xaxis,free_pgplanner5, '-^',free_xaxis,free_pgplanner6, '-v','MarkerSize',15);
% 
% %title('UW','FontSize',22);
% xlabel('# Drug pairs','FontSize',40);
% ylabel('F3 Score','FontSize',40);
% legend({'All-Min','RW','FS','SS','FP','MACCS'},'Location','southeast','FontSize',40);
% set(graph,'LineWidth',3);
% ylim([0 1]);
% xlim([0,500]);
% 
% fig3.CurrentAxes.FontSize=40;
% yL = get(gca,'YLim');
% fig3.PaperUnits = 'inches';
% fig3.PaperPosition = [0 0 20 15];
% %line([8 8],yL,'Color','k', 'LineWidth', 3);
% print('\\Client\C$\Users\mxd174630\Desktop\NewResults\F3Score','-dpng');
% 
