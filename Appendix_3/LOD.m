%Maria Leiloglou 06_10_2018
%this code inputs mean and standard deviation for signal and background
%outputs figure of SNR with concentration
 
clear all;
close all;
clc;
BackgroundmeanM=zeros(11,1);%preallocation of background mean values of the 11 concentrations
BackgroundstdM=zeros(11,1);%preallocation of background standard deviation values of the 11 concentrations
SignalmeanM=zeros(11,1);%preallocation of fluorescence mean values of the 11 concentrations
SignalstdM=zeros(11,1);%preallocation of fluorescence standard deviation values of the 11 concentrations
SNR=zeros(11,1);%preallocation for the SNR of each concentration
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_3/data');%change current folder
for i=1:1:11% for the different concentrations
    load(['SNR',num2str(i),'.mat']);%load the data from a specific concentration
    BackgroundmeanM(i,1)=Bgrdmean1;%mean of background values
    BackgroundstdM(i,1)=Bgrdstd1;%standard deviation of background values
    SignalmeanM(i,1)=Flmean1;%mean of fluorescence values
    SignalstdM(i,1)=Flstd1;%standard deviation of fluorescence values
    SNR(i,1)=(SignalmeanM(i,1)-BackgroundmeanM(i,1))/BackgroundstdM(i,1);%calculate the SNR
end
 
C=[1 10 20 30 40 50 60 70 80 90 100]; %nanoMolar concentrations of the solutions
%create figure with SNR for specific lens configuration
Figure1=figure;
scatter(C,SNR,'blue');hold on;
xlabel('concentration (nM)'); ylabel('SNR');  
ylim([0 max(SNR)+1]);
%example for specific working distance
title('SNR values for a specific working distance');hold on;
hline = refline([0 3.28]);%SNR above 3.28 is acceptable
hline.Color = 'r';
legend('SNR values','SNR limit');
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_3/results');%change current folder
saveas(Figure1,'Sensitivity_measurements.fig');
saveas(Figure1,'Sensitivity_measurements.png');


for i=1:11 %for each concentration
    if SNR(i,1)<3.28 %if the SNR is below the limit
        if i>1 %only if we are not in the first concentration
           lod=C(i-1);%save the previous concentration as the limit of detection
           save('lod_in_nM.mat','lod');
           break
        else %if we are in the first concentration
           disp('none of the solutions is detectable');%means we shoud have tried lower concentration
        break
        end
    elseif SNR(i,1)>=3.28 && i==11 %if the SNR of all is above limit
        disp('all of the solutions are detectable');%means we shoud have tried higher concentration
   end
end

 



