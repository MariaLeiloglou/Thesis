%Maria Leiloglou 06_10_2018
%this code inputs mean and standard deviation for signal and background
%outputs figure of SNR with concentration
 
clear all;
close all;
clc;
Backgroundmean=zeros(11,1);
Backgroundstd=zeros(11,1);
Signalmean=zeros(11,1);
Signalstd=zeros(11,1);
SNR=zeros(11,1);%SNR for each concentration
for i=1:1:11
    load(['SNR',num2str(i),'.mat']);%load the data from a specific concentration
    Backgroundmean(i,1)=NonFlmean;
    Backgroundstd(i,1)=NonFlsd;
    Signalmean(i,1)=Flmean;
    Signalstd(i,1)=Flsd;
    %calculate the SNR
    SNR(i,1)=(Signalmean(i,1)-Backgroundmean(i,1))/Backgroundstd(i,1);
end
 
C=[1 10 20 30 40 50 60 70 80 90 100]; %nanoMolar 
%create figure with SNR for specific lens configuration
figure(1);
scatter(C,SNR,'blue');hold on;
xlabel('concentration (nM)'); ylabel('SNR');  
ylim([0 max(SNR)+1]);
%example for working distance of 55.8 cm
%focal length of 70 mm and f/2.8
title('f28fl70wd558');hold on;
hline = refline([0 3.28]);%SNR above 3.28 is acceptable
hline.Color = 'r';
 



