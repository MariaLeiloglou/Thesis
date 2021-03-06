% Maria Leiloglou 19/02/2020
% comparison of different filters

clc
clear all
close all


% two times for the two methods
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_1/data');


promptk='How many trials/images for each method (up to 15)?';%here we have 5 images
h=input(promptk); 
S=zeros(2,h);%preallocation for fluorescence signal
B=zeros(2,h);%preallocation for background signal
Flsd=zeros(2,h);%preallocation for fluorescence standard deviation
sd=zeros(2,h);%preallocation for background standard deviation
trial=[11 12 13 14 15 16 17 18 19 110 111 112 113 114 115; 21 22 23 24 25 26 27 28 29 210 211 212 213 214 215];% row: system/method, column:trial
trial=trial(:,1:h);%here it is 2 raws for two filtrations and h columns for the trials
for i=1:2
    prompth=['the method tested is the number',num2str(i),'with this image'];
    prompth
    for j=1:h
        prompty=['the image tested is the number',num2str(j),'for this method'];
        prompty
        im=imread([num2str(trial(i,j)),'.tif']); %[think this should be i and not h]
        windo=40;
        waitfor(msgbox('Choose a Fluoresence Pixel'));
        figure(1);
        set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
        [c1,r1,~]=impixel(im);
        close figure 1
        waitfor(msgbox('Choose a Background Pixel'));
        figure(2);
        set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
        [c2,r2,~]=impixel(im);
        close figure 2
        FluorT=im(r1:r1+windo-1,c1:c1+windo-1);%fluorescence sample
        NonFluorT=im(r2:r2+windo-1,c2:c2+windo-1);%background sample
        FluorT= double(FluorT);
        NonFluorT= double(NonFluorT);
        FluorT=reshape(FluorT,1,numel(FluorT));
        NonFluorT=reshape(NonFluorT,1,numel(NonFluorT));
        S(i,j)=mean(FluorT);%mean from fluorescence signal sample
        B(i,j)=mean(NonFluorT);%mean from background signal sample
        Flsd(i,j)=std(FluorT);%standard deviation from fluorescence signal sample
        sd(i,j)=std(NonFluorT);%standard deviation from background signal sample
    end
end
difference=S-B;
SNRs=difference./sd;

%find mean SNR for each method
SNR1=SNRs(1,:);%first method
SNR1mean=mean(SNR1);
SD1=std(SNR1);
SNR2=SNRs(2,:);%second system
SNR2mean=mean(SNR2);
SD2=std(SNR2);


% compare standard deviations
n1=length(SNR1);%can use h
n2=length(SNR2);
% Fcritical table at 95 % confidence level, two-tailed
Fcrit=[161.45 199.50 215.71 224.58 230.16 233.99 236.77 238.88 240.54 241.88;
18.51 19.00 19.16 19.25 19.30 19.33 19.35 19.37 19.39 19.40;
10.13 9.55 9.28 9.12 9.01 8.94 8.89 8.85 8.81 8.79;
7.71 6.94 6.59 6.39 6.26 6.16 6.09 6.04 6.00 5.96;
6.61 5.79 5.41 5.19 5.05 4.95 4.88 4.82 4.77 4.74;
5.99 5.14 4.76 4.53 4.39 4.28 4.21 4.15 4.10 4.06;
5.59 4.74 4.35 4.12 3.97 3.87 3.79 3.73 3.68 3.64;
5.32 4.46 4.07 3.84 3.69 3.58 3.50 3.44 3.39 3.35;
5.12 4.26 3.86 3.63 3.48 3.37 3.29 3.23 3.18 3.14;
4.97 4.10 3.71 3.48 3.33 3.22 3.14 3.07 3.02 2.98;
4.84 3.98 3.59 3.36 3.20 3.10 3.01 2.95 2.90 2.85;
4.75 3.89 3.49 3.26 3.11 3.00 2.91 2.85 2.80 2.75;
4.67 3.81 3.41 3.18 3.03 2.92 2.83 2.77 2.71 2.67;
4.60 3.74 3.34 3.11 2.96 2.85 2.76 2.70 2.65 2.60;
4.54 3.68 3.29 3.06 2.90 2.79 2.71 2.64 2.59 2.54];

Fcrit=Fcrit(n1,n2); 
Z=['The Fcrit is ',num2str(Fcrit)];
disp(Z);

% Test for homogeneity, i.e. are the variances different? We calculate the experimental value of F
if SD1>=SD2
    Ftest= ((SD1)^2)/((SD2)^2); %for n1=6,n2=6 fcrit=4.28 95% confidence if Fexp < Fcrit there is no significant difference in variances
else
    Ftest= ((SD2)^2)/((SD1)^2);
end

if Ftest>=Fcrit
    Y=['The Ftest is',num2str(Ftest),'and there is a difference in variances at 0.05 significance level (one-tailed)'];
    disp(Y);
    ttest=(SNR1mean-SNR2mean)/(sqrt((((SD1)^2)/n1)+(((SD2)^2)/n2)));
    df=((((((SD1)^2)/n1)+(((SD2)^2)/n2))^(2))/((((((SD1)^2)/n1)^2)/(n1-1))+(((((SD2)^2)/n2)^2)/(n2-1))));
    df=round(df);
    %tcrit for 95% confidence level, one tailed or 90% confidence level for two tailed
    tcrit=[6.314 2.920 2.353 2.132 2.015 1.943 1.894 1.860 1.833 1.812 1.796 1.782 1.771 1.761 1.753]; 
    tcrit=tcrit(:,df);
    P=['The t-test is ',num2str(ttest)];
    disp(P);
    O=['The t-crit is ',num2str(tcrit)];
    disp(O);
    if ttest>=tcrit
        disp('and there is a significant difference between the methods at 10% significance level (one-tailed)');
        else
            disp('and there is not a significant difference between the methods at 10% significance level (one-tailed)');
    end 



else 
    Y=['The Ftest is',num2str(Ftest),'and there is not a difference in variances at 0.05 significance level'];
    disp(Y);
    spooled=sqrt(((((SD1)^2)*(n1-1))+(((SD2)^2)*(n2-1)))/(n1+n2-2));
    ttest=(SNR1mean-SNR2mean)/(spooled* sqrt((1/n1)+(1/n2)));
    tcrit=[6.314 2.920 2.353 2.132 2.015 1.943 1.894 1.860 1.833 1.812 1.796 1.782 1.771 1.761 1.753]; 
    df=n1+n2-2;
    tcrit=tcrit(:,df);
    P=['The ttest is ',num2str(ttest)];
    disp(P);
    O=['The tcrit is ',num2str(tcrit)];
    disp(O);
    if ttest>=tcrit
        disp('there is a significant difference between the methods at 10% significance level (one-tailed)');
    else
        disp('and there is not a significant difference between the methods at 10% significance level (one-tailed)');
    end 
end
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_1/results');
save('meat_PBS_stock_425_625vs500_625.mat','Fcrit','Ftest','tcrit','ttest','SNR1mean','SD1','SNR2mean','SD2');