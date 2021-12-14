%Maria Leiloglou 06_10_2018
%this code calculates the SNR in the fluorescence image
clc;
clear all
close all
[filename, pathname] = uigetfile({'*.*';'*.tif';'*.png';'*.jpg';},'File Selector');
str=convertCharsToStrings(filename);
Newstr=split(str,".");
filenamenew=Newstr(1,1);
filenamenew= convertStringsToChars(filenamenew);
cd(pathname);
im=imread(filename);
windo=40;
sample=zeros(windo,windo);
waitfor(msgbox('Choose a fluorescence pixel'));
figure(1);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c1,r1,~]=impixel(im);
close figure 1
sample=im(r1:r1+windo-1,c1:c1+windo-1);
sample= double(sample);
sample=reshape(sample,1,numel(sample));
Flmean=mean(sample);
Flsd=std(sample);

sample2=zeros(windo,windo);
waitfor(msgbox('Choose an background pixel'));
figure(2);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c2,r2,~]=impixel(im);
close figure 2
sample2=im(r2:r2+windo-1,c1:c1+windo-1);
sample2= double(sample2);
sample2=reshape(sample2,1,numel(sample2));
NonFlmean=mean(sample2);
NonFlsd=std(sample2);


save(['SNR',filenamenew,'.mat'],'Flmean','Flsd','NonFlmean','NonFlsd');
