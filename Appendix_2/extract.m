%Maria Leiloglou 06_10_2018
%this code calculates the mean and standard 
%deviation of fluorescence and background samples in the image
clc;
clear all
close all
[filename, pathname] = uigetfile({'*.*';'*.tif';'*.png';'*.jpg';},'File Selector');%choose image to load
str=convertCharsToStrings(filename);%keep the name of th eimage for use
Newstr=split(str,".");
filenamenew=Newstr(1,1);
filenamenew= convertStringsToChars(filenamenew);
cd(pathname);
im=imread(filename);%read image
windo=40;%preallocate sample size
sample=zeros(windo,windo);%preallocate sample matrix
waitfor(msgbox('Choose a fluorescence pixel'));
figure(1);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);%choose the corner from where the sample will be extracted
[c1,r1,~]=impixel(im);
close figure 1
sample=im(r1:r1+windo-1,c1:c1+windo-1);%this is the sample with fluorescence pixel values
sample= double(sample);
sample=reshape(sample,1,numel(sample));
Flmean1=mean(sample);%mean of fluorescence pixel values in the sample
Flstd1=std(sample);%standard deviation of fluorescence pixel values in the sample

sample2=zeros(windo,windo);
waitfor(msgbox('Choose an background pixel'));
figure(2);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);%choose the corner from where the sample will be extracted
[c2,r2,~]=impixel(im);
close figure 2
sample2=im(r2:r2+windo-1,c1:c1+windo-1);%this is the sample with background pixel values
sample2= double(sample2);
sample2=reshape(sample2,1,numel(sample2));
Bgrdmean1=mean(sample2);%mean of background pixel values in the sample
Bgrdstd1=std(sample2);%standard deviation of background pixel values in the sample


save(['SNR_',filenamenew,'.mat'],'Flmean1','Flstd1','Bgrdmean1','Bgrdstd1');
