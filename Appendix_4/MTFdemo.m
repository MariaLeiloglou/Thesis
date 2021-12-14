%Maria Leiloglou 02_10_2018
%this demo opens a single USAF 1951 image
%the user draws a line along the chosen group of elements (2 times) & inputs the number of chosen group
%the demo opens the intensity profile for the chosen line
%the user collects the maximum and minimum data points for each one of the elements (which corresponds to a different spatial frequency [lp/mm])
%the demo opens the plot of contrast vs spatial frequency and 
%gives the resolution in the image for a contrast threshold defined by the Rayleigh criterion

clc
clear all
close all
waitfor(msgbox('Choose the USAF image'));
[filename, pathname] = uigetfile({'*.*';'*.tif';'*.png';'*.jpg';},'File Selector');
cd(pathname);im=imread(filename);imshow(im,[]);
%input the number of group you are working on
promptgroup='What is the number of group you have chosen?';
group=input(promptgroup);
waitfor(msgbox('Draw a line across the elements of group'));
improfile(5000);% select the group of elements

Cmatrix=zeros(1,5);%this is for 2nd group (5 elements)
spatialfrequency=zeros(1,5);
for i=1:5 % i for element of the chosen group
    sf=2^(group+(((i)-1)/6));%resolutionn in lp/mm
    spatialfrequency(1,i)=sf;
    waitfor(msgbox(['click on maximum and minimum values of the element number ', num2str(i),' then press enter']));
    [x,y]=ginput;% select the element, x is location y is intesity value
    Imax=max(y);
    Imin=min(y);
    C=((Imax-Imin)/(Imax+Imin))*(100);%contrast in %
    Cmatrix(1,i)=C;
    Climit=26.4;
    if C<=Climit
        Y=['This spatial frequency is:',num2str(sf),'[lp/mm]'];
        Z=['The contrast is',num2str(C),'[%]'];
        disp(Y);disp(Z);

    else 
        Y=['This spatial frequency is:',num2str(sf),'[lp/mm]'];
        Z=['The contrast is',num2str(C),'[%]'];
        disp(Y);disp(Z);
    end
end

%Create the modulation transfer funnction
figure;scatter(spatialfrequency,Cmatrix,'blue');
xlim([min(spatialfrequency) max(spatialfrequency)]);

xlabel('spatial frequency(lp/mm)'); ylabel('Contrast(%)');    
title('Modulation Tranfer function');hold on;
hline = refline([0 Climit]);
hline.Color = 'r';
%h = findobj(gca,'Type','line');
%x=get(h,'Xdata');
%y=get(h,'Ydata');
