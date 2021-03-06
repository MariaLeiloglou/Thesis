%Maria Leiloglou 02_10_2018
%this demo opens a single USAF 1951 image
%the user inputs the number of chosen group
%the user draws a line along the elements of the chosen group
%the demo opens the intensity profile for the chosen line
%the user clicks on the maximum and minimum data points for each one of the elements (which corresponds to a different spatial frequency [lp/mm])
%the demo opens the plot of contrast vs spatial frequency and 
%gives the resolution in the image for a contrast threshold defined by the Rayleigh criterion

clc
clear all
close all
waitfor(msgbox('Choose the USAF image'));
[filename, pathname] = uigetfile({'*.*';'*.tif';'*.png';'*.jpg';},'File Selector');%load the image
cd(pathname);im=imread(filename);imshow(im,[]);%view the image
%input the number of group you are working on
promptgroup='What is the number of group you have chosen?';
group=input(promptgroup);%choose the group to use
waitfor(msgbox('Draw a line across the elements of group'));
improfile(5000);% draw a line across the elements of the group

Cmatrix=zeros(1,6);%preallocation for the contrast in each element/this example is for the 3rd group (6 elements)
spatialfrequency=zeros(1,6);%preallocation for the corresponding spatial frequency of each element
for i=1:6 % i for element of the chosen group
    sf=2^(group+(((i)-1)/6));%resolutionn in lp/mm
    spatialfrequency(1,i)=sf;
    waitfor(msgbox(['click on maximum and minimum values of the element number ', num2str(i),' then press enter']));
    [x,y]=ginput;% input of values/ x is location y is intesity value
    Imax=max(y);%maximum intensity of element
    Imin=min(y);%minimum intensity of element
    C=((Imax-Imin)/(Imax+Imin))*(100);%contrast in %
    Cmatrix(1,i)=C;%put contrast in the matrix
    Climit=26.4;%contrast limit
    if C<=Climit
        Y=['This spatial frequency is:',num2str(sf),'[lp/mm] and it is below the limit'];
        Z=['The contrast is',num2str(C),'[%]'];
        disp(Y);disp(Z);

    else 
        Y=['This spatial frequency is:',num2str(sf),'[lp/mm]'];
        Z=['The contrast is',num2str(C),'[%]'];
        disp(Y);disp(Z);
        
    end
end

%Create the modulation transfer funnction
Figure1=figure;scatter(spatialfrequency,Cmatrix,'blue');%plot contast vs spatial frequency
xlim([min(spatialfrequency) max(spatialfrequency)]); 

xlabel('spatial frequency(lp/mm)'); ylabel('Contrast(%)');    
title('Modulation Tranfer function');hold on;
hline = refline([0 Climit]);%plot the contrast limit
hline.Color = 'r';
saveas(Figure1,'Modulation_Trasfer_Function.fig');
saveas(Figure1,'Modulation_Trasfer_Function.png');

for i=1:6 %for each element
    if Cmatrix(1,i)<Climit %if the contrast is below the limit
        if i>1 %only if we are not inn the first element of the group
           resolution=spatialfrequency(1,i-1);%save the previous element's spatial frequency as the resolution
           save('resolution_in_lp_per_mm.mat','resolution');
           break
        else %if we are in the first element of the group
           disp('Please try a group of lower spatial frequencies');%means we need to try another group of lower spatial frequencies
        break
        end
    elseif Cmatrix(1,i)>=Climit && i==6 %if the contrast of all elements are above the limit
        disp('Please try a group of higher spatial frequencies');% we need to use groups of higher spatial frequencies
   end
end

                

