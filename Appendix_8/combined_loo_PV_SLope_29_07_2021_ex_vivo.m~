%Maria Leiloglou 17/04/2021

%leave-one-out-cross-validation-logistic-regression
%here we use both the slope of the PSD curves and the normalised pixel
%values to classify tissue 



clc;
clear all;
close all;

%load the slope data
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Slope_Intercept_Results/Angiography_cohort/ex_vivo');
matslope = dir('*.mat');


sumhslope=0;%preallocation for sum_of_healthy_pixels

sumtslope=0;%preallocationn for sum_of_tumour_pixels
%load the files one by one 
 for q = 1:length(matslope) % for each one of the files
     contslope = load(matslope(q).name); %load the file
     healthyslope=contslope.Pballb(:,1);
     tumourslope=contslope.Pballm(:,1);
     healthys=length(healthyslope); %find number of healthy samples
     eval(['healthy_number_slope' num2str(q) '=healthys;']);% save the number of healthy slope values
     tumours=length(tumourslope);%find number of tumour samples   
     eval(['tumour_number_slope' num2str(q) '=tumours;']);% save the number of tumour slope values

     eval(['healthy_slopevalue_' num2str(q) '=healthyslope;']); %save healthy slope values from each mat file
     eval(['tumour_slopevalue_' num2str(q) '=tumourslope;']); %save tumour slope values from each mat file
     sumhslope=sumhslope+healthys;%here you sum the number of healthy samples
     sumtslope=sumtslope+tumours;%here you sum the number of tumour samples
 end


 %here you save in the vector 'sihealthyslope'  the number of healthy samples
 %per case 
 sihealthyslope=zeros(1,length(matslope));
 for i=1:length(matslope)
     eval(['numbers' '=healthy_number_slope' num2str(i) ';']);
     sihealthyslope(1,i)=numbers;
 end
 
 %here you save in the vector 'situmourslope'  the number of tumour samples
 %per case 
  situmourslope=zeros(1,length(matslope));
 for i=1:length(matslope)
     eval(['numbers' '=tumour_number_slope' num2str(i) ';']);
     situmourslope(1,i)=numbers;
 end
 
 
 trainingslope=1:length(matslope);
 fulllengthslope=length(matslope);% number of all cases/images

 lengthslope=(fulllengthslope-1);%  number of training cases
 trainingnewslope=zeros(1,lengthslope);
 
%load the normalised fluorescence  pixel values 
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Normalised_pixel_values_Results');

for l=1:fulllengthslope%number of ex vivo cases 
    namevalidation=matslope(l).name;%keep the name of the case used for validation  for use
    str=convertCharsToStrings(namevalidation);
    Newstr=split(str,"_");
    filenamenew=Newstr(1,1);
    filenamenew= convertStringsToChars(filenamenew);
    
    eval(['healthyslopesamples' '=healthy_number_slope' num2str(l) ';']);%save number of healthy samples
    eval(['tumourslopesamples' '=tumour_number_slope' num2str(l) ';']);%save number of tumour samples
    eval([filenamenew 'healthydoublefeatures' '=zeros(2,healthyslopesamples*64*64);']);%define a matrix for both features from healthy parts
    eval([filenamenew 'tumourdoublefeatures' '=zeros(2,tumourslopesamples*64*64);']);%define a matrix for both features from tumour parts

    %put all the samples in one matrix from each case
    for t=1:healthyslopesamples
        healthypixels=load([filenamenew,'healthy_pixels',num2str(t),'.mat']);
        healthypixels=healthypixels.healthypixels;
        eval([filenamenew 'healthypixelvalues' num2str(t) '=healthypixels;']);
        eval([filenamenew 'healthydoublefeatures(1,(t-1)*4096+1:t*4096)' '=healthypixels;']);
        eval([filenamenew 'healthydoublefeatures(2,(t-1)*4096+1:t*4096)' '=ones(1,4096)*healthy_slopevalue_' num2str(l) '(' num2str(t) ');']);
    end
       
   for t=1:tumourslopesamples
       tumourpixels=load([filenamenew,'tumour_pixels',num2str(t),'.mat']);
       tumourpixels=tumourpixels.tumourpixels;
       eval([filenamenew 'tumourpixelvalues' num2str(t) '=tumourpixels;']);
       eval([filenamenew 'tumourdoublefeatures(1,(t-1)*4096+1:t*4096)' '=tumourpixels;']);
       eval([filenamenew 'tumourdoublefeatures(2,(t-1)*4096+1:t*4096)' '=ones(1,4096)*tumour_slopevalue_' num2str(l) '(' num2str(t) ');']);
   end
   
   %save the matrices with the two features for each case
   cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/results/Double_features');
   save([filenamenew,'doublefeatures.mat'],[filenamenew,'healthydoublefeatures'],[filenamenew,'tumourdoublefeatures']);
   cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Normalised_pixel_values_Results');

end




sumh=0;%preallocation for sum_of_healthy_pixels

sumt=0;%preallocation for sum_of_tumour_pixels
%load the files one by one 
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Slope_Intercept_Results/Angiography_cohort/ex_vivo');
matslope = dir('*.mat');

 for q = 1:length(matslope) % for each one of the files
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Slope_Intercept_Results/Angiography_cohort/ex_vivo');
     contslope = load(matslope(q).name); %load the file
     namevalidation=matslope(q).name;%keep the name of the case used for validation  for use
     str=convertCharsToStrings(namevalidation);
     Newstr=split(str,"_");
     filenamenew=Newstr(1,1);
     filenamenew= convertStringsToChars(filenamenew);
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/results/Double_features');
     load([filenamenew,'doublefeatures.mat'],[filenamenew,'healthydoublefeatures'],[filenamenew,'tumourdoublefeatures']);
     % healthypixels=cont.healthypixels;
     %tumourpixels=cont.tumourpixels;
     eval(['healthy=length([' filenamenew 'healthydoublefeatures]);']) %find number of healthy pixel values
     eval(['healthy_number' num2str(q) '=healthy;']);% save the number of healthy pixel values
     eval(['tumour=length([' filenamenew 'tumourdoublefeatures]);'])%find number of tumour pixel values
     eval(['tumour_number' num2str(q) '=tumour;']);% save the number of tumour pixel values
     
     eval(['healthy_' num2str(q) '=' filenamenew 'healthydoublefeatures(1,:);']); %save healthy pixels from each mat file
     eval(['healthys_' num2str(q) '=' filenamenew 'healthydoublefeatures(2,:);']); %save healthy slope values from each mat file
     
     eval(['tumour_' num2str(q) '=' filenamenew 'tumourdoublefeatures(1,:);']); %save tumour pixels from each mat file
     eval(['tumours_' num2str(q) '=' filenamenew 'tumourdoublefeatures(2,:);']); %save tumour slope values from each mat file

           
     sumh=sumh+healthy;%here you sum the number of healthy pixel values
     sumt=sumt+tumour;%here you sum the number of tumour pixel values
 end


 %here you save in the vector 'sihealthy'  the number of healthy pixel values
 %per case 
 sihealthy=zeros(1,length(matslope));
 for i=1:length(matslope)
     eval(['number' '=healthy_number' num2str(i) ';']);
     sihealthy(1,i)=number;
 end
 
 %here you save in the vector 'situmour'  the number of tumour pixel values
 %per case 
  situmour=zeros(1,length(matslope));
 for i=1:length(matslope)
     eval(['number' '=tumour_number' num2str(i) ';']);
     situmour(1,i)=number;
 end
 
 
 training=1:length(matslope);
 fulllength=length(matslope);% number of all cases/images

 length1=(fulllength-1);%  number of training cases
 trainingnew=zeros(1,length1);
 
 
 

 
 
% cd('C:\Users\ml1016\Desktop\Angiography_normalised_PV\ex_vivo');

 for i=1:fulllength%the number of the case we will use for validation
     
     %here you save in the vector 'sihealthy'  the number of healthy pixel values
     %per case that will be used for training
     sihealthynew=zeros(1,fulllength-1);
     sihealthynew(1:i-1)=sihealthy(1:i-1);
     sihealthynew(i:fulllength-1)=sihealthy(i+1:fulllength);
     
     
     %here you save in the vector 'situmour'  the number of tumour pixel values
     %per case that will be used for training
     situmournew=zeros(1,fulllength-1);
     situmournew(1:i-1)=situmour(1:i-1);
     situmournew(i:fulllength-1)=situmour(i+1:fulllength);
     
     %read the name of the case/image you will validate on
     namevalidation=matslope(i).name;
     str=convertCharsToStrings(namevalidation);
     Newstr=split(str,"_");
     filenamenew=Newstr(1,1);
     filenamenew= convertStringsToChars(filenamenew);
   
    
     %add the pixel value ground truth data
     trainingnew(1:(i-1))=1:(i-1);
     trainingnew(i:length1)=(i+1):fulllength;%the number of all the other cases we will use for training
     traininghealthynumber= sum(sihealthy(1:(i-1)))+sum(sihealthy((i+1):fulllength));% number of healthy pixels for training
     trainingtumournumber= sum(situmour(1:(i-1)))+sum(situmour((i+1):fulllength));% number of tumour pixels for training
     validatinghealthynumber=sihealthy(i);% number of healthy pixels for validation
     validatingtumournumber=situmour(i);% number of tumour pixels for validation
     %put all the pixel values in two vectors for training
     vector_healthy=zeros(2,traininghealthynumber);%preallocation for double feature healthy matrix for training
     vector_tumour=zeros(2,trainingtumournumber);%preallocation for double feature tumour matrix for training
     
     %this for loop puts the pixel values for training in the healthy vector
     for q=1:length1%this is just the number of cases, not the case id number
         
         k=trainingnew(q);%this is the case id number
         
         %cont = load(mat(k).name);
         %healthypixels=cont.healthypixels;
        
         eval(['healthypixels=healthy_' num2str(k) ';']);
         eval(['healthyslopes=healthys_' num2str(k) ';']); 
         if q==1
             vector_healthy(1,1:sihealthynew(q))=healthypixels;
             vector_healthy(2,1:sihealthynew(q))=healthyslopes;
         else
             
            allprevious=sum(sihealthynew(1:q-1));
            start2=sihealthynew(q);
            startall=allprevious+1;      
            endall=start2+allprevious;  
            vector_healthy(1,startall:endall)=healthypixels;%all the healthy pixel values for training    
            vector_healthy(2,startall:endall)=healthyslopes;
         end
     end
     
     %this for loop puts the pixel values for training in the tumour vector
     for q=1:length1%this is just the number of cases, not the case id number
         
         k=trainingnew(q);%this is the case id number
         
         %cont = load(mat(k).name);
         eval(['tumourpixels=tumour_' num2str(k) ';']);
         eval(['tumourslopes=tumours_' num2str(k) ';']); 
         
         if q==1
             vector_tumour(1,1:situmournew(q))=tumourpixels;
             vector_tumour(2,1:situmournew(q))=tumourslopes;

         else
            allprevious=sum(situmournew(1:q-1));
            start2=situmournew(q);      
            startall=allprevious+1; 
            endall=start2+allprevious;     
            vector_tumour(1,startall:endall)=tumourpixels;%all the healthy pixel values for training
            vector_tumour(2,startall:endall)=tumourslopes;
         end
          
    end

  
     %%put all the pixel values in two vectors for validation
     validationhealthynumber= validatinghealthynumber;
     validationtumournumber= validatingtumournumber;
     validation_vector_healthy=zeros(2,validationhealthynumber);
     validation_vector_tumour=zeros(2,validationtumournumber);
     
     eval(['validation_vector_healthy(1,:)' '=healthy_' num2str(i) ';']);
     eval(['validation_vector_tumour(1,:)' '=tumour_' num2str(i) ';']);
     eval(['validation_vector_healthy(2,:)' '=healthys_' num2str(i) ';']);
     eval(['validation_vector_tumour(2,:)' '=tumours_' num2str(i) ';']);
  
     %prepare the response vector for the trainnig
     response_training=zeros(1,traininghealthynumber+trainingtumournumber);
     response_training(1,1:trainingtumournumber)=1;
   
     %prepare the predictor vector for the training
     predictor_training=zeros(2,traininghealthynumber+trainingtumournumber);
     predictor_training(:,1:trainingtumournumber)=vector_tumour;
     predictor_training(:,trainingtumournumber+1:trainingtumournumber+traininghealthynumber)=vector_healthy;  
   
 
     
     %prepare the response vector for validation
     response_validation=zeros(1,validationhealthynumber+validationtumournumber);
     response_validation(1,1:validationtumournumber)=1;
     
     %prepare the predictor vector for validation
     predictor_validation=zeros(2,validationhealthynumber+validationtumournumber);
     predictor_validation(:,1:validationtumournumber)=validation_vector_tumour;
     predictor_validation(:,validationtumournumber+1:validationtumournumber+validationhealthynumber)=validation_vector_healthy;
     predictor_training=predictor_training.';
     response_training=response_training.';
     
     %the below trains the model
     mdl=fitglm(predictor_training,response_training,'Distribution','binomial');%this line trains the logistic regression model
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/results/LGM_LOO_double_features');%change current folder
     save(['logitmodel_LOO_',filenamenew,'.mat'],'mdl');%we save the trained model
    
     %the below visualizes the training data and the trained logistic regression model
     xx = linspace(min(predictor_training(:,1)), max(predictor_training(:,1)), 1000);
     xxnew=xx.';

     xxs = linspace(min(predictor_training(:,2)), max(predictor_training(:,2)), 1000);
     xxsnew=xxs.';

     xxdouble=zeros(1000,2);
     xxdouble(:,1)=xxnew;
     xxdouble(:,2)=xxsnew;

     score=predict(mdl,xxdouble);
     Figure1=figure;
     scatter(predictor_training(:,1),response_training);hold on;
     scatter(xxdouble(:,1),score,'r','.');
     title('Training of logistic regression model LOO');
     xlabel('pixel value');ylabel('classification');
     legend('data','logistic regression');
     saveas(Figure1,['Trained model_LOO',filenamenew,'.svg']); 
     saveas(Figure1,['Trained model_LOO',filenamenew,'.png']);

     score_log1=predict(mdl,predictor_training);
     response_training=logical(response_training);
     score_log1=score_log1.';
     [X,Y,T,AUC,OPT] = perfcurve(response_training,score_log1,'true');
     optimalthreshold=T((X==OPT(1))&(Y==OPT(2)));%threshold of probability found by ROC curve

     %the below validates the model-uses the already trained model to make
     %predictions on new data and compares these predictions with the ground
     %truth
     xxv = linspace(min(predictor_validation(:,1)), max(predictor_validation(:,1)), 1000);
     xxvnew=xx.';

     xxsv = linspace(min(predictor_validation(:,2)), max(predictor_validation(:,2)), 1000);
     xxsvnew=xxs.';

     xxdoublev=zeros(1000,2);
     xxdoublev(:,1)=xxvnew;
     xxdoublev(:,2)=xxsvnew;

     scorev=predict(mdl,xxdoublev);

     predictor_validation=predictor_validation.';
     score_log2=predict(mdl,predictor_validation);%apply trained model to predict propabilities for cancer for the new validation data
     %the below visualizes the validation data (in blue circles) and the prediction of the
     %trained logistic regression model for these data (in red curve)
     predictor_validation=predictor_validation.';
     Figure2=figure;
     scatter(predictor_validation(1,:),response_validation);hold on;
     %score_log2=score_log2.';

     scatter(xxdoublev(:,1),scorev,'r','.');
     title('Validation of logistic regression model');
     xlabel('pixel value');ylabel('classification');
     legend('data','logistic regression prediction');
     saveas(Figure2,['Validation of trained model_LOO',filenamenew,'.svg']); 
     saveas(Figure2,['Validation of trained model_LOO',filenamenew,'.png']);

     clear vector_healthy vector_tumour

 
 
    %Now to understand whether these predictions that the trained model did are good
    %we will use ROC analysis to find accuracy of the trained model in
    %detecting tumour.
    
    %in the below for loop we use the learned probability threshold  to
    %convert the continuous logistic regression probability outcome into a
    %class (0 for healthy and 1 for tumour)
    response_validation=logical(response_validation);
    score_log2=score_log2.';
    binaryoutcome=zeros(1,size(score_log2,2));
    for t=1:size(score_log2,2)
      if  score_log2(1,t)>optimalthreshold 
        binaryoutcome(1,t)=1;
      else
        binaryoutcome(1,t)=0;
      end
    end

    [X,Y,T,AUC,OPT] = perfcurve(response_validation,binaryoutcome,'true');% this is where the ROC analysis is implemented
    %ROC analysis plots the sensitivity vs 1-specificity for various
    %classification thresholds, T is the classification thresholds used, X,Y
    %are the corresponding 1-specificity and sensitivity scores for every classification threshold, respectively 
    %AUC is the arrea under the curve, it gives us the propability for this trained model to correctly
    %classify a random data point correclty.
    


    %below we vizualise the ROC curve and optimal threshold for classification
    %note that the optimal threhold for classification is useless here as
    %we use the threshold learned with the training set to classify data
    %and ROC analysis is used here to get the
    %Accuracy/Sensitivity/Specificity when this learned threshold is used
    Figure3=figure;
    plot(X,Y)
    hold on
    plot(OPT(1),OPT(2),'ro')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title(['ROC Curve with optimal treshold ',num2str(optimalthreshold),' indicated'])
    hold off
    saveas(Figure3,['ROC curve ',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.svg']); 
    saveas(Figure3,['ROC curve ',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.png']); 

    save(['Roc_Analysis_',filenamenew,'_LOO.mat'],'X','Y','T','AUC','OPT','optimalthreshold')


    %Overlay/ optional 
    cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Images');

    %the below code will show the result as a green pseudocolour map on top of the colour image, the propability for tumour for every pixel is converted to 
    %transparency of the green map and optimal threshold is used as the cut-off
    %threshold-meaning the transparency of the green map is 100% for the pixels
    %classified below this threshold
    imf=imread([filenamenew,'_fluorescence.tiff']);%read fluorescence image
    dark=imread([filenamenew,'_dark.tiff']);%read fluorescence image with light source off
    imf=imf-dark;%correct for ambient light
    immin=min(min(imf));%find minimum fluorescence pixel value
    immin=double(immin);
    immax=max(max(imf));%find maximum fluorescence pixel value
    immax=double(immax);
    [s1,s2]=size(imf);

    imc=imread([filenamenew,'_colour.tiff']);%read colour image

    %the below for loop predicts a classification(tumour or healthy) for each
    %one of the pixels in fluorescence image
    propabilitymap=zeros(s1,s2);
    propabilitymapsample=zeros(64,64);

    pixeldimension=[0.04050978
    0.04336
    0.0493066];%one pixel corresponds to this value in [mm]/different for every image
    res=1./((50*64).*pixeldimension);
    res1=res(i);
    samplefeatures=zeros(2,64,64);
    for v=1:64:s2-63
        for j=1:64:s1-63
            sample=imf(j:j+63,v:v+63);%square samples from image
            cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8');
            [Pf,f1,~,~]=raPsd2d((sample),res1);%radially averaged spectral density curve
            format long
            Pb = polyfit(f1(1:2),Pf(1:2),1);%linear fit of curve

            samplefeatures(2,:,:)=ones(64,64)*Pb(1);%put the feature-slope of PSD curves
            sample=double(sample);
            samplefeatures(1,:,:)=(sample-immin)./(immax-immin);%put the feature-normalised pixel values 
            %prob=predict(mdl,Pb(1));
            for k=1:64
                column=samplefeatures(:,:,k);
                column=column.';
                propabilitymapsample(:,k)=predict(mdl,column);%use the trained model and features to predict probability for tumour
            end
            propabilitymap(j:j+63,v:v+63)=propabilitymapsample;
        end

    end








    %the below for loop zeros all the pixel values that are classified below 
    %the optimal classification threshold (changes continuous probability outcome
    %to class)
    for x=1:s1
        for y=1:s2
            if propabilitymap(x,y)<=optimalthreshold
                propabilitymap(x,y)=0;
            end
        end
    end
    cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/masks');


    load([filenamenew,'mask.mat']); %load mask to remove background
   
    propabilitymap=immultiply(propabilitymap,ROIspecimen);%apply mask to remove background
    %the below visualises the classification of the trained model
    green = cat(3, zeros(size(imf)),ones(size(imf)), zeros(size(imf))); %create the green plane
    Figure4=figure;imshow(imc); %open the colour image
    hold on;
    h = imshow(green); %overlay the green plane
    hold off 
    set(h, 'AlphaData', propabilitymap)%set the transparency of the green plane
    %to be inversely proportional with the propability for a pixel to be tumour
    cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/results/Overlays');
    saveas(Figure4,['Overlay_lgm',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.svg']); 
    saveas(Figure4,['Overlay_lgm',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.png']);
    imf=double(imf);
    imfnew=immultiply(imf,propabilitymap);
    Figure5=figure;imshow(imfnew,[]);
    saveas(Figure5,['Fluorescence_lgm',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.svg']); 
    saveas(Figure5,['Fluorescence_lgm',filenamenew,'with optimal probability threshold_LOO',num2str(optimalthreshold),'.png']);
    close all;
    cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_8/data/Slope_Intercept_Results/Angiography_cohort/ex_vivo');
 end
  



     
 

 


 



