%Maria Leiloglou 02_06_2021
%model training (with slope of PSD curve) and validation with ROC nalysis
clc;
clear all;
close all;
%go to the folder with the slope of PSD curves values 
%extracted from the healthy and tumour parts
%of the image you work on
cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/data');


%find the mat files with the vales in the folder
mat = dir('*.mat');


%load the files one by one 
 for q = 1:length(mat) % for each one of the files
     cont = load(mat(q).name); %load the file
     healthyslope=cont.Pballb(:,1);%values from healthy regions
     tumourslope=cont.Pballm(:,1);%values from tumour regions
     tumourslope = tumourslope(randperm(length(tumourslope)));%shuffle
     healthyslope = healthyslope(randperm(length(healthyslope)));%shuffle
     healthy=length(healthyslope); %find number of healthy values
     tumour=length(tumourslope);%find number of tumour values  
     %name of the mat file to use
     namevalidation=mat(q).name;
     str=convertCharsToStrings(namevalidation);
     Newstr=split(str,"_");
     filenamenew=Newstr(1,1);
     filenamenew= convertStringsToChars(filenamenew);


     tumourslope=reshape(tumourslope,1,length(tumourslope)); 
     healthyslope=reshape(healthyslope,1,length(healthyslope));
     [~,malignantnumber]=size(tumourslope);%number of tumour values
     [~,benignnumber]=size(healthyslope);%number of healthy values
     resp=zeros(1,malignantnumber+benignnumber);%preallocation to store the classification: 1 for tumour, 0 for healthy
     pred=zeros(1,malignantnumber+benignnumber);%preallocation to store the corresponding values
     resp(1,1:malignantnumber)=1;%here we are storing the classification: 1 for tumour, 0 for healthy
     pred(1,1:malignantnumber)=tumourslope;%we put the tumour values
     pred(1,1+malignantnumber:malignantnumber+benignnumber)=healthyslope;%we put the healthy values

     %train in random 70% of this data and validate in random 30% of this data

     trainmalignantnumber=round(malignantnumber/1.43);%100/70->number of 70% of tumour data
     trainbenignnumber=round(benignnumber/1.43);%100/70->number of 70% of healthy data
     validmalignantnumber=round(malignantnumber/3.33);%100/30->number of 15% of tumour data
     validbenignnumber=round(benignnumber/3.33);%100/30->number of 15% of healthy data

     predtrain=zeros(trainmalignantnumber+trainbenignnumber,1);%preallocation to store the slope values (only the data for training)
     resptrain=zeros(trainmalignantnumber+trainbenignnumber,1);%preallocation to store the classification: 1 for tumour, 0 for healthy (only the data for training)
     resptrain(1:trainmalignantnumber,1)=1;%here we are storing the classification: 1 for tumour, 0 for healthy(only 70% for training)
     predtrain(1:trainmalignantnumber,1)=tumourslope(1,1:trainmalignantnumber);%we put the tumour values for training
     predtrain(1+trainmalignantnumber:trainmalignantnumber+trainbenignnumber,1)=healthyslope(1,1:trainbenignnumber);%we put the healthy values for training

     %the below repeats the same but stores values for the validation process
     predvalidation=zeros(validmalignantnumber+validbenignnumber,1);
     respvalidation=zeros(validmalignantnumber+validbenignnumber,1);
     respvalidation(1:validmalignantnumber,1)=1;
     predvalidation(1:validmalignantnumber,1)=tumourslope(1,trainmalignantnumber+1:trainmalignantnumber+validmalignantnumber);
     predvalidation(1+validmalignantnumber:validmalignantnumber+validbenignnumber,1)=healthyslope(1,trainbenignnumber+1:trainbenignnumber+validbenignnumber);
     obs1=length(resptrain);
     obs2=length(respvalidation);
     %check if data are missing
     if sum(respvalidation) == 0 || sum(respvalidation) == obs2 || sum(resptrain) == 0 || sum(resptrain) == obs1
        disp([filenamenew,'missing one of the classes']);
     else

     resptrain=logical(resptrain);
     %the below trains the model

     %mdl=fitglm(predtrain,resptrain,'Distribution','binomial');%this line trains the logistic regression model
     %cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/Results/TextureSlope_Case_Wise');%change current folder
     %save([filenamenew,'_TEXTURE_logitmodel_70_30.mat'],'mdl');%we save the trained model
     mdl=fitcsvm(predtrain,resptrain,'Standardize',true);%this line trains the SVM model
     mdlSVM = fitPosterior(mdl);%this line trains the SVM model
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/Results/TextureSlope_Case_Wise');%change current folder
     save([filenamenew,'_TEXTURE_SVMmodel_70_30.mat'],'mdlSVM');%we save the trained model
     


     %The below visualizes the training data and the OUTCOME OF THE TRAINED
     %model on the same data/this will help us understand where the learned threshold
     %is.
     xx = linspace(min(predtrain), max(predtrain), 1000);
     xxnew=xx.';
     %score=predict(mdl,xxnew);
     score=predict(mdlSVM,xxnew);%same for the SVM model

     Figure1=figure;
     scatter(predtrain,resptrain);hold on;
     scatter(xxnew,score,'r','.');
     %title('Training of logistic regression model 70% of data');
     title('Training of SVM model 70% of data');
     xlabel('slope value');ylabel('classification');
     %legend('data','logistic regression model');
     legend('data','SVM model');%same for the SVM model
     %saveas(Figure1,[filenamenew,'_Training of logistic regression_30percentofdata.fig']);
     %saveas(Figure1,[filenamenew,'_Training of logistic regression_30percentofdata.png']);
     
     saveas(Figure1,[filenamenew,'Training of SVM model_30percentofdata.fig']);%same for the SVM model
     saveas(Figure1,[filenamenew,'Training of SVM model_30percentofdata.png']);%same for the SVM model
     
     %The below validates the model-uses the already trained model to make
     %predictions on new data and compares these predictions with the ground
     %trouth
     %score_log1=predict(mdl,predvalidation);%apply trained model to classify (0 or 1) for cancer for the new validation data
     score_log1=predict(mdlSVM,predvalidation); %same for SVM model
     %the below visualizes the validation data (in blue circles) and the prediction of the
     %trained LR model for these data (in red curve)/again we can see the
     %learned threshold
     Figure2=figure;
     scatter(predvalidation,respvalidation);hold on;
     scatter(predvalidation,score_log1,'r','.');
     %title('Validation of logistic regression model');
     title('Validation of SVM model');%same for SVM model
     xlabel('slope value');ylabel('classification');
     legend('data','Logistic Regression prediction');
     %saveas(Figure2,[filenamenew,'_Validation of logit trained model_30percentofdata.fig']); 
     %saveas(Figure2,[filenamenew,'_Validation of logit trained model_30percentofdata.png']);
     saveas(Figure2,[filenamenew,'_Validation of SVM trained model_30percentofdata.fig']);%same for SVM model 
     saveas(Figure2,[filenamenew,'_Validation of SVM trained model_30percentofdata.png']);%same for SVM model
     Rval=(sum(((score_log1-respvalidation).^2)))/(validmalignantnumber+validbenignnumber);

     %Now to understand whether these predictions that the trained model did are good
     %We will use ROC analysis to find accuracy of teh trained model in
     %detecting tumour.
     respvalidation=logical(respvalidation);%the ground truth
     score_log1=double(score_log1);%the model's prediction
     [X,Y,T,AUC,OPT] = perfcurve(respvalidation,score_log1,'true');% this is where the ROC analysis is implemented
     %ROC analysis plots the sensitivity vs 1-specificity for various
     %classification thresholds, T is the classification thresholds used, X,Y
     %are the corresponding 1-specificity and sensitivity scores for every classification threshold, respectively 
     %AUC is the arrea under the curve, it gives us the propability for this trained model to correctly
     %classify a random data point correclty.
     %OPT corresponds to the best threshold we can use so that we can have the
     %best specificity and sensitivity outcome
     %optimalthreshold=T((X==OPT(1))&(Y==OPT(2)));%only for logistic regression/suppress if SVM
     %it was a probability above which we classified as tumour/now we have 0 or
     %1 so we cannot use it


     %below we vizualise the ROC curve and optimal threshold for classification
     Figure3=figure;
     plot(X,Y)
     hold on
     plot(OPT(1),OPT(2),'ro')
     xlabel('False positive rate') 
     ylabel('True positive rate')
     %title(['ROC Curve with optimal treshold ',num2str(optimalthreshold),' indicated'])
     title('ROC Curve')
     hold off
     %saveas(Figure3,[filenamenew,'_LGM_ROC curve with optimal threshold_70_30',num2str(optimalthreshold),'.fig']); 
     %saveas(Figure3,[filenamenew,'_LGM_ROC curve with optimal threshold_70_30',num2str(optimalthreshold),'.png']);

     %save([filenamenew,'Roc_LGM_Analysis_70_30.mat'],'X','Y','T','AUC','OPT','optimalthreshold');
     
     saveas(Figure3,[filenamenew,'_SVM_ROC curve with optimal threshold_70_30.fig']); 
     saveas(Figure3,[filenamenew,'_SVM_ROC curve with optimal threshold_70_30.png']);

     save([filenamenew,'Roc_SVM_Analysis_70_30.mat'],'X','Y','T','AUC','OPT');


     %The below code will show the result as a green pseudocolour map on top of the colour image, the LR classification for tumour for every pixel is converted to 
     %transparency of the green map where optimal threshold found by LR model is used as the cut-off
     %threshold-meaning the transparency of the green map is 100% for the pixels
     %classified below this threshold and 0% above this threshold
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/data');%change current folder
     imf=imread([filenamenew,'_fluorescence.tiff']);%load fluorescence image
     dark=imread([filenamenew,'_dark.tiff']);%load dark image

     imf=imf-dark;
     [s1,s2]=size(imf);
     imc=imread([filenamenew,'_colour.tiff']);%load colour image


     %The below for loop predicts a classification(tumour or healthy) for each
     %one of the pixels in fluorescence image
     propabilitymap=zeros(s1,s2);%preallocation for the pseudocolour map
     %one pixel corresponds to this value in [mm]
     pixeldimension=[ 0.0422164
     0.03769
     0.03769
     0.04050978
     0.03764705
     0.04336
     0.038186
     0.038186
     0.04
     0.0493066
     0.041184
     0.0344086
     0.0448808
     0.0434783
     0.0448179
     0.0448808
     0.0402516
     0.0402516
     0.0422164
     0.04290655
     0.0435967
     0.0380499
     0.0375147
     0.0377358
     0.0376028
     0.0453258
     0.0460432
     0.0321285
     0.0319043
     0.0340426
     0.0425532
     0.0424967
     0.0415584
     0.0425532
     0.0427236
     0.0421053
     0.042838
     0.0427807
     0.0474074
     0.0344828
     0.0430686
     0.0461095
     0.0382775
     0.0380952
     0.0380952
     0.0382775
     0.03950617
     0.0395062
     0.0458453];
     res=1./((50*64).*pixeldimension);%resolution input to function raPsd2d
     res1=res(q);
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5');

     for i=1:64:s2-63
        for j=1:64:s1-63
        sample=imf(j:j+63,i:i+63);%square sample from image
        [Pf,f1,~,~]=raPsd2d((sample),res1);%radially averaged spectral density curve
        format long
        Pb = polyfit(f1,Pf,1);%linear fit of the curve
        prob=predict(mdl,Pb(1));%use the trained model to predict probability for tumour given the slope of the fit
        propabilitymap(j:j+63,i:i+63)=prob*ones(64,64);%use this probability for the whole image square
        end
     end

     %in the SVM model we don't need the below as our values are 0 or 1

     %{
     %the below for loop zeros all the pixel values that are classified below 
     %the optimal classification threshold
     for x=1:s1
        for y=1:s2
            if propabilitymap(x,y)<=optimalthreshold
                propabilitymap(x,y)=0;
            end
        end
     end

     %}
     %create extra mask for the specimen to remove background artefacts
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/masks');%change current folder
     load([filenamenew,'mask.mat']);%load mask to remobe background
     propabilitymap=immultiply(propabilitymap,ROIspecimen);%apply mask
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5/Results/TextureSlope_Case_Wise');%change current folder


     %The below demonstrates the classification of the trained model
     %BE CAREFUL the model has been trained/validated in 70%/30% of the ground truth data
     %from this image. This means that: 1) We do not use this to validate the model's performance 
     %as inn the below step we essentially apply the model partially on the same data we used for training 
     %2) We cannot compare the below overlay outcome with the validation AUC result
     % since in the below step we apply the model on all ground truth data
     %plus the rest of unknown data in the image while the validation is done
     %only in the 30% of the ground truth data
     green = cat(3, zeros(size(imf)),ones(size(imf)), zeros(size(imf))); %create the green plane
     Figure4=figure;imshow(imc); %open the colour image
     hold on;
     h = imshow(green); %overlay the green plane
     hold off 
     set(h, 'AlphaData', propabilitymap)%set the transparency of the green plane
     %to be inversely proportional with the classification for a pixel to be tumour
     %saveas(Figure4,[filenamenew,'Overlay_lgm with optimal threshold_70_30',num2str(optimalthreshold),'.fig']); 
     %saveas(Figure4,[filenamenew,'Overlay_lgm_with optimal threshold_70_30',num2str(optimalthreshold),'.png']); 
     saveas(Figure4,[filenamenew,'Overlay_svm_with optimal threshold_70_30.fig']); %same for SVM model
     saveas(Figure4,[filenamenew,'Overlay_svm_with optimal threshold_70_30.png']); %same for SVM model
     imf=double(imf);
     imfnew=immultiply(imf,propabilitymap);
     Figure5=figure;imshow(imfnew,[]);
     %saveas(Figure5,[filenamenew,'processed_fluorescence_lgm_with optimal threshold_70_30',num2str(optimalthreshold),'.fig']); 
     %saveas(Figure5,[filenamenew,'processed_fluorescence_lgm_with optimal threshold_70_30',num2str(optimalthreshold),'.png']); 
     saveas(Figure5,[filenamenew,'processed_fluorescence_svm_with optimal threshold_70_30.fig']); %same for SVM model
     saveas(Figure5,[filenamenew,'processed_fluorescence_svm_with optimal threshold_70_30.png']); %same for SVM model
     cd('/Users/marialeiloglou/Documents/GitHub/Thesis/Appendix_5');
     close all;
     end
 end