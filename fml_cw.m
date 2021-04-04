%read the FULL  training data from excel file using xls read function 
xread1 = csvread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\bring-back-the-sun\training.csv',1,1);
%store prediciton column in a variable
x1_prediction =  xread1(:,4609);
%delete prediction column
xread1(:,4609) = [];

%read the missing training data from excel 
% x2 = xlsread(('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\bring-back-the-sun\additional_training.csv');
xread2 = csvread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\bring-back-the-sun\additional_training.csv',1,1);
%store prediction column in a variable 
x2_prediction = xread2(:,4609);
%delete prediction column
xread2(:,4609) =  [];


%IMPUTATION
%concatenate each column of the clean_training to each column of the missing_training
big_trainingtable= vertcat(xread1, xread2);

%concatenate the predictions 
trainingpredictions = vertcat(x1_prediction, x2_prediction);

%calculate the mean of the big training table, excluding the NaN
trainingmean = mean(big_trainingtable, 'omitnan');

%imputate to fill in the missing values from the table 
for x = 1:4608
%     xrow = big_trainingtable(:,x);
%     xsum = sum(big_trainingtable(:,x));
    fill = fillmissing(big_trainingtable, 'constant', trainingmean);
    big_trainingtable = fill;
end    

%read the class priors csv. [sunny(1), notsunny(0)]
priors = [0.3233 ;0.6767];

%cross validation, k-fold
cvp = cvpartition(2590, 'KFold', 4);

%call svm classification function
% svm_mdl = fitcsvm(big_trainingtable, trainingpredictions, 'KernelScale', 'auto', 'Standardize',true, 'solver', 'ISDA', 'verbose', 1, 'Nu', 0.3233, 'CrossVal', 'on');
% svm_mdl = fitcsvm(big_trainingtable, trainingpredictions, 'Standardize',true, 'KernelFunction',  'linear', 'solver', 'ISDA', 'verbose', 1, 'Prior',[0.3233 0.6767],'Weights',confidenceread, 'CrossVal', 'on', 'CVPartition', cvp);
confidenceread = csvread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\bring-back-the-sun\annotation_confidence.csv',1,1);
svm_mdl = fitcsvm(big_trainingtable, trainingpredictions, 'KernelScale','auto','KernelFunction', 'linear', 'Standardize',true, 'solver', 'ISDA', 'Prior',[0.3233 0.6767],'Weights',confidenceread, 'OptimizeHyperparameters', 'auto');
%Cross validtae mdl
crossval_mdl = crossval(svm_mdl, 'CVPartition', cvp);
%read the testing data so it can be passed to the predict function
testread = csvread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\bring-back-the-sun\testing.csv',1,1);


%predict SVM
% [label]= predict(svm_mdl, testread);

%predict using the cv model and test set, using the regular predict function
% % [label1] = predict(svm_mdl, testread);

%predict function for cross validation
[label2] = kfoldpredict(crossval_mdl);
%will return 2590 predictions corresponding to different folds
Y = kfoldPredict(crossval_mdl);
%test function to retrieve labels from of the folds
test_index = test(cvp, 1);
%use this test index in the variable 
predicted_class_labels = Y(test_index);



%contains trueclass labels, belonging to the first fold
%plot class performance curve on the first fold 
actual_class_labels = crossval_mdl.Y(test_index)
[labels, scores] = kfoldPredict(crossval_mdl);
predicted_class_labels = labels(test_index)
classification_scores = scores(test_index,:);
[X1, Y1] = perfcurve(actual_class_labels, classification_scores(:,1), '0')
[X2, Y2] = perfcurve(actual_class_labels, classification_scores(:,2), '1')
figure;
plot([X1, X2], [Y1, Y2])
title('Performance Curve of the First(of 4) K-Fold of the Cross Validated Model Between notSunny(0) Class and Sunny Class(1)')
xlabel('Actual Class Labels')
ylabel('Classifcation Scores Obtained from the KfoldPredict Function on the Cross Validated Model')
legend({'notSunny class(0)', 'Sunny Class(1)'})
%plot ROC cruve on the first fold,(using the confusion matrix)
% %plot confusion of larger data 
% t_trainingpredictions = transpose(trainingpredictions);
% t_label2 = transpose(label2);
% plotconfusion(t_trainingpredictions, t_label2)
% plotroc(t_trainingpredictions, t_label2)
% %plot on roc graph FP vs TP

%plot confusion matrix using confusionmat of the first fold
[C, order] = confusionmat(original_labels, predicted_class_labels);
t_original_labels = transpose(original_labels)
t_predicted_class_labels = transpose(predicted_class_labels)
plotconfusion(t_original_labels, t_predicted_class_labels)
plotroc(t_original_labels, t_predicted_class_labels);


%compare to logistic regression


mdl_logreg = fitglm(big_trainingtable, trainingpredictions)






%distributin link weights 




%write into csv file
% %first column  in the test 
% column_test = csvread('testing.csv', 1,0);
% column_test = column_test(:,1);
% csvwrite('finalprediction.csv', [0, 0; column_test, label])

column_test = csvread('testing.csv', 1,0);
column_test = column_test(:,1);
csvwrite('finalprediction_HO.csv', [0, 0; column_test, label])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%linear classification 

linreg_mdl = fitclinear(big_trainingtable, trainingpredictions,'Learner', 'logistic', 'Regularization', 'lasso', 'Prior',[0.3233 0.6767],'Weights',confidenceread)
[label3] = predict(linreg_mdl, testread);




column_test = csvread('testing.csv', 1,0);
column_test = column_test(:,1);
csvwrite('finalprediction_LR.csv', [0, 0; column_test, label3]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot
confusionmat(trainingpredictions, label2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%logistic linear regression
logreg_mdl = fitglm(big_trainingtable, trainingpredictions, 'Distribution', 'binomial', 'weights', confidenceread);
[label4] = predict(logreg_mdl, testread);



d = (label4)';
for L = 1:2818
    if d(L) ~= 0.0000 || d(L)~=1.0000
       d(L) = 0;
    end
end
b = de2bi(d);