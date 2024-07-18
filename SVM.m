clc;clear;close all;	
load('R_18_Jul_2024_22_47_31.mat')	
	
data_str=G_out_data.data_path_str ;   	
	
	
data1=readtable(data_str,'VariableNamingRule','preserve'); 
data2=data1(:,2:end); 	
data=table2array(data1(:,2:end));	
data_biao=data2.Properties.VariableNames;  
str_label=0; 
  A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	
	
  x_feature_label=data_select(:,1:end-1);    	
  y_feature_label=data_select(:,end);          	
 index_label1=randperm(size(x_feature_label,1));	
 index_label=G_out_data.spilt_label_data;  	
 if isempty(index_label)	
     index_label=index_label1;	
 end	
spilt_ri=G_out_data.spilt_rio;  
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); 
	
 train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
 train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
 vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
 test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
 test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
	
 x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
 train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;   	
 y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    	
	
 vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    	
 vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  	

test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    	
 test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    
	

num_pop=G_out_data.num_pop1;   
num_iter=G_out_data.num_iter1;   
method_mti=G_out_data.method_mti1;   
BO_iter=G_out_data.BO_iter;   
min_batchsize=G_out_data.min_batchsize;   
max_epoch=G_out_data.max_epoch1;   
hidden_size=G_out_data.hidden_size1;   
attention_label=G_out_data.attention_label;   
attention_head=G_out_data.attention_head;   
	
	
	

	
	
	
	
disp('SVM') 	
t1=clock; 	
t=templateSVM('Standardize',true,'KernelFunction','gaussian');	
   Mdl = fitcecoc(train_x_feature_label_norm,train_y_feature_label,'Learners',t,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',BO_iter));    	
	
	
	
	
[~,score_train] = predict(Mdl,train_x_feature_label_norm);[~,y_train_predict] = max(score_train');y_train_predict=y_train_predict';	
[~,score_vaild] = predict(Mdl,vaild_x_feature_label_norm);[~,y_vaild_predict] = max(score_vaild');y_vaild_predict=y_vaild_predict';	
[~,score_test] = predict(Mdl,test_x_feature_label_norm);[~,y_test_predict] = max(score_test');y_test_predict=y_test_predict';	
t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
 CVMdl = crossval(Mdl);classAUC=1-kfoldLoss(CVMdl);disp(['10-fold',num2str(classAUC)])	
	
disp(['time ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; 	
FP_train = sum(confMat_train, 1)  - TP_train;  	
FN_train = sum(confMat_train, 2)' - TP_train;  
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  
	
disp('Train')	
accuracy_train = sum(TP_train) / sum(confMat_train(:));  disp(['train accuracy：',num2str(mean(accuracy_train))])

disp('Validation')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; 
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:));  disp(['validation accuracy：',num2str(accuracy_vaild)])	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; 
FP_test = sum(confMat_test, 1)  - TP_test;  
FN_test = sum(confMat_test, 2)' - TP_test;  
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  
	
disp('Test') 	
accuracy_test = sum(TP_test) / sum(confMat_test(:));  disp(['test accuracy：',num2str(accuracy_test)])
