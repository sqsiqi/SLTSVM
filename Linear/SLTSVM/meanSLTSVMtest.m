function [t,meanACC,std_ACC,meanF1,std_F1]=meanSLTSVMtest(X,Y,c,C,k,n)
%10 times 10-fold cross-validation, n=10,k=10. 
tic;
acc=zeros(k,1);
F1_1=zeros(k,1);
ACC=zeros(n,1);
F1=zeros(n,1);
for j=1:n
  [X_train,Y_train,X_validation,Y_validation]=SLTSVMcv(X,Y,k);
    for i=1:k
       [ww1,bb1,ww2,bb2]=SLTSVM(c,C,X_train{i},Y_train{i});
       [err,f1]=SLTSVMerror(X_validation{i},Y_validation{i},ww1,ww2,bb1,bb2);
       acc(i,1)=1-err;
       F1_1(i,1)=f1;
    end
    ACC(j,1)=mean(acc);
    F1(j,1)=mean(F1_1);
    fprintf('j is %d \n',j)
end
meanACC=mean(ACC);
std_ACC=std(ACC);
meanF1=mean(F1);
std_F1=std(F1);
mmm=toc;
t=mmm/n;