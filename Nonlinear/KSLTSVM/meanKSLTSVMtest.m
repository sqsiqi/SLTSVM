function [t,meanACC,std_ACC,meanF1,std_F1]=meanKSLTSVMtest(X,Y,c,C,sigma,k,n)
tic;
acc=zeros(k,1);
F1_1=zeros(k,1);
ACC=zeros(n,1);
F1=zeros(n,1);
for j=1:n
[KAC,KBC,Y1]=RBF(X,Y,sigma);
K=[KAC;KBC];
[X_train,Y_train,X_validation,Y_validation]=KSLTSVMcv(K,Y1,k);
    for i=1:k
       [uu1,bb1,uu2,bb2]=KSLTSVM(c,C,X_train{i},Y_train{i});
       [err,f1]=KSLTSVMerror(X_validation{i},Y_validation{i},uu1,uu2,bb1,bb2);
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