function [X_train,Y_train,X_validation,Y_validation]=KSLTSVMcv(X,Y,k)
%Cross-validation divides the training and test sets and k denotes the number of folds
A=[X,Y];
[M1,~]=size(A);
indices1=crossvalind('Kfold',M1,k);
for i=1:k
    test1=(indices1==i);
    train1=~test1;
    x_train=A(train1,:);
    x_test=A(test1,:);
    [~,m1]=size(x_train);
    [~,m3]=size(x_test);
    X_train{i}=x_train(:,1:m1-1);
    Y_train{i}=x_train(:,m1);
    X_validation{i}=x_test(:,1:m3-1);
    Y_validation{i}=x_test(:,m3);
end