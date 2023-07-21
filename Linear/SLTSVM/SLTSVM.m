function  [ww1,bb1,ww2,bb2]=SLTSVM(c,C,X,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLTSVM: Symmetric LINEX loss twin support vector machine (linear version)
% Use method: Import the data first, and then run the "meanSLTSVMtest" procedure
% Input:
%    X: Training data.
%    Y: Training data labels. (Y must include 1 and -1)
% Parameters:
% c: The loss term regularization parameter
% C: The structural risk term regularization parameter
% a1: The LINEX loss function parameter 
% a2: The LINEX loss function parameter  
% Output:
% ww1: The positive hyperplane parameter w_+
% ww2: The negative hyperplane parameter w_-
% bb1: The positive hyperplane parameter b_+
% bb2: The negative hyperplane parameter b_-
% fun1 and fun2 are used to analyze the convergence of the objective function
% Reference:
%    Qi Si, Zhi-Xia Yang, Jun-You Ye, et. al. "Symmetric LINEX loss twin support vector machine for robust
%    classification and its fast iterative algorithm"  Submitted 2023
%
%    Written by Qi Si (1418426889@qq.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1=1;
a2=-1;
X1=[X,ones(size(X,1),1)];
A=[X1,Y];
x1=A(A(:,end)==1,1:end-1);%positive sample matrix E
x2=A(A(:,end)==-1,1:end-1);%negative sample matrix F
n1=size(x1,2);%Number of data feature dimensions
n2=size(x2,2);%Number of data feature dimensions

%Compute the objective function of SLTSVM1
% function f1=fun1(Z1)
% M1=zeros(1,size(x1,1));
% M2=zeros(1,size(x2,1));
% for j=1:size(x1,1)
% M1(1,j)=(1/2)*(x1(j,:)*Z1).^2;
% end
% for j=1:size(x2,1)
% M2(1,j)=c*(exp(-a1*(x2(j,:)*Z1+1))+a1*(x2(j,:)*Z1+1)-1);
% end
% f1=sum(M1)+sum(M2)+norm(Z1,2)^2;
% end

%Calculate the gradient of the loss term for each iteration of SLTSVM1
function f1_Z1=gfun1(Z1)
R2=zeros(size(x2,2),size(x2,1));
for j=1:size(x2,1)
    if(-(x2(j,:)*Z1+1)>0)
        a=a2;
    else
        a=a1;
    end
R2(:,j)=-c*a.*x2(j,:)'.*(exp(-a*(x2(j,:)*Z1+1))-1);
end
f1_Z1=sum(R2,2);
end

%Compute the objective function of SLTSVM2
% function f2=fun2(Z2)
% N2=zeros(1,size(x1,1));
% N1=zeros(1,size(x2,1));
% for j=1:size(x1,1)
% N2(1,j)=c*(exp(a2*(x1(j,:)*Z2-1))-a2*(x1(j,:)*Z2-1)-1);
% end
% for j=1:size(x2,1)
% N1(1,j)=(1/2)*(x2(j,:)*Z2).^2;
% end
% f2=sum(N1)+sum(N2)+norm(Z2,2)^2;
% end

%Calculate the gradient of the loss term for each iteration of SLTSVM2
function f2_Z2=gfun2(Z2) 
S2=zeros(size(x1,2),size(x1,1));
for j=1:size(x1,1)
    if(x1(j,:)*Z2-1>0)
       a=a2;
    else
       a=a1;
    end
S2(:,j)=c*a.*x1(j,:)'.*(exp(a*(x1(j,:)*Z2-1))-1);
end
f2_Z2=sum(S2,2);
end
N=100;%Maximum number of iterations
t=1;%Initialize the number of iterations t
Z10=zeros(size(x1,2),1);%Initialize w_1=[w_+;b_+]
Z20=zeros(size(x2,2),1);%Initialize w_2=[w_-;b_-]
%Solve w_1
while(t<N)   
g1t=feval(@gfun1,Z10);   
Z1=(x1'*x1+C*eye(n1))\g1t;
    if(norm(Z1-Z10,1)<1e-8)
        break;
    end
    Z10=Z1;
    t=t+1;
end
ww1=Z1(1:(size(Z1,1)-1),1);%Obtain w_+
bb1=Z1(end,1);%Obtain b_+
t=1;
%Solve w_2
while(t<N)
g2t=feval(@gfun2,Z20);   
Z2=-(x2'*x2+C*eye(n2))\g2t;
    if(norm(Z2-Z20,1)<1e-8)
        break;
    end
    Z20=Z2;
    t=t+1;
end
ww2=Z2(1:(size(Z2,1)-1),1);%Obtain w_-
bb2=Z2(end,1);%Obtain b_-
end