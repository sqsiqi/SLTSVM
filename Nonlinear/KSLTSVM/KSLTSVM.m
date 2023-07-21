function [uu1,bb1,uu2,bb2]=KSLTSVM(c,C,X,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KSLTSVM: Symmetric LINEX loss twin support vector machine (Nonlinear version)
% Use method: Import the data first, and then run the "meanKSLTSVMtest" procedure
% Input:
%    X: Training data.
%    Y: Training data labels. (Y must include 1 and -1)
% Parameters:
% c: The loss term regularization parameter
% C: The structural risk term regularization parameter
% a1: The LINEX loss function parameter 
% a2: The LINEX loss function parameter  
% Output:
% uu1: The positive hypersurface parameter u_+
% uu2: The negative hypersurface parameter u_-
% bb1: The positive hypersurface parameter b_+
% bb2: The negative hypersurface parameter b_-
% fun1 and fun2 are used to analyze the convergence of the objective function
% Reference:
%    Qi Si, Zhi-Xia Yang, Jun-You Ye, et. al. "Symmetric LINEX loss twin support vector machine for robust
%    classification and its fast iterative algorithm"  Submitted 2023
%
%    Written by Qi Si (1418426889@qq.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1=1;
a2=-1;
A=[X,Y];
KAC=A(A(:,end)==1,1:end-1);
KBC=A(A(:,end)==-1,1:end-1);
n1=size(KAC,2);
n2=size(KBC,2);
n3=size(KAC,1);
n4=size(KBC,1);
% function f1=fun1(Z1)
% M1=zeros(1,size(KAC,1));
% M2=zeros(1,size(KBC,1));
% for j=1:size(KAC,1)
% M1(1,j)=(1/2)*(KAC(j,:)*Z1).^2;
% end
% for j=1:size(KBC,1)
%      if(-(KBC(j,:)*Z1+1)>0)
%         a=a2;
%      else
%         a=a1;
%      end
% M2(1,j)=c*(exp(-a*(KBC(j,:)*Z1+1))+a*(KBC(j,:)*Z1+1)-1);
% end
% f1=sum(M1)+sum(M2)+norm(Z1,2);
% end

function f1_Z1=gfun1(Z1)
R2=zeros(size(KBC,2),size(KBC,1));
for j=1:size(KBC,1)
      if(-(KBC(j,:)*Z1+1)>0)
         a=a2;
       else
         a=a1;
       end
R2(:,j)=-c*a.*KBC(j,:)'.*(exp(-a*(KBC(j,:)*Z1+1))-1);
end
f1_Z1=sum(R2,2);
end

% function f2=fun2(Z2)
% N2=zeros(1,size(KAC,1));
% N1=zeros(1,size(KBC,1));
% for j=1:size(KAC,1)
%        if(KAC(j,:)*Z2-1>0)
%          a=a2;
%          else
%          a=a1;
%        end
% N2(1,j)=c*(exp(a*(KAC(j,:)*Z2-1))-a*(KAC(j,:)*Z2-1)-1);
% end
% for j=1:size(KBC,1)
% N1(1,j)=(1/2)*(KBC(j,:)*Z2).^2;
% end
% f2=sum(N1)+sum(N2)+norm(Z2,2)^2;
% end

function f2_Z2=gfun2(Z2) 
S2=zeros(size(KAC,2),size(KAC,1));
for j=1:size(KAC,1)
      if(KAC(j,:)*Z2-1>0)
        a=a2;
       else
        a=a1;
      end
S2(:,j)=c*a.*KAC(j,:)'.*(exp(a*(KAC(j,:)*Z2-1))-1);
end
f2_Z2=sum(S2,2);
end
N=50;
t=1;
Z10=zeros(size(KAC,2),1);
Z20=zeros(size(KBC,2),1);
% f1=[];%To store the objective function of KSLTSVM1 at each iteration
% f2=[];%To store the objective function of KSLTSVM2 at each iteration
while(t<N)   
 g1t=feval(@gfun1,Z10);   
 Z1=1/C*(eye(n1)-KAC'*((C*eye(n3)+KAC*KAC')\KAC))*g1t;
    if(norm(Z1-Z10)<1e-8)
        break;
    end
    %f1=[f1,feval(@fun1,Z1)]; 
    Z10=Z1;
    t=t+1;
end
uu1=Z1(1:(size(Z1,1)-1),1);
bb1=Z1(end,1);
t=1;
while(t<N)
    g2t=feval(@gfun2,Z20);   
    Z2=-1/C*(eye(n2)-KBC'*((C*eye(n4)+KBC*KBC')\KBC))*g2t;
    if(norm(Z2-Z20)<1e-8)
        break;
    end
    %f2=[f2,feval(@fun2,Z2)]; 
    Z20=Z2;
    t=t+1;
end
uu2=Z2(1:(size(Z2,1)-1),1);
bb2=Z2(end,1);
end


