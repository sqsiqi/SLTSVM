function [err,f1]=KSLTSVMerror(Xv,Yv,ww1,ww2,bb1,bb2)
w1=[ww1;bb1];
w2=[ww2;bb2];
d1=abs(Xv*w1);
d2=abs(Xv*w2);
y=d1-d2;
y(y<0)=1;
y(y~=1)=-1;
preY=y;
err=sum(preY~=Yv)/size(Xv,1);
f1 = calculate_f1_score(Yv,preY);


