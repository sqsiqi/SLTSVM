function [err,f1]=SLTSVMerror(Xv,Yv,ww1,ww2,bb1,bb2)
d1=abs(Xv*ww1+bb1)/norm(ww1);%Distance of the test sample point from the positive hyperplane
d2=abs(Xv*ww2+bb2)/norm(ww2);%Distance of the test sample point from the negative hyperplane
y=d1-d2;
y(y<0)=1;%If y<0 i.e. d1<d2, it means that the sample is closer to the positive hyperplane
y(y~=1)=-1;
preY=y;
err=sum(preY ~=Yv)/size(Xv,1);
f1 = calculate_f1_score(Yv,preY);
