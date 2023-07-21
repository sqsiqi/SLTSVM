function f1_score = calculate_f1_score(y_true, y_pred)

%计算混淆矩阵
tp = sum((y_true == 1) & (y_pred == 1));
fp = sum((y_true == 1) & (y_pred == -1));
fn = sum((y_true == -1) & (y_pred == 1));
tn = sum((y_true == -1) & (y_pred == -1));

%计算F1得分
f1_score = 2*tp/(2*tp+fn+fp);

end