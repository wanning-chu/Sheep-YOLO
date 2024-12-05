import numpy as np

# 输入混淆矩阵
confusion_matrix = np.array([
    [48, 1, 0, 0, 1, 0, 0, 0, 2],
    [1, 156, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 57, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 148, 7, 0, 0, 3, 7],
    [0, 0, 0, 4, 21, 0, 0, 1, 3],
    [0, 0, 0, 0, 0, 108, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 56, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 41, 3],
    [0, 0, 0, 1, 0, 0, 2, 0, 0],
])

# 排除"background"类的行和列
# 获取除最后一行和最后一列之外的所有行和列
confusion_matrix_no_bg = confusion_matrix[:-1, :-1]

# 2-TP/FP/FN的计算（不需要计算TN，因为TN在多类别中通常不直接计算）
FP = confusion_matrix_no_bg.sum(axis=0) - np.diag(confusion_matrix_no_bg)
FN = confusion_matrix_no_bg.sum(axis=1) - np.diag(confusion_matrix_no_bg)
TP = np.diag(confusion_matrix_no_bg)

# 将这些值转换为浮点数以进行后续计算
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)

# 3-其他的性能参数的计算
TPRs = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
PPVs = TP / (TP + FP)  # Precision/ positive predictive value
FNRs = FN / (TP + FN)  # False negative rate
FDRs = FP / (TP + FP)  # False discovery rate

# 计算全局准确率（不包括"background"类）
total_correct = TP.sum()
total_samples_no_bg = confusion_matrix_no_bg.sum()
ACC = total_correct / total_samples_no_bg  # accuracy of the model (excluding background class)

# 打印结果
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TPRs:", TPRs)
print("PPVs:", PPVs)
print("FNRs:", FNRs)
print("FDRs:", FDRs)
print("ACC (excluding background class):", ACC)