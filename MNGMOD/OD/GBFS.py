import numpy as np
from scipy.spatial.distance import cdist
from AttributeGB import getAttributeGranularBall
from scipy.io import loadmat
import math

import numpy as np
import math
from scipy.stats import entropy


def Feature_selection(data, select_ratio):
    """
    简化版特征选择：使用熵计算重要性并全局排序后按比例选择
    """
    Gbs = getAttributeGranularBall(data.T)
    total_features = data.shape[1]
    target_num = math.ceil(total_features * select_ratio)

    # 计算每个特征的重要性（熵值）
    feature_importance = []

    for i in range(len(Gbs[2])):
        features_in_gb = Gbs[1][i][:, -1].astype(int)
        gb_data = Gbs[1][i][np.isin(Gbs[1][i][:, -1], features_in_gb)]

        for j, feature_id in enumerate(gb_data[:, -1].astype(int)):
            feature_values = gb_data[j, :-1]

            # 计算熵
            value_counts = np.bincount(feature_values.astype(int))
            value_counts = value_counts[value_counts > 0]
            probabilities = value_counts / len(feature_values)
            entropy_value = entropy(probabilities, base=2)

            feature_importance.append((feature_id, entropy_value))

    # 按重要性降序排序
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # 提取特征索引并选择前N个
    sorted_indices = [feat[0] for feat in feature_importance]
    selected_indices = sorted_indices[:target_num]

    # 返回结果
    return data.iloc[:, selected_indices] if hasattr(data, 'iloc') else data[:, selected_indices]
if __name__ == '__main__':
    file_path = 'mydata/glass.mat'
    # 打开并读取.pkl文件
    mat_data = loadmat(file_path)
    data = mat_data['trandata']
    data = data[:,:-1]