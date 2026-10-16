import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import os
import time
import importlib
import pandas as pd
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接目标路径
target_path = os.path.join(current_dir, 'OD')
sys.path.append(target_path)

# Windows uses spawn: each process imports this module and initializes its own
# copy of the dataset once. Never seed multiple threads against NumPy's global RNG.
_worker_data = None
_worker_method = None


def _init_worker(data, method_name):
    global _worker_data, _worker_method
    _worker_data = data
    module = importlib.import_module(method_name)
    _worker_method = getattr(module, method_name)


def _single_repeat(task):
    p1, p2, seed = task
    start = time.perf_counter()
    scores = _worker_method(_worker_data, p1, p2, random_state=seed)
    return np.asarray(scores, dtype=float).reshape(-1), time.perf_counter() - start


def calculate_metrics(labels, scores):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("异常分数长度不匹配或包含NaN/Inf")
    count = max(1, int(np.ceil(len(labels) * PREDICTED_OUTLIER_RATIO)))
    predictions = np.zeros(len(labels), dtype=int)
    predictions[np.argsort(-scores, kind='stable')[:count]] = 1
    return dict(AUC=roc_auc_score(labels, scores),
                AP=average_precision_score(labels, scores),
                Precision=precision_score(labels, predictions, zero_division=0),
                Recall=recall_score(labels, predictions, zero_division=0))


def parallel_repeat_scores(executor, p1, p2, repeats=10, base_seed=42):
    # executor.map preserves input order: row j corresponds to seed base_seed+j.
    results = list(executor.map(
        _single_repeat, [(p1, p2, base_seed + i) for i in range(repeats)]))
    return np.stack([r[0] for r in results], axis=0), [r[1] for r in results]


PREDICTED_OUTLIER_RATIO = 0.05
BASE_SEED = 42
METRICS = ('AUC', 'AP', 'Precision', 'Recall')


def sort_files_by_size(folder_path, sort_by='size'):
    """
    按不同方式排序文件
    :param folder_path: 文件夹路径
    :param sort_by: 排序方式 'size'-按大小, 'name'-按文件名, 'time'-按修改时间, 'none'-不排序
    :return: 排序后的文件列表
    """
    # 获取文件夹中所有文件的路径
    all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, file))]

    if sort_by == 'none':
        # 不排序，保持原始顺序
        return [(f, 0) for f in all_files]

    elif sort_by == 'name':
        # 按文件名排序
        sorted_files = sorted(all_files, key=lambda x: os.path.basename(x))
        return [(f, 0) for f in sorted_files]

    elif sort_by == 'time':
        # 按修改时间排序
        files_with_time = [(f, os.path.getmtime(f)) for f in all_files]
        sorted_files = sorted(files_with_time, key=lambda x: x[1])
        return sorted_files

    elif sort_by == 'size':
        # 按文件大小排序
        files_with_size = [(f, os.path.getsize(f)) for f in all_files]
        sorted_files = sorted(files_with_size, key=lambda x: x[1])
        return sorted_files

    else:
        # 默认按大小排序
        files_with_size = [(f, os.path.getsize(f)) for f in all_files]
        sorted_files = sorted(files_with_size, key=lambda x: x[1])
        return sorted_files


def check_if_processed(file_path, result_path, F_name):
    """Only skip results that already contain every parameter's repeated scores."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    pkl_file = os.path.join(result_path, F_name, base_name + '.pkl')
    excel_file = os.path.join(result_path, F_name, base_name + '.xlsx')
    if not (os.path.exists(pkl_file) and os.path.exists(excel_file)):
        return False
    try:
        with open(pkl_file, 'rb') as handle:
            result = pickle.load(handle)
        complete = (isinstance(result, dict) and result.get('ResultVersion') == 2
                    and result.get('Aggregation') == 'mean_of_per_run_metrics'
                    and result.get('Seeds') == list(range(BASE_SEED, BASE_SEED + REPEATS))
                    and result.get('PredictedOutlierRatio') == PREDICTED_OUTLIER_RATIO
                    and len(result.get('Params', [])) == len(result.get('OF_runs', [])) > 0
                    and all(np.asarray(r).ndim == 2 and len(r) == REPEATS
                            for r in result['OF_runs']))
        if complete:
            print(f"数据集 {base_name} 已处理过，跳过...")
            return True
    except Exception:
        pass
    return False


def alternating_parameter_search(OD, trandata, true_labels, param1_range, param2_range,
                                 first_fixed_value=None, first_fixed_param='p1'):
    """Original two-stage search, using mean per-run AUC to select parameters."""
    records = []

    def evaluate(p1, p2):
        start = time.perf_counter()
        runs, times = OD(trandata, p1, p2)
        wall_time = time.perf_counter() - start
        runs = np.asarray(runs, dtype=float)
        if runs.ndim != 2 or runs.shape != (REPEATS, len(true_labels)):
            raise ValueError('每组参数必须返回[重复次数, 样本数]的完整分数矩阵')
        values = [calculate_metrics(true_labels, scores) for scores in runs]
        record = dict(Params=(p1, p2), OF_runs=runs, OF=np.mean(runs, axis=0),
                      Time=wall_time, Time_runs=list(times))
        for metric in METRICS:
            individual = [value[metric] for value in values]
            record[metric + '_runs'] = individual
            record[metric] = float(np.mean(individual))
            record[metric + '_std'] = float(np.std(individual, ddof=1)) if len(individual) > 1 else np.nan
        records.append(record)
        print(f"p1={p1:.2f}, p2={p2:.2f}, AUC={record['AUC']:.4f}")
        return record

    if first_fixed_param == 'p1':
        print(f"第一步：固定param1={first_fixed_value:.2f}，寻找最优param2")
        first = [evaluate(first_fixed_value, p2) for p2 in param2_range]
        best_p2 = max(first, key=lambda r: r['AUC'])['Params'][1]
        print(f"第二步：固定param2={best_p2:.2f}，寻找最优param1")
        second = [evaluate(p1, best_p2) for p1 in param1_range]
    elif first_fixed_param == 'p2':
        print(f"第一步：固定param2={first_fixed_value:.2f}，寻找最优param1")
        first = [evaluate(p1, first_fixed_value) for p1 in param1_range]
        best_p1 = max(first, key=lambda r: r['AUC'])['Params'][0]
        print(f"第二步：固定param1={best_p1:.2f}，寻找最优param2")
        second = [evaluate(best_p1, p2) for p2 in param2_range]
    else:
        raise ValueError("first_fixed_param必须为'p1'或'p2'")
    best = max(second, key=lambda r: r['AUC'])
    return records, (*best['Params'], best['AUC'])


if __name__ == '__main__':
    # 算法名称
    F_name = 'MNGMOD'
    result_path = 'Result'
    REPEATS = 10
    MAX_WORKERS = min(4, os.cpu_count() or 1, REPEATS)

    # 确保结果目录存在
    os.makedirs(os.path.join(result_path, F_name), exist_ok=True)

    module = importlib.import_module(F_name)
    OD = getattr(module, F_name)
    sort_files = sort_files_by_size(os.path.join(current_dir, "Data"), "size")
    sort_files = sort_files[0:]

    # 在这里设置第一次优化时固定的参数值和参数名称
    FIRST_FIXED_VALUE = 1  # 第一次优化时固定的参数值
    FIRST_FIXED_PARAM = 'p2'  # 'p1'或'p2'，表示第一次优化时固定哪个参数

    for file in sort_files:
        file_path = file[0]

        # 检查是否已经处理过
        if check_if_processed(file_path, result_path, F_name):
            continue

        try:
            mat_data = sio.loadmat(file_path)  # 读取数据
            data = mat_data['trandata']
            trandata = data[:, :-1]
            trandata = trandata[:, ~np.all((trandata == 0) | (trandata == 1), axis=0)]
            scaler = MinMaxScaler()
            trandata = scaler.fit_transform(trandata)
            true_labels = data[:, -1]

            print("-----------", file_path, "-----------\n")



            # Both ratios are in (0, 1]. A zero prototype ratio yields one
            # prototype, which cannot be split into granular balls; zero
            # feature ratio yields an empty feature subset.
            param1 = np.arange(0.05, 1.05, 0.05)
            param2 = np.arange(0.05, 1.05, 0.05)
            # 使用交替参数搜索策略
            with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                     initializer=_init_worker,
                                     initargs=(trandata, F_name)) as executor:
                def repeated_OD(data, p1, p2):
                    return parallel_repeat_scores(executor, p1, p2, REPEATS, BASE_SEED)

                records, (best_p1, best_p2, best_auc) = alternating_parameter_search(
                    repeated_OD, trandata, true_labels, param1, param2,
                    first_fixed_value=FIRST_FIXED_VALUE,
                    first_fixed_param=FIRST_FIXED_PARAM
                )

            # All arrays align by parameter-history index; scores align with original sample rows.
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            data_dict = dict(
                ResultVersion=2, Aggregation='mean_of_per_run_metrics',
                Params=[r['Params'] for r in records],
                Seeds=list(range(BASE_SEED, BASE_SEED + REPEATS)),
                OF=[r['OF'] for r in records],  # Convenience only; not used to compute reported metrics.
                OF_runs=[r['OF_runs'] for r in records],
                Time=[r['Time'] for r in records],  # Original parallel wall-time convention.
                Time_runs=[r['Time_runs'] for r in records],
                Labels=np.asarray(true_labels),
                PredictedOutlierRatio=PREDICTED_OUTLIER_RATIO,
                BestParams=(best_p1, best_p2), BestAUC=best_auc)
            excel_rows = []
            for i, record in enumerate(records):
                row = dict(Par=record['Params'], Time=record['Time'])
                for metric in METRICS:
                    row[metric] = record[metric]
                    row[metric + '_std'] = record[metric + '_std']
                excel_rows.append(row)
            for metric in METRICS:
                data_dict[metric] = [r[metric] for r in records]
                data_dict[metric + '_mean'] = list(data_dict[metric])
                data_dict[metric + '_std'] = [r[metric + '_std'] for r in records]
                data_dict[metric + '_runs'] = [r[metric + '_runs'] for r in records]
            df = pd.DataFrame(excel_rows)
            df.to_excel(os.path.join(result_path, F_name, base_name + '.xlsx'), index=False)
            with open(os.path.join(result_path, F_name, base_name + '.pkl'), 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 显示最优解
            print(f"{file_path} 最优参数: p1={best_p1:.2f}, p2={best_p2:.2f}, Max_AUC={best_auc:.4f}\n")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue


def Pt(y_true, y_pred):
    """
    复现论文中的P(t)计算
    :param y_true: 真实标签（1=异常，0=正常）
    :param y_pred: 模型输出的异常分数（越高越异常）
    :return: P(t)值
    """
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算真实异常数量t
    t = np.sum(y_true == 1)

    # 按预测分数降序排序，获取索引
    sorted_indices = np.argsort(-y_pred)

    # 获取前t个预测异常的样本标签
    top_t_predictions = y_true[sorted_indices[:t]]

    # 计算OS(t) ∩ OT（预测正确数）
    true_positives = np.sum(top_t_predictions == 1)

    # 计算P(t)
    Pt = true_positives / t

    return Pt
