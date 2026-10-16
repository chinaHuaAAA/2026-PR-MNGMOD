import numpy as np
from scipy.spatial import distance
from GB import getGranularBall
from GBFS import Feature_selection


def select_prototypes(data, prototype_ratio, random_state=None):
    """选择原型数据 - 使用基于粒球内样本数量的加权平均距离"""
    if random_state is not None:
        np.random.seed(random_state)

    n_prototypes = max(1, int(len(data) * prototype_ratio))

    # 首先生成粒球
    centers, _, sample_nums, radii, _ = getGranularBall(data)

    # 计算每个样本到所有粒球中心的加权平均距离
    avg_distances = []
    for i in range(len(data)):
        # 计算当前样本到所有粒球中心的距离
        dists = distance.cdist([data[i]], centers, 'euclidean')[0]
        weighted_avg_distance = np.sum(dists * sample_nums) / np.sum(sample_nums)
        avg_distances.append(weighted_avg_distance)

    prototype_indices = np.argsort(avg_distances)[:n_prototypes]
    return data[prototype_indices], prototype_indices


def low_noise_cosine_schedule(t, num_steps):
    """极低噪声的余弦调度 - 保证数据不被破坏"""
    s = 0.0001  # 更小的s值，减少噪声影响
    t = t / num_steps
    # 调整公式，让噪声强度极低
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    # 限制噪声范围，确保alpha_t接近1.0
    return np.clip(f / np.cos(s * np.pi / 2) ** 2, 0.95, 0.999)  # 提高下限


def MNGMOD(data, prototype_ratio, k, granular_ball_steps=5,
                      num_reconstructions=1, num_runs=1, random_state=42):
    """
    极低噪声版本的扩散过程 - 保留创新点但避免数据破坏
    """
    data = Feature_selection(data, k)

    # The schedule and selected scales do not depend on run or reconstruction.
    num_steps = 100
    alphas_cumprod = np.array([low_noise_cosine_schedule(t, num_steps)
                               for t in range(num_steps)])
    selected_steps = np.linspace(0, num_steps - 1,
                                 granular_ball_steps, dtype=int)

    if random_state is not None:
        np.random.seed(random_state)

    all_scores = []

    for run in range(num_runs):
        run_seed = random_state + run if random_state is not None else None
        if run_seed is not None:
            np.random.seed(run_seed)

        run_scores = np.zeros(len(data))

        for rec in range(num_reconstructions):
            rec_seed = run_seed + rec if run_seed is not None else None
            if rec_seed is not None:
                np.random.seed(rec_seed)

            # 选择原型数据
            prototypes, _ = select_prototypes(data, prototype_ratio)

            granular_balls = {}

            for step_idx, step in enumerate(selected_steps):
                step_seed = rec_seed + step_idx if rec_seed is not None else None
                if step_seed is not None:
                    np.random.seed(step_seed)

                alpha_t = alphas_cumprod[step]
                # 极低强度的噪声
                noise = np.random.randn(*prototypes.shape) # 减小噪声幅度
                noisy_prototypes = np.sqrt(alpha_t) * prototypes + np.sqrt(1 - alpha_t) * noise

                centers, _, sample_num, radii, _ = getGranularBall(noisy_prototypes)
                granular_balls[step] = (centers, sample_num, radii)

            reconstruction_errors = np.zeros(len(data))
            # Keep the original per-point seed and scale order. Batch only the
            # distance calculation, with a bounded number of point-center pairs.
            valid_steps = [step for step in selected_steps
                           if len(granular_balls[step][0]) > 0]
            max_pairs = 262144
            largest_ball_count = max((len(granular_balls[step][0])
                                      for step in valid_steps), default=1)
            batch_size = max(1, min(len(data), max_pairs // largest_ball_count))
            for start in range(0, len(data), batch_size):
                stop = min(start + batch_size, len(data))
                noisy_batches = [np.empty((stop - start, data.shape[1]), dtype=float)
                                 for _ in valid_steps]
                for local_i, i in enumerate(range(start, stop)):
                    point_seed = rec_seed + i if rec_seed is not None else None
                    if point_seed is not None:
                        np.random.seed(point_seed)
                    for scale_i, step in enumerate(valid_steps):
                        alpha_t = alphas_cumprod[step]
                        noise = np.random.randn(data.shape[1]) * 0.1
                        noisy_batches[scale_i][local_i] = (
                            np.sqrt(alpha_t) * data[i]
                            + np.sqrt(1 - alpha_t) * noise)

                total_error = np.zeros(stop - start)
                for noisy_points, step in zip(noisy_batches, valid_steps):
                    centers = granular_balls[step][0]
                    total_error += np.min(distance.cdist(noisy_points, centers), axis=1)
                if valid_steps:
                    reconstruction_errors[start:stop] = total_error / len(valid_steps)

            run_scores += reconstruction_errors

        run_avg_scores = run_scores / num_reconstructions
        all_scores.append(run_avg_scores)

    final_avg_scores = np.mean(all_scores, axis=0)
    return final_avg_scores


# 测试代码
if __name__ == "__main__":
    from sklearn.datasets import make_moons
