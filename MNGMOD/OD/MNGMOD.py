import numpy as np
from scipy.spatial import distance
from GB import getGranularBall
from GBFS import Feature_selection


def select_prototypes(data, prototype_ratio, random_state=None):
    """Select samples with the smallest ball-size-weighted mean distances.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Input feature matrix.
    prototype_ratio : float
        Fraction of samples to select, in (0, 1].
    random_state : int or None, default=None
        NumPy random seed; None leaves the random state unchanged.

    Returns
    -------
    prototypes : ndarray of shape (n_prototypes, n_features)
        Selected samples, ordered by increasing weighted mean distance.
    prototype_indices : ndarray of shape (n_prototypes,)
        Indices of the selected samples in data.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_prototypes = max(1, int(len(data) * prototype_ratio))


    centers, _, sample_nums, radii, _ = getGranularBall(data)


    avg_distances = []
    for i in range(len(data)):

        dists = distance.cdist([data[i]], centers, 'euclidean')[0]
        weighted_avg_distance = np.sum(dists * sample_nums) / np.sum(sample_nums)
        avg_distances.append(weighted_avg_distance)

    prototype_indices = np.argsort(avg_distances)[:n_prototypes]
    return data[prototype_indices], prototype_indices


def low_noise_cosine_schedule(t, num_steps):
    """Return the clipped cosine-schedule coefficient for a noise step.

    Parameters
    ----------
    t : int or float
        Noise-step index.
    num_steps : int
        Positive number of noise steps.

    Returns
    -------
    alpha_t : float
        Schedule coefficient clipped to [0.95, 0.999].
    """
    s = 0.0001
    t = t / num_steps

    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

    return np.clip(f / np.cos(s * np.pi / 2) ** 2, 0.95, 0.999)


def MNGMOD(data, prototype_ratio, k, granular_ball_steps=5,
                      num_reconstructions=1, num_runs=1, random_state=42):
    """Compute MNGMOD anomaly scores for an input feature matrix.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Numeric feature matrix without labels.
    prototype_ratio : float
        Fraction of samples used as manifold skeleton prototypes, in (0, 1].
    k : float
        Attribute-selection ratio passed to Feature_selection, in (0, 1].
    granular_ball_steps : int, default=5
        Number of noise scales selected from the 100-step schedule (1 to 100).
    num_reconstructions : int, default=1
        Positive number of reference reconstructions per internal run.
    num_runs : int, default=1
        Positive number of internal runs whose score vectors are averaged.
    random_state : int or None, default=42
        Base NumPy random seed; None disables explicit seeding.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Anomaly scores in input row order, averaged across noise scales,
        reconstructions, and internal runs. Larger values indicate anomalies.
    """
    data = Feature_selection(data, k)


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


            prototypes, _ = select_prototypes(data, prototype_ratio)

            granular_balls = {}

            for step_idx, step in enumerate(selected_steps):
                step_seed = rec_seed + step_idx if rec_seed is not None else None
                if step_seed is not None:
                    np.random.seed(step_seed)

                alpha_t = alphas_cumprod[step]

                noise = np.random.randn(*prototypes.shape)
                noisy_prototypes = np.sqrt(alpha_t) * prototypes + np.sqrt(1 - alpha_t) * noise

                centers, _, sample_num, radii, _ = getGranularBall(noisy_prototypes)
                granular_balls[step] = (centers, sample_num, radii)

            reconstruction_errors = np.zeros(len(data))


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


if __name__ == "__main__":
    from sklearn.datasets import make_moons
