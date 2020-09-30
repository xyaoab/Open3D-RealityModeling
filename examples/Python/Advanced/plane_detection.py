import open3d as o3d
import numpy as np
import argparse
import matplotlib.pyplot as plt

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
          [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]]


def normal_hist(normals,
                phi_res=(15.0 / 180.0) * np.pi,
                theta_res=(15.0 / 180) * np.pi):

    phi_bins = max(int(np.floor(2 * np.pi / phi_res)), 1)
    theta_bins = max(int(np.floor(np.pi / theta_res)), 1)

    # Reparameterize normals
    # [pi, pi] -> [0, m]
    label_phis = np.floor((np.arctan2(normals[:, 1], normals[:, 0]) + np.pi) /
                          (2 * np.pi) * phi_bins)

    # [0, pi] -> [0, n]
    label_thetas = np.floor(np.arccos(normals[:, 2]) / np.pi * theta_bins)

    hist = np.zeros((phi_bins, theta_bins))
    for k in range(len(label_phis)):
        i = int(label_phis[k])
        j = int(label_thetas[k])
        hist[i, j] += 1

    return hist, np.stack((label_phis, label_thetas))


def merge_normal_hist(normals,
                      hist,
                      labels,
                      plane_size_thr=3000,
                      merge_thr=0.95):
    shape = hist.shape
    active_bins = hist > plane_size_thr
    hist = active_bins * hist

    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    yy, xx = np.meshgrid(y, x)

    active_hist = hist[active_bins]
    active_xx = xx[active_bins]
    active_yy = yy[active_bins]
    active_dict = {}
    for i in range(len(active_hist)):
        active_dict[(active_xx[i], active_yy[i])] = active_hist[i]
    active_dict = {
        k: v for k, v in sorted(
            active_dict.items(), key=lambda item: item[1], reverse=True)
    }

    kvs = list(active_dict.items())
    for k, v in kvs:
        if k not in active_dict:
            continue

        i, j = k
        active_ij = np.logical_and(labels[0, :] == i, labels[1, :] == j)
        mean_normal_ij = np.mean(normals[active_ij, :], axis=0)

        for nbs in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
                    [1, 0], [1, 1]]:
            i_nb = (i + nbs[0] + shape[0]) % shape[0]
            j_nb = (j + nbs[1] + shape[1]) % shape[1]

            if not (i_nb, j_nb) in active_dict:
                continue

            active_ij_nb = np.logical_and(labels[0, :] == i_nb,
                                          labels[1, :] == j_nb)
            mean_normal_ij_nb = np.mean(normals[active_ij_nb, :], axis=0)

            dot_prod = np.sum(mean_normal_ij * mean_normal_ij_nb)
            if dot_prod > merge_thr:
                labels[0, active_ij_nb] = i
                labels[1, active_ij_nb] = j
                hist[i, j] += hist[i_nb, j_nb]
                hist[i_nb, j_nb] = 0

                del active_dict[(i_nb, j_nb)]

    clusters = []
    indices = np.arange(len(normals))
    for k, v in active_dict.items():
        i, j = k
        res = indices[np.logical_and(labels[0, :] == i, labels[1, :] == j)]
        clusters.append(res)

    return hist, labels, clusters


def max_hist_indices(hist, n=1):
    indices = np.unravel_index(np.argsort(-hist, axis=None), hist.shape)
    return np.stack((indices[0][:n + 1], indices[1][:n + 1])).T


def dist_hist(dists, bins=10):
    max_dist = np.max(dists)
    min_dist = np.min(dists)

    hist = np.zeros((bins, 1))
    normalized_dists = np.round(
        (dists - min_dist) / (max_dist - min_dist) * bins)

    for i in range(bins):
        hist[i, 0] = np.sum(normalized_dists == i)

    return hist, normalized_dists


def merge_dist_hist(dists, hist, labels):
    active_dict = {i: v[0] for i, v in enumerate(hist)}
    active_dict = {
        k: v for k, v in sorted(
            active_dict.items(), key=lambda item: item[1], reverse=True)
    }

    kvs = list(active_dict.items())
    for k, v in kvs:
        if k not in active_dict:
            continue

        i = k
        active_i = (labels == i)
        if np.sum(active_i) < max(0.1 * len(hist), 1000):
            break

        mean_dist_i = np.mean(dists[active_i])

        for nb in [-2, -1, 1, 2]:
            i_nb = i + nb
            if i_nb not in active_dict:
                continue
            active_i_nb = (labels == i_nb)
            mean_dist_i_nb = np.mean(dists[active_i_nb])

            if np.abs(mean_dist_i - mean_dist_i_nb) < 0.1:
                labels[active_i_nb] = i
                hist[i] += hist[i_nb]
                hist[i_nb] = 0

                del active_dict[i_nb]

    clusters = []
    indices = np.arange(len(dists))
    for k, v in active_dict.items():
        res = indices[labels == k]
        if len(res) > max(0.1 * len(hist), 1000):
            clusters.append(res)

    return hist, labels, clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd', type=str)
    args = parser.parse_args()

    voxel_size = 0.025
    radius_normal = voxel_size * 2
    pcd = o3d.io.read_point_cloud(args.pcd)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    n_hist, n_label = normal_hist(normals)
    n_hist, n_label, n_cluster = merge_normal_hist(normals,
                                                   n_hist,
                                                   n_label,
                                                   plane_size_thr=0.01 *
                                                   len(points))

    pcd_clusters = []
    for i, mask_cluster in enumerate(n_cluster):
        points_n = points[mask_cluster, :]
        normals_n = normals[mask_cluster, :]

        # pcd_b = o3d.geometry.PointCloud()
        # pcd_b.points = o3d.utility.Vector3dVector(points_n + 0.01)
        # pcd_b.normals = o3d.utility.Vector3dVector(normals_n)
        # pcd_b.paint_uniform_color(COLORS[i % len(COLORS)])
        # pcd_clusters.append(pcd_b)

        dists = np.sum(normals_n * points_n, axis=1)
        d_hist, d_label = dist_hist(dists)
        d_hist, d_label, d_cluster = merge_dist_hist(dists, d_hist, d_label)

        for d_mask in d_cluster:
            dist_cluster = dists[d_mask]
            dist_medium = np.sort(dist_cluster)[len(dist_cluster) // 2]
            d_mask_plane = np.abs(dist_cluster - dist_medium) < 0.03

            points_d = points_n[d_mask][d_mask_plane]
            normals_d = normals_n[d_mask][d_mask_plane]
            color_d = COLORS[len(pcd_clusters) % len(COLORS)]

            pcd_d = o3d.geometry.PointCloud()
            pcd_d.points = o3d.utility.Vector3dVector(points_d + 0.01)
            pcd_d.normals = o3d.utility.Vector3dVector(normals_d)
            pcd_d.paint_uniform_color(color_d)
            # pcd_clusters.append(pcd_d)

            n = np.mean(normals_d, axis=0)
            d = -np.mean(np.dot(points_d, n))

            max_x = np.max(points_d[:, 0])
            min_x = np.min(points_d[:, 0])
            max_y = np.max(points_d[:, 1])
            min_y = np.min(points_d[:, 1])
            xys = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y],
                            [max_x, max_y]]).astype(np.float64)
            zs = -(np.dot(xys, n[:2]) + d) / n[2]
            xyzs = np.concatenate((xys, np.expand_dims(zs, axis=1)), axis=1)
            tris = np.array([[0, 1, 2], [2, 1, 3]])

            mesh_d = o3d.geometry.TriangleMesh()
            mesh_d.vertices = o3d.utility.Vector3dVector(xyzs)
            mesh_d.triangles = o3d.utility.Vector3iVector(tris)
            mesh_d.paint_uniform_color(color_d)
            pcd_clusters.append(mesh_d)

            # o3d.visualization.draw_geometries([pcd_d, mesh_d])

    pcd_clusters.append(pcd)
    o3d.visualization.draw_geometries(pcd_clusters)
