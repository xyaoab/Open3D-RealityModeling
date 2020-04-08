# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/ReconstructionSystem/register_fragments.py

import numpy as np
import open3d as o3d
import os, sys
sys.path.append("../Utility")
from file import join, get_file_list, make_clean_folder
from visualization import draw_registration_result
sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_scene
from refine_registration import multiscale_icp


def model_loader_fn():
    m = None
    device = None

    def closure():
        nonlocal m, device
        if m is None:
            import sys
            sys.path.append('FCGF')

            import torch
            from urllib.request import urlretrieve
            from model.resunet import ResUNetBN2C

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

            if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
                print('Downloading weights...')
                urlretrieve(
                    "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
                    'ResUNetBN2C-16feat-3conv.pth')
            checkpoint = torch.load('ResUNetBN2C-16feat-3conv.pth')
            m = ResUNetBN2C(1,
                            16,
                            normalize_feature=True,
                            conv1_kernel_size=3,
                            D=3)
            m.load_state_dict(checkpoint['state_dict'])

            m.eval()
            m = m.to(device)

        return m, device

    return closure


def extract_features(model,
                     pcd,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    import MinkowskiEngine as ME
    import torch
    import copy

    if is_eval:
        model.eval()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xyz = np.array(pcd.points)
    rgb = np.array(pcd.colors)

    feats = []
    feats.append(np.ones((len(xyz), 1)))
    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)
    inds = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[inds]

    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]
    return_colors = rgb[inds]

    return_pcd = o3d.geometry.PointCloud()
    return_pcd.points = o3d.utility.Vector3dVector(return_coords)
    return_pcd.colors = o3d.utility.Vector3dVector(return_colors)

    feats = feats[inds]
    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(feats, coords=coords).to(device)
    feats = model(stensor).F.detach().cpu().numpy()

    return_feats = o3d.registration.Feature()
    return_feats.resize(feats.shape[1], feats.shape[0])
    return_feats.data = copy.deepcopy(feats.T)

    return return_pcd, return_feats


def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))

    if config["feature_type"] == "fpfh":
        pcd_feature = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                                 max_nn=100))

    elif config["feature_type"] == "fcgf":
        model_loader = model_loader_fn()
        model, device = model_loader()

        pcd_down, pcd_feature = extract_features(model,
                                                 pcd=pcd_down,
                                                 voxel_size=voxel_size,
                                                 device=device,
                                                 skip_check=True)

    else:
        raise NotImplementedError('Unsupported feature type {}'.format(
            config['feature_type']))

    return (pcd_down, pcd_feature)


def register_point_cloud_feature(source, target, source_feature, target_feature,
                                 config):
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source, target, source_feature, target_feature,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source, target, source_feature, target_feature, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))
    information = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))
    return (True, result.transformation, information)


def compute_initial_registration(s, t, source_down, target_down, source_feature,
                                 target_feature, path_dataset, config):

    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        pose_graph_frag = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % s))
        n_nodes = len(pose_graph_frag.nodes)
        transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes -
                                                                  1].pose)
        (transformation, information) = \
                multiscale_icp(source_down, target_down,
                [config["voxel_size"]], [50], config, transformation_init)
    else:  # loop closure case
        (success, transformation,
         information) = register_point_cloud_feature(source_down, target_down,
                                                     source_feature,
                                                     target_feature, config)
        if not success:
            print("No resonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))
    print(transformation)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)
    return (True, transformation, information)


def update_posegrph_for_scene(s, t, transformation, information, odometry,
                              pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(s,
                                           t,
                                           transformation,
                                           information,
                                           uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(s,
                                           t,
                                           transformation,
                                           information,
                                           uncertain=True))
    return (odometry, pose_graph)


def register_point_cloud_pair(ply_file_names, s, t, config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])
    (source_down, source_feature) = preprocess_point_cloud(source, config)
    (target_down, target_feature) = preprocess_point_cloud(target, config)
    (success, transformation, information) = \
            compute_initial_registration(
            s, t, source_down, target_down,
            source_feature, target_feature, config["path_dataset"], config)
    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.infomation = np.identity(6)


def make_posegraph_for_scene(ply_file_names, config):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t)

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(matching_results), 1))
        results = Parallel(n_jobs=MAX_THREAD)(delayed(
            register_point_cloud_pair)(ply_file_names, matching_results[r].s,
                                       matching_results[r].t, config)
                                              for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
                    matching_results[r].information) = \
                    register_point_cloud_pair(ply_file_names,
                    matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegrph_for_scene(
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation,
                matching_results[r].information, odometry, pose_graph)
    o3d.io.write_pose_graph(
        join(config["path_dataset"], config["template_global_posegraph"]),
        pose_graph)


def run(config):
    print("register fragments.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")
    make_clean_folder(join(config["path_dataset"], config["folder_scene"]))
    make_posegraph_for_scene(ply_file_names, config)
    optimize_posegraph_for_scene(config["path_dataset"], config)
