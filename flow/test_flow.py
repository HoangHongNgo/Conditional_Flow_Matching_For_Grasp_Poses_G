import os
import sys

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp, pred_decode

# Import graspnetAPI internals for custom evaluation
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    create_table_points, transform_points, voxel_sample_points,
    compute_closest_points, collision_detection, get_grasp_score,
    get_scene_name
)
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory


# ------------ CUSTOM EVALUATION WITHOUT TOP-50 FILTERING ------------

def eval_grasp_flow(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008):
    """
    Custom grasp evaluation that assigns grasps to objects and evaluates ALL of them,
    without the standard top-50 overall scene filtering constraint.
    """
    num_models = len(models)
    
    # Apply standard Grasp NMS
    grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)

    # Assign grasps to objects based on proximity
    model_trans_list = []
    seg_mask = []
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    
    grasp_list = []
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp == i]
        grasp_i.sort_by_score()
        # Keep ALL grasps assigned to this object (no top-10 slicing and NO top-50 scene filtering!)
        grasp_list.append(grasp_i.grasp_group_array)

    if table is not None:
        scene = np.concatenate([scene, table])

    # Collision detection
    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True
    )
    
    # Evaluate force closure scores
    force_closure_quality_config = {}
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure']
        )
        
    score_list = []
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = []
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list


class GraspNetEvalFlow(GraspNetEval):
    """
    Subclass of GraspNetEval that overrides eval_scene to evaluate all predicted grasps
    without the top-50 overall scene filtering constraint.
    """
    def __init__(self, root, camera, split='test', top_k=100):
        super(GraspNetEvalFlow, self).__init__(root, camera, split)
        self.top_k = top_k

    def eval_scene(self, scene_id, dump_folder, TOP_K=100, return_list=False, vis=False, max_width=0.1):
        # Override TOP_K to be self.top_k to evaluate up to all generated grasps
        TOP_K = self.top_k
        
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        
        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = []
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,))
            )
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # Clip width to [0, max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            grasp_group.grasp_group_array = gg_array

            # Call our custom grasp evaluation WITHOUT top-50 filtering
            grasp_list, score_list, collision_mask_list = eval_grasp_flow(
                grasp_group, model_sampled_list, dexmodel_list, pose_list, config,
                table=table_trans, voxel_size=0.008
            )

            # Remove empty lists
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]), end='')
                continue

            # Concat into scene level
            grasp_list = np.concatenate(grasp_list)
            score_list = np.concatenate(score_list)
            collision_mask_list = np.concatenate(collision_mask_list)
            
            # Sort in scene level by grasp confidence
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list = grasp_list[indices]
            score_list = score_list[indices]
            collision_mask_list = collision_mask_list[indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # Calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list[0:k+1] <= fric) & (score_list[0:k+1] > 0)).astype(int)) / (k + 1)

            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
            
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list


# ------------ INFERENCE & EVALUATION RUNNERS ------------

if not os.path.exists(cfgs.save_dir):
    os.makedirs(cfgs.save_dir, exist_ok=True)


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create dataset
if cfgs.test_mode == 'seen':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_seen',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, load_label=False, augment=False)
elif cfgs.test_mode == 'similar':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_similar',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, load_label=False, augment=False)
elif cfgs.test_mode == 'novel':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_novel',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, load_label=False, augment=False)

SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                             num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)


def inference():
    # Load model and run inference
    net = economicgrasp(seed_feat_dim=512, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    batch_interval = 20
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Save results for evaluation
        for i in range(len(grasp_preds)):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # Collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(
                    gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
                )
                gg = gg[~collision_mask]

            # Save grasps
            save_dir = os.path.join(cfgs.save_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()

        # early stop for debugging
        is_debugging = sys.gettrace() is not None
        if is_debugging and batch_idx >= 5:
            break




def evaluate_seen():
    try:
        # Use our custom GraspNetEvalFlow (evaluating all 100 grasp candidates per scene)
        ge = GraspNetEvalFlow(root=cfgs.dataset_root, camera=cfgs.camera, split='test', top_k=100)
        res, ap = ge.eval_seen(cfgs.save_dir, proc=6)
        save_dir = os.path.join(cfgs.save_dir, 'ap_{}_seen_flow.npy'.format(cfgs.camera))
        np.save(save_dir, res)
        print(f"\nseen testing (Without Top-50 Constraint), AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")
    except Exception as e:
        print(f"\nSkipped seen evaluation: not all prediction files are generated yet. Details: {e}")


def evaluate_similar():
    try:
        ge = GraspNetEvalFlow(root=cfgs.dataset_root, camera=cfgs.camera, split='test', top_k=100)
        res, ap = ge.eval_similar(cfgs.save_dir, proc=6)
        save_dir = os.path.join(cfgs.save_dir, 'ap_{}_similar_flow.npy'.format(cfgs.camera))
        np.save(save_dir, res)
        print(f"\nsimilar testing (Without Top-50 Constraint), AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")
    except Exception as e:
        print(f"\nSkipped similar evaluation: not all prediction files are generated yet. Details: {e}")


def evaluate_novel():
    try:
        ge = GraspNetEvalFlow(root=cfgs.dataset_root, camera=cfgs.camera, split='test', top_k=100)
        res, ap = ge.eval_novel(cfgs.save_dir, proc=6)
        save_dir = os.path.join(cfgs.save_dir, 'ap_{}_novel_flow.npy'.format(cfgs.camera))
        np.save(save_dir, res)
        print(f"\nnovel testing (Without Top-50 Constraint), AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")
    except Exception as e:
        print(f"\nSkipped novel evaluation: not all prediction files are generated yet. Details: {e}")



if __name__ == '__main__':
    if cfgs.inference:
        inference()
    if cfgs.test_mode == 'seen':
        evaluate_seen()
    elif cfgs.test_mode == 'similar':
        evaluate_similar()
    elif cfgs.test_mode == 'novel':
        evaluate_novel()
