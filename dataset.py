import os
import numpy as np
import torch
import torch.utils.data
import open3d as o3d
import numpy as _np
try:
    from .topology import compute_topo_features
except Exception:
    try:
        from topology import compute_topo_features
    except Exception:
        compute_topo_features = None


class Lung250MDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = True if phase == 'train' else False
        self.split = split

        if self.split == 'train':
            self.pcd_template = os.path.join(args.cloudfolder_train, 'case_{:03d}_{}.pth')
            self.gt_template = os.path.join(args.supfolder_train, 'case_{:03d}.pth')
        else:
            self.pcd_template = os.path.join(args.cloudfolder_val, 'case_{:03d}_{}.pth')
            self.gt_template = os.path.join(args.supfolder_val, 'case_{:03d}.pth')
        self.idx_16k = torch.load('../ind_16384_train.pth', map_location='cpu')

        if split == 'train':
            val_cases = np.array([2, 8, 54, 55, 56, 94, 97])
            self.case_list = np.arange(104)
            self.case_list = self.case_list[~np.isin(self.case_list, val_cases)]
        elif split == 'val':
            self.case_list = np.array([2, 8, 94, 97])
        else:
            raise NotImplementedError()

        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.augm_setting = cfg.AUGMENTATIONS
        
        # Check if topology coupling is enabled
        self.use_topo = cfg.MODEL.TOPO_FEAT_DIM > 0
        self.topo_dim = cfg.MODEL.TOPO_FEAT_DIM

    def __getitem__(self, idx):
        # load input pcds
        case = self.case_list[idx]
        pcd_tgt = torch.load(self.pcd_template.format(case, 1))[2]
        pcd_src = torch.load(self.pcd_template.format(case, 2))[2]
        idx_16k_tgt = self.idx_16k['all_ind_fix'][case]
        idx_16k_src = self.idx_16k['all_ind_mov'][case]
        pcd_tgt = pcd_tgt[idx_16k_tgt].float().numpy()
        pcd_src = pcd_src[idx_16k_src].float().numpy()
        corrfield_flow = torch.load(self.gt_template.format(case))['cloud_gt_mov']
        corrfield_flow = corrfield_flow[idx_16k_src].float().numpy()
        lm_src = pcd_src.copy()
        lm_tgt = corrfield_flow + lm_src

        # prealignment
        mean_tgt = np.mean(pcd_tgt, axis=0)
        std_tgt = np.std(pcd_tgt, axis=0)
        mean_src = np.mean(pcd_src, axis=0)
        std_src = np.std(pcd_src, axis=0)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        lm_src = (lm_src - mean_src) * std_tgt / std_src + mean_tgt

        # mean center and scale
        mean = np.mean(pcd_tgt, axis=0)
        pcd_tgt = (pcd_tgt - mean) / self.norm_factor
        pcd_src = (pcd_src - mean) / self.norm_factor
        lm_tgt = (lm_tgt - mean) / self.norm_factor
        lm_src = (lm_src - mean) / self.norm_factor
        gt_flow = lm_tgt - lm_src

        # compute topology features
        topo_feat_src = None
        topo_feat_tgt = None
        if self.use_topo and compute_topo_features is not None:
            try:
                topo_feat_src = compute_topo_features(pcd_src)
                topo_feat_tgt = compute_topo_features(pcd_tgt)
                # Validate output dimensions
                if topo_feat_src.shape[0] != self.topo_dim or topo_feat_tgt.shape[0] != self.topo_dim:
                    print(f"WARNING: Unexpected topology dim. Expected {self.topo_dim}, got src:{topo_feat_src.shape}, tgt:{topo_feat_tgt.shape}")
                    topo_feat_src = np.zeros(self.topo_dim, dtype=np.float32)
                    topo_feat_tgt = np.zeros(self.topo_dim, dtype=np.float32)
            except Exception as e:
                print(f"Topology extraction failed for case {case}: {e}")
                topo_feat_src = np.zeros(self.topo_dim, dtype=np.float32)
                topo_feat_tgt = np.zeros(self.topo_dim, dtype=np.float32)
        
        # Default to zeros if not computed or disabled
        if topo_feat_src is None:
            topo_feat_src = np.zeros(self.topo_dim if self.use_topo else 1, dtype=np.float32)
        if topo_feat_tgt is None:
            topo_feat_tgt = np.zeros(self.topo_dim if self.use_topo else 1, dtype=np.float32)

        if self.is_train:
            if self.augm_setting.METHOD == 'multiscale_local_global':
                if np.random.uniform() < 0.5:
                    pcd = pcd_src
                    feat_for_augm = topo_feat_src
                else:
                    pcd = pcd_tgt
                    feat_for_augm = topo_feat_tgt

                setting = self.augm_setting
                num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
                max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
                kernel_std_local = setting.KERNEL_STD_LOCAL
                global_grid_spacing = setting.GLOBAL_GRID_SPACING
                max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
                kernel_std_global = setting.KERNEL_STD_GLOBAL

                local_control_idx = np.random.permutation(pcd.shape[0])[:num_control_points_local]
                local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
                local_control_pts = pcd[local_control_idx]
                sq_dist = np.sum(np.square(pcd[:, None] - local_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_local ** 2)
                local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
                local_pcd_shifts = np.nan_to_num(local_pcd_shifts)
                pcd_augm = pcd + local_pcd_shifts

                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
                o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(global_grid_spacing,
                                                                        min_bound=np.array([-10., -10., -10.]),
                                                                        max_bound=np.array([10., 10., 10.]))

                global_control_pts = np.float32(np.asarray(o3d_cloud.points))
                global_control_shifts = np.random.uniform(-1, 1., (
                global_control_pts.shape[0], 3)) * max_control_shift_global
                sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
                global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(
                    weights[:, :, None], axis=1)

                pcd_augm = pcd_augm + global_pcd_shifts

                gt_flow = pcd - pcd_augm
                permutation = np.random.permutation(16384)
                pcd_src = pcd_augm[permutation[:8192]]
                gt_flow = gt_flow[permutation[:8192]]
                pcd_tgt = pcd[permutation[8192:]]
                
                # After augmentation, recompute topology or use pre-augmentation topo
                topo_feat_src = feat_for_augm
                topo_feat_tgt = feat_for_augm

            elif self.augm_setting.METHOD == 'rigid_one':
                setting = self.augm_setting
                max_transl = setting.MAX_TRANSLATION
                scale_offset = setting.MAX_SCALE_OFFSET
                rot_max = setting.MAX_ROTATION_ANGLE
                transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
                scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
                rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))

                theta = rot_angles[0]
                rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
                theta = rot_angles[1]
                rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                theta = rot_angles[2]
                rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

                if np.random.uniform() < 0.5:
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

                else:
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    pcd_tgt = np.dot(pcd_tgt, rot_mat) * scale + transl
                    lm_tgt = np.dot(lm_tgt, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

            else:
                pcd_src = pcd_src[:8192]
                pcd_tgt = pcd_tgt[:8192]
                gt_flow = gt_flow[:8192]

        else:
            pcd_src = pcd_src[:8192]
            pcd_tgt = pcd_tgt[:8192]
            gt_flow = gt_flow[:8192]

        # Construct input features
        if self.use_topo:
            # For topology-coupled model: 
            # - color arrays contain coords + broadcast topo (for backward compatibility in level0)
            # - topo_feat is returned separately for explicit coupling
            N_src = pcd_src.shape[0]
            N_tgt = pcd_tgt.shape[0]
            feat_src_tile = _np.tile(topo_feat_src.reshape(1, -1), (N_src, 1))
            feat_tgt_tile = _np.tile(topo_feat_tgt.reshape(1, -1), (N_tgt, 1))
            color_src = _np.concatenate([pcd_src, feat_src_tile], axis=1)
            color_tgt = _np.concatenate([pcd_tgt, feat_tgt_tile], axis=1)
            
            return (
                np.float32(pcd_src), 
                np.float32(pcd_tgt), 
                np.float32(color_src), 
                np.float32(color_tgt), 
                np.float32(gt_flow), 
                np.float32(topo_feat_src),
                np.float32(topo_feat_tgt),
                idx
            )
        else:
            # Original behavior without topology
            color_src = pcd_src
            color_tgt = pcd_tgt
            dummy_topo = np.zeros(1, dtype=np.float32)
            return (
                np.float32(pcd_src), 
                np.float32(pcd_tgt), 
                np.float32(color_src), 
                np.float32(color_tgt), 
                np.float32(gt_flow),
                np.float32(dummy_topo),
                np.float32(dummy_topo),
                idx
            )

    def __len__(self):
        return len(self.case_list)


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = phase == 'train'
        self.split = split
        self.kitti_root = args.kitti_root if hasattr(args, 'kitti_root') and args.kitti_root else '../mmdetection3d/data/kitti/training/velodyne'

        all_files = sorted([
            os.path.join(self.kitti_root, file_name)
            for file_name in os.listdir(self.kitti_root)
            if file_name.endswith('.bin')
        ])

        split_idx = int(len(all_files) * 0.8)
        if split == 'train':
            self.file_list = all_files[:split_idx]
        elif split == 'val':
            self.file_list = all_files[split_idx:]
        else:
            raise NotImplementedError()

        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.augm_setting = cfg.AUGMENTATIONS
        self.use_topo = cfg.MODEL.TOPO_FEAT_DIM > 0
        self.topo_dim = cfg.MODEL.TOPO_FEAT_DIM

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        if pcd.shape[0] >= 16384:
            sample_idx = np.random.permutation(pcd.shape[0])[:16384]
            pcd = pcd[sample_idx]
        else:
            pad_size = 16384 - pcd.shape[0]
            pcd = np.pad(pcd, ((0, pad_size), (0, 0)), mode='wrap')

        mean = np.mean(pcd, axis=0)
        pcd = (pcd - mean) / self.norm_factor

        pcd_src = pcd.copy()
        pcd_tgt = pcd.copy()
        lm_src = pcd.copy()
        lm_tgt = pcd.copy()
        gt_flow = np.zeros_like(pcd_src, dtype=np.float32)

        topo_feat_src = None
        topo_feat_tgt = None
        if self.use_topo and compute_topo_features is not None:
            try:
                topo_feat_src = compute_topo_features(pcd_src)
                topo_feat_tgt = compute_topo_features(pcd_tgt)
                if topo_feat_src.shape[0] != self.topo_dim or topo_feat_tgt.shape[0] != self.topo_dim:
                    topo_feat_src = np.zeros(self.topo_dim, dtype=np.float32)
                    topo_feat_tgt = np.zeros(self.topo_dim, dtype=np.float32)
            except Exception:
                topo_feat_src = np.zeros(self.topo_dim, dtype=np.float32)
                topo_feat_tgt = np.zeros(self.topo_dim, dtype=np.float32)

        if topo_feat_src is None:
            topo_feat_src = np.zeros(self.topo_dim if self.use_topo else 1, dtype=np.float32)
        if topo_feat_tgt is None:
            topo_feat_tgt = np.zeros(self.topo_dim if self.use_topo else 1, dtype=np.float32)

        if self.augm_setting.METHOD == 'multiscale_local_global':
            setting = self.augm_setting
            num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
            max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
            kernel_std_local = setting.KERNEL_STD_LOCAL
            global_grid_spacing = setting.GLOBAL_GRID_SPACING
            max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
            kernel_std_global = setting.KERNEL_STD_GLOBAL

            local_control_idx = np.random.permutation(pcd.shape[0])[:num_control_points_local]
            local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
            local_control_pts = pcd[local_control_idx]
            sq_dist = np.sum(np.square(pcd[:, None] - local_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_local ** 2)
            local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            local_pcd_shifts = np.nan_to_num(local_pcd_shifts)
            pcd_augm = pcd + local_pcd_shifts

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
            o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(
                global_grid_spacing,
                min_bound=np.array([-10., -10., -10.]),
                max_bound=np.array([10., 10., 10.])
            )

            global_control_pts = np.float32(np.asarray(o3d_cloud.points))
            global_control_shifts = np.random.uniform(-1., 1., (global_control_pts.shape[0], 3)) * max_control_shift_global
            sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
            global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            global_pcd_shifts = np.nan_to_num(global_pcd_shifts)

            pcd_augm = pcd_augm + global_pcd_shifts
            gt_flow = pcd - pcd_augm

            permutation = np.random.permutation(16384)
            pcd_src = pcd_augm[permutation[:8192]]
            gt_flow = gt_flow[permutation[:8192]]
            pcd_tgt = pcd[permutation[8192:]]

        elif self.augm_setting.METHOD == 'rigid_one':
            setting = self.augm_setting
            max_transl = setting.MAX_TRANSLATION
            scale_offset = setting.MAX_SCALE_OFFSET
            rot_max = setting.MAX_ROTATION_ANGLE
            transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
            scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
            rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))

            theta = rot_angles[0]
            rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            theta = rot_angles[1]
            rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            theta = rot_angles[2]
            rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

            pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
            lm_src = np.dot(lm_src, rot_mat) * scale + transl
            gt_flow = lm_tgt - lm_src

            permutation = np.random.permutation(16384)
            pcd_src = pcd_src[permutation[:8192]]
            gt_flow = gt_flow[permutation[:8192]]
            pcd_tgt = pcd_tgt[permutation[8192:]]

        else:
            pcd_src = pcd_src[:8192]
            pcd_tgt = pcd_tgt[:8192]
            gt_flow = gt_flow[:8192]

        if self.use_topo:
            feat_src_tile = _np.tile(topo_feat_src.reshape(1, -1), (pcd_src.shape[0], 1))
            feat_tgt_tile = _np.tile(topo_feat_tgt.reshape(1, -1), (pcd_tgt.shape[0], 1))
            color_src = _np.concatenate([pcd_src, feat_src_tile], axis=1)
            color_tgt = _np.concatenate([pcd_tgt, feat_tgt_tile], axis=1)
            return (
                np.float32(pcd_src),
                np.float32(pcd_tgt),
                np.float32(color_src),
                np.float32(color_tgt),
                np.float32(gt_flow),
                np.float32(topo_feat_src),
                np.float32(topo_feat_tgt),
                idx
            )

        color_src = pcd_src
        color_tgt = pcd_tgt
        dummy_topo = np.zeros(1, dtype=np.float32)
        return (
            np.float32(pcd_src),
            np.float32(pcd_tgt),
            np.float32(color_src),
            np.float32(color_tgt),
            np.float32(gt_flow),
            np.float32(dummy_topo),
            np.float32(dummy_topo),
            idx
        )

    def __len__(self):
        return len(self.file_list)
