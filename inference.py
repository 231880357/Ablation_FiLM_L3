import argparse
import os
import numpy as np
import glob
from tqdm import tqdm

from defaults import get_cfg_defaults
from ppwc import Topo9_PointPWC
try:
    from topology import compute_topo_features
except ImportError:
    compute_topo_features = None


def main(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # computational stuff
    use_amp = cfg.USE_AMP
    device = 'cuda'

    # model: Topo9
    print("Using Topo9 (Original Topo + L4-only sparse residual topology coupling)")
    model = Topo9_PointPWC(cfg)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # data
    cases = [2, 8, 54, 55, 56, 94, 97, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
             120, 121, 122, 123]

    # Inference
    model.eval()

    if getattr(args, 'dataset', 'lung') == 'kitti':
        kitti_root = getattr(args, 'kitti_root', '../mmdetection3d/data/kitti/testing/velodyne')
        file_list = sorted(glob.glob(os.path.join(kitti_root, '*.bin')))

        if not os.path.exists(args.outfile):
            os.makedirs(args.outfile)

        infer_bar = tqdm(range(len(file_list) - 1), desc='KITTI inference', leave=False)
        for i in infer_bar:
            pcd_src_np = np.fromfile(file_list[i], dtype=np.float32).reshape(-1, 4)[:, :3]
            pcd_tgt_np = np.fromfile(file_list[i + 1], dtype=np.float32).reshape(-1, 4)[:, :3]

            if pcd_src_np.shape[0] >= 8192:
                pcd_src_np = pcd_src_np[np.random.permutation(pcd_src_np.shape[0])[:8192]]
            else:
                pcd_src_np = np.pad(pcd_src_np, ((0, 8192 - pcd_src_np.shape[0]), (0, 0)), mode='wrap')

            if pcd_tgt_np.shape[0] >= 8192:
                pcd_tgt_np = pcd_tgt_np[np.random.permutation(pcd_tgt_np.shape[0])[:8192]]
            else:
                pcd_tgt_np = np.pad(pcd_tgt_np, ((0, 8192 - pcd_tgt_np.shape[0]), (0, 0)), mode='wrap')

            pcd_src = torch.from_numpy(pcd_src_np).float().unsqueeze(0).to(device)
            pcd_tgt = torch.from_numpy(pcd_tgt_np).float().unsqueeze(0).to(device)
            pcd_src_orig = pcd_src.clone()

            norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
            mean = torch.mean(pcd_tgt, axis=1, keepdim=True)
            pcd_tgt = (pcd_tgt - mean) / norm_factor
            pcd_src = (pcd_src - mean) / norm_factor

            color_src = pcd_src
            color_tgt = pcd_tgt
            topo_src = torch.zeros(1, 6, device=device)
            topo_tgt = torch.zeros(1, 6, device=device)

            if cfg.MODEL.TOPO_FEAT_DIM > 0 and compute_topo_features is not None:
                feat_src = compute_topo_features(pcd_src[0].cpu().numpy())
                feat_tgt = compute_topo_features(pcd_tgt[0].cpu().numpy())
                t_feat_src = torch.from_numpy(feat_src).float().to(device)
                t_feat_tgt = torch.from_numpy(feat_tgt).float().to(device)
                topo_src = t_feat_src.unsqueeze(0)
                topo_tgt = t_feat_tgt.unsqueeze(0)
                topo_src_exp = t_feat_src.unsqueeze(0).unsqueeze(0).repeat(1, pcd_src.shape[1], 1)
                topo_tgt_exp = t_feat_tgt.unsqueeze(0).unsqueeze(0).repeat(1, pcd_tgt.shape[1], 1)
                color_src = torch.cat([pcd_src, topo_src_exp], dim=2)
                color_tgt = torch.cat([pcd_tgt, topo_tgt_exp], dim=2)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred_flows, _, _, _, _ = model(
                        pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                    )
                    pred_flow = pred_flows[0].permute(0, 2, 1)

            pred_flow = pred_flow * cfg.INPUT.SCALE_NORM_FACTOR
            tensor_to_save = torch.cat((pcd_src_orig + pred_flow, pcd_src_orig), dim=2)[0]
            out_path = os.path.join(args.outfile, f'kitti_{i:06d}_to_{i + 1:06d}.csv')
            np.savetxt(out_path, tensor_to_save.cpu().numpy(), delimiter=",", fmt='%.3f')
            infer_bar.set_postfix(file=os.path.basename(out_path))

        return

    infer_bar = tqdm(cases, desc='Lung inference', leave=False)
    for case in infer_bar:
        pcd_tgt = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 1)))[0].float()
        pcd_src = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 2)))[0].float()
        pcd_tgt = pcd_tgt.unsqueeze(0).to(device)
        pcd_src = pcd_src.unsqueeze(0).to(device)

        # prealignment
        pcd_src_orig = pcd_src.clone()
        mean_tgt = torch.mean(pcd_tgt, dim=1)
        std_tgt = torch.std(pcd_tgt, dim=1)
        mean_src = torch.mean(pcd_src, dim=1)
        std_src = torch.std(pcd_src, dim=1)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        pre_align_flow = pcd_src - pcd_src_orig

        # mean center and scale
        norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        mean = torch.mean(pcd_tgt, axis=1)
        pcd_tgt = (pcd_tgt - mean) / norm_factor
        pcd_src = (pcd_src - mean) / norm_factor

        # Prepare topology features
        color_src = pcd_src
        color_tgt = pcd_tgt
        topo_src = torch.zeros(1, 6, device=device)
        topo_tgt = torch.zeros(1, 6, device=device)
        
        if cfg.MODEL.TOPO_FEAT_DIM > 0 and compute_topo_features is not None:
            cpu_src = pcd_src[0].cpu().numpy()
            cpu_tgt = pcd_tgt[0].cpu().numpy()
            
            feat_src = compute_topo_features(cpu_src)
            feat_tgt = compute_topo_features(cpu_tgt)
            
            t_feat_src = torch.from_numpy(feat_src).float().to(device)
            t_feat_tgt = torch.from_numpy(feat_tgt).float().to(device)
            
            # Expand to [1, N, D] for color (level0 input)
            N = pcd_src.shape[1]
            t_feat_src_exp = t_feat_src.unsqueeze(0).unsqueeze(0).repeat(1, N, 1)
            t_feat_tgt_exp = t_feat_tgt.unsqueeze(0).unsqueeze(0).repeat(1, N, 1)
            
            color_src = torch.cat([color_src, t_feat_src_exp], dim=2)
            color_tgt = torch.cat([color_tgt, t_feat_tgt_exp], dim=2)
            
            # Global topo features for L4 adapter
            topo_src = t_feat_src.unsqueeze(0)  # [1, topo_dim]
            topo_tgt = t_feat_tgt.unsqueeze(0)  # [1, topo_dim]

        # inference
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                pred_flows, _, _, _, _ = model(
                    pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                )
                pred_flow = pred_flows[0].permute(0, 2, 1)

        pred_flow = pred_flow * cfg.INPUT.SCALE_NORM_FACTOR + pre_align_flow

        tensor_to_save = torch.cat((pcd_src_orig + pred_flow, pcd_src_orig), dim=2)[0]
        out_path = os.path.join(args.outfile, 'case_{:03d}.csv'.format(case))
        np.savetxt(out_path, tensor_to_save.cpu().numpy(), delimiter=",", fmt='%.3f')
        infer_bar.set_postfix(case=f'{case:03d}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference of Topology-Coupled PointPWC on Lung250M-4B')

    parser.add_argument('-M', '--model', default='ppwc_sup.pth', help="model file (pth)")
    parser.add_argument('-C', '--cloudfolder', default='cloudsTs',
                        help="folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-O', '--outfile', default='prediction_sup',
                        help="output file for keypoint displacement predictions")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--gpu", default="0", help="gpu to train on")
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti'],
                        help='dataset to use')
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/testing/velodyne',
                        help='folder containing KITTI .bin files')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    print(args)
    main(args)
