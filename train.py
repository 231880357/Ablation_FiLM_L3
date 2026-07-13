import argparse
import time
import os
import numpy as np
from tqdm import tqdm

from defaults import get_cfg_defaults


def train(cfg, args):
    root = cfg.BASE_DIRECTORY
    exp_name = cfg.EXPERIMENT_NAME
    out_folder = os.path.join(root, exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model_path = os.path.join(out_folder, 'model.pth')
    model_path_ep = os.path.join(out_folder, 'model_ep={}.pth')

    # hyperparameters
    init_lr = cfg.SOLVER.LEARNING_RATE
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    lr_steps = cfg.SOLVER.LR_MILESTONES
    lr_gamma = cfg.SOLVER.LR_LAMBDA
    batch_size = cfg.SOLVER.BATCH_SIZE

    # computational stuff
    use_amp = cfg.USE_AMP
    num_workers = 0 if args.debug else cfg.NUM_WORKERS
    device = torch.device('cuda:0' if cfg.DEVICE.startswith('cuda') else cfg.DEVICE)

    # model: Topo9 (Original Topo + L4-only topology coupling)
    print("Using Topo9 (Original Topo + L4-only sparse residual topology coupling)")
    model = Topo9_PointPWC(cfg)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), init_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)
    else:
        raise ValueError()

    # datasets
    if getattr(args, 'dataset', 'lung') == 'kitti':
        train_set = KittiDataset(cfg, args, phase='train', split='train')
    else:
        train_set = Lung250MDataset(cfg, args, phase='train', split='train')
    if args.debug:
        if hasattr(train_set, 'case_list'):
            train_set.case_list = train_set.case_list[:8]
        else:
            train_set.file_list = train_set.file_list[:8]
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    
    if getattr(args, 'dataset', 'lung') == 'kitti':
        val_set = KittiDataset(cfg, args, phase='test', split='val')
    else:
        val_set = Lung250MDataset(cfg, args, phase='test', split='val')
    if args.debug:
        if hasattr(val_set, 'case_list'):
            val_set.case_list = val_set.case_list[:8]
        else:
            val_set.file_list = val_set.file_list[:8]
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # logging
    validation_log = np.zeros([num_epochs, 3])

    for ep in range(1, num_epochs + 1):
        print('Started epoch {}/{}'.format(ep, num_epochs))
        model.train()
        loss_values = []
        start_time = time.time()

        lambda_topo = 0.01  # Auxiliary topology consistency loss weight

        train_pbar = tqdm(train_loader, desc=f'Train {ep}/{num_epochs}', leave=False)
        for it, data in enumerate(train_pbar, 1):
            pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, topo_src, topo_tgt, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            color_src = color_src.to(device)
            color_tgt = color_tgt.to(device)
            gt_flow = gt_flow.to(device)
            topo_src = topo_src.to(device)
            topo_tgt = topo_tgt.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_flows, fps_pc1_idxs, _, _, _ = model(
                    pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                )
                loss_flow = multiScaleLoss(pred_flows, gt_flow, fps_pc1_idxs)
                loss_topo = topo_pyramid_loss(pcd_src, pcd_tgt, pred_flows[0], k=20)
                loss = loss_flow + lambda_topo * loss_topo

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at iteration {it}, skipping batch")
                continue
                
            loss_values.append(loss.item())
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', batches=it)
            if it % 100 == 0:
                tqdm.write(f'[Train] epoch={ep} iter={it} loss={loss.item():.6f}')
            loss = loss * cfg.SOLVER.LOSS_FACTOR
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

        train_loss = np.mean(loss_values) if loss_values else np.nan
        validation_log[ep - 1, 0] = train_loss
        lr_scheduler.step()

        # Validation
        model.eval()
        epe_3d = 0
        epe_initial = 0
        val_pbar = tqdm(val_loader, desc=f'Val {ep}/{num_epochs}', leave=False)
        for it, data in enumerate(val_pbar, 1):
            pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, topo_src, topo_tgt, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            color_src = color_src.to(device)
            color_tgt = color_tgt.to(device)
            topo_src = topo_src.to(device)
            topo_tgt = topo_tgt.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred_flows, _, _, _, _ = model(
                        pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                    )
                    pred_flow = pred_flows[0].permute(0, 2, 1)

            gt_flow = gt_flow.to(device)
            err_per_sample = (pred_flow - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
            epe_3d += err_per_sample.sum().item()
            epe_initial += gt_flow.square().sum(dim=2).sqrt().mean(dim=1).sum().item()
            val_pbar.set_postfix(epe=f'{(epe_3d / (it * pcd_src.shape[0])) * val_loader.dataset.norm_factor:.4f}', batches=it)
            if it % 100 == 0:
                current_epe = (epe_3d / (it * pcd_src.shape[0])) * val_loader.dataset.norm_factor
                tqdm.write(f'[Val] epoch={ep} iter={it} epe={current_epe:.6f}')

        epe_3d = epe_3d / len(val_loader.dataset) * val_loader.dataset.norm_factor
        epe_initial = epe_initial / len(val_loader.dataset) * val_loader.dataset.norm_factor
        validation_log[ep - 1, 1:] = [epe_initial, epe_3d]

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
              'initial error', epe_initial, 'EPEs', epe_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--debug", default=False, help="whether to use debug mode", type=bool)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="physical GPU index to expose to this process",
    )
    parser.add_argument('-CTr', '--cloudfolder_train', default='../cloudsTr/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-CVal', '--cloudfolder_val', default='../cloudsTs/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-STr','--supfolder_train', default='../corrfieldFlowPcdTr', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('-SVal','--supfolder_val', default='../corrfieldFlowPcdTs', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti'],
                        help='dataset to use')
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/training/velodyne',
                        help='folder containing KITTI .bin files')

    args = parser.parse_args()

    if args.gpu < 0:
        parser.error('--gpu must be a non-negative physical GPU index')

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Select the physical GPU before importing Torch-dependent project modules.
    # CUDA remaps the selected physical device to logical cuda:0 in this process.
    import torch

    if cfg.DEVICE.startswith('cuda'):
        if not torch.cuda.is_available():
            parser.error(
                f'physical GPU {args.gpu} is unavailable; '
                'check --gpu, the NVIDIA driver, and CUDA_VISIBLE_DEVICES'
            )
        torch.cuda.set_device(0)
        print(
            f'GPU selection: physical GPU {args.gpu} -> logical cuda:0 '
            f'({torch.cuda.get_device_name(0)})'
        )

    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dataset import Lung250MDataset, KittiDataset
    from ppwc import Topo9_PointPWC, multiScaleLoss, topo_pyramid_loss

    train(cfg, args)
