import argparse
import os
import numpy as np
import cv2 

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

from tqdm import tqdm
import time

def load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name):
    """
    Load precomputed FD map and filtered image based on the image name.
    Args:
        fd_dir (str): Directory where FD maps are stored.
        filtered_image_dir (str): Directory where filtered images are stored.
        image_name (str): Name of the input image file.
    Returns:
        fd_map (torch.Tensor): Loaded FD map as a tensor.
        filtered_image (torch.Tensor): Loaded filtered image as a tensor.
    """
    # Remove extension to get the base name
    base_name = os.path.splitext(image_name)[0]

    # FD map path (assuming saved as .npy)
    fd_map_path = os.path.join(fd_dir, f"{base_name}.npy")

    # Load FD map
    if not os.path.exists(fd_map_path):
        raise FileNotFoundError(f"FD map not found: {fd_map_path}")
    fd_map = np.load(fd_map_path)
    fd_map = torch.tensor(fd_map, dtype=torch.float32)

    # Filtered image path (assuming same extension as input)
    filtered_image_path = os.path.join(filtered_image_dir, image_name)

    # Load filtered image
    if not os.path.exists(filtered_image_path):
        raise FileNotFoundError(f"Filtered image not found: {filtered_image_path}")
    filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR)
    filtered_image = torch.tensor(filtered_image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

    return fd_map, filtered_image




def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if hasattr(v, 'shape'):  # v가 텐서인 경우만 처리
                log('  {}: shape={}'.format(k, tuple(v.shape)))
            else:  # v가 문자열인 경우
                log('  {}: {}'.format(k, v))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader



def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader



import os
from torchvision.utils import save_image

def load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name):
    """
    Load precomputed FD map and filtered image based on the image name.

    Args:
        fd_dir (str): Directory containing FD maps.
        filtered_image_dir (str): Directory containing filtered images.
        image_name (str): Name of the input image file.

    Returns:
        fd_map (torch.Tensor): Loaded FD map as a tensor.
        filtered_image (torch.Tensor): Loaded filtered image as a tensor.
    """
    base_name = os.path.splitext(image_name)[0]
    fd_map_path = os.path.join(fd_dir, f"{base_name}.npy")
    filtered_image_path = os.path.join(filtered_image_dir, image_name)

    # Load FD map
    if not os.path.exists(fd_map_path):
        raise FileNotFoundError(f"FD map not found: {fd_map_path}")
    fd_map = torch.tensor(np.load(fd_map_path), dtype=torch.float32)

    # Load filtered image
    if not os.path.exists(filtered_image_path):
        raise FileNotFoundError(f"Filtered image not found: {filtered_image_path}")
    filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR)
    filtered_image = torch.tensor(filtered_image).permute(2, 0, 1).float() / 255.0

    return fd_map, filtered_image


def eval_psnr(loader, model, epoch, fd_dir, filtered_image_dir, eval_type=None, save_dir='./predictions'):
    """
    Evaluate the model's performance on a dataset.

    Args:
        loader (DataLoader): The data loader for evaluation.
        model (nn.Module): The model to evaluate.
        epoch (int): Current epoch number.
        fd_dir (str): Directory containing FD maps.
        filtered_image_dir (str): Directory containing filtered images.
        eval_type (str): Type of evaluation metric to use.
        save_dir (str): Directory to save prediction results.

    Returns:
        tuple: Evaluation metrics and corresponding names.
    """
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define evaluation metrics
    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric_names = ('f1', 'auc', 'none', 'none')
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric_names = ('f_mea', 'mae', 'none', 'none')
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric_names = ('shadow', 'non_shadow', 'ber', 'none')
    elif eval_type == 'cod':
        def metric_fn(pred_list, gt_list):
            metrics = utils.calculate_metrics(gt_list, pred_list)
            return (metrics['iou'], metrics['fmeasure'], metrics['accuracy'],
                    metrics['mae'], metrics['ber'], metrics['shapecontext'])
        metric_names = ('iou', 'fmeasure', 'accuracy', 'mae', 'ber', 'shapecontext')
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    pbar = tqdm(total=len(loader), leave=False, desc='val') if dist.get_rank() == 0 else None

    pred_list, gt_list = [], []
    for i, batch in enumerate(loader):
        # Handle batch items carefully
        for k, v in batch.items():
            if k == 'image_name':
                # Keep image names unchanged
                continue
            elif isinstance(v, list):
                batch[k] = torch.tensor(v).cuda()
            elif isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
            else:
                raise ValueError(f"Unsupported type for batch[{k}]: {type(v)}")

        inp, gt, image_name = batch['inp'], batch['gt'], batch['image_name'][0]

        # Load FD map and filtered image
        fd_map, filtered_image = load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name)

        # Perform inference
        torch.cuda.empty_cache()  # 💡 추가: 메모리 부족 방지
        pred = torch.sigmoid(model.infer(inp, image_name, fd_dir, filtered_image_dir))

        # Binarize GT and predictions
        binary_gt = (gt > 0.5).float()
        binary_pred = (pred > 0.4).float()

        # Save predictions and GT for first 20 samples
        if i < 20:
            epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            save_image(binary_pred, os.path.join(epoch_dir, f'pred_{i}.png'))
            save_image(binary_gt, os.path.join(epoch_dir, f'gt_{i}.png'))

        # Gather predictions and GT across GPUs
        batch_pred = [torch.zeros_like(binary_pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(binary_gt) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, binary_pred)
        dist.all_gather(batch_gt, binary_gt)

        pred_list.extend(batch_pred)
        gt_list.extend(batch_gt)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Concatenate predictions and GT lists
    pred_list = torch.cat(pred_list, dim=0)
    gt_list = torch.cat(gt_list, dim=0)

    # Compute evaluation metrics
    results = metric_fn(pred_list, gt_list)

    return (*results, *metric_names)




def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, fd_dir, filtered_image_dir):
    model.train()  # 모델을 트레이닝 모드로 설정
    loss_list = []  # 손실을 저장할 리스트

    # tqdm으로 진행 상황 표시
    pbar = tqdm(total=len(train_loader), desc="Training Progress", unit="batch", leave=True)
    
    start_time = time.time()  # 에포크 시작 시간 기록

    for batch_idx, batch in enumerate(train_loader):
        inp = batch['inp'].to(device)  # 입력 데이터를 디바이스로 이동
        gt = batch['gt'].to(device)  # Ground Truth 데이터를 디바이스로 이동
        image_name = batch['image_name'][0]  # 배치에서 이미지 이름 가져오기

        # 모델 입력 설정
        model.set_input(inp, gt, image_name, fd_dir, filtered_image_dir)

        # 파라미터 최적화 수행
        model.optimize_parameters()

        # 손실을 수집
        batch_loss = model.loss_G.item()
        loss_list.append(batch_loss)

        # 진행 상황 업데이트
        pbar.set_postfix(loss=f"{batch_loss:.4f}")
        pbar.update(1)

    pbar.close()  # tqdm 종료
    
    # 평균 손실 계산
    average_loss = sum(loss_list) / len(loss_list)

    # 에포크 완료 시간 계산
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch completed in {epoch_duration:.2f} seconds. Average Loss: {average_loss:.4f}")

    return average_loss

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 데이터 로더 생성
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # 모델, 옵티마이저, 스케줄러 준비
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    # FD 맵 및 필터링된 이미지 경로 로드
    fd_dir = config['fd_dir']
    filtered_image_dir = config['filtered_image_dir']

    # SAM 체크포인트 로드
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()

        # FD 및 필터링된 이미지 경로 전달
        train_loss_G = train(train_loader, model, fd_dir, filtered_image_dir)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, result5, result6, metric1, metric2, metric3, metric4, metric5, metric6 = eval_psnr(
                val_loader, model, epoch=epoch, fd_dir=fd_dir, filtered_image_dir=filtered_image_dir, eval_type=config.get('eval_type'))

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric5, result5))
                writer.add_scalars(metric5, {'val': result5}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric6, result6))
                writer.add_scalars(metric6, {'val': result6}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result4 < max_val_v:
                        max_val_v = result4
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()



def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
