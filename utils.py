import os
import time
import shutil

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter

import sod_metric
from sod_metric import IOU, PixelAccuracy, MAE, BER, WeightedFmeasure, ShapeContext  # Import the SOD metrics

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret


def calculate_metrics(ground_truths, predictions):
    assert len(ground_truths) == len(predictions), "Number of ground truth images and predicted images must be the same."
   
    iou_metric = IOU()
    pixel_acc_metric = PixelAccuracy()
    mae_metric = MAE()
    ber_metric = BER()
    wfm_metric = WeightedFmeasure()
    shape_context_metric = ShapeContext()  # ShapeContext 메트릭 추가

    for gt, pred in zip(ground_truths, predictions):
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
       
        iou_metric.step(pred, gt)
        pixel_acc_metric.step(pred, gt)
        mae_metric.step(pred, gt)
        ber_metric.step(pred, gt)
        wfm_metric.step(pred, gt)
        shape_context_metric.step(pred, gt)  # ShapeContext 계산 추가

    return {
        'iou': iou_metric.get_results()['iou'],
        'fmeasure': wfm_metric.get_results()['wfm'],
        'accuracy': pixel_acc_metric.get_results()['pacc'],
        'mae': mae_metric.get_results()['mae'],
        'ber': ber_metric.get_results()['ber'],
        'shapecontext': shape_context_metric.get_results()['shape_context'],  # ShapeContext 결과 반환
    }


def calc_cod(y_pred, y_true):
    batchsize = y_true.shape[0]

    metric_FM = sod_metric.WeightedFmeasure()
    metric_MAE = sod_metric.MAE()

    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        for i in range(batchsize):
            true, pred = \
                y_true[i, 0].cpu().data.numpy() * 255, y_pred[i, 0].cpu().data.numpy() * 255

            metric_FM.step(pred=pred, gt=true)
            metric_MAE.step(pred=pred, gt=true)

        fm = metric_FM.get_results()["wfm"]  # Weighted F-measure 사용
        mae = metric_MAE.get_results()["mae"]

    return fm, mae  # 필요에 따라 추가 반환값 수정
