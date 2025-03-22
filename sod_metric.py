import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist

_EPS = np.spacing(1)

def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt

class IOU(object):
    def __init__(self):
        self.ious = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        intersection = np.sum(np.logical_and(pred > 0.5, gt))
        union = np.sum(np.logical_or(pred > 0.5, gt))
        iou = intersection / union if union > 0 else 0
        self.ious.append(iou)

    def get_results(self) -> dict:
        return dict(iou=np.mean(self.ious))

class PixelAccuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        pred_label = pred > 0.5
        self.correct += np.sum(pred_label == gt)
        self.total += gt.size

    def get_results(self) -> dict:
        return dict(pacc=self.correct / self.total)

class MAE(object):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        mae = np.mean(np.abs(pred - gt))
        self.maes.append(mae)

    def get_results(self) -> dict:
        return dict(mae=np.mean(self.maes))

class BER(object):
    def __init__(self):
        self.pos_err = 0
        self.neg_err = 0
        self.count = 0

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        TP, TN, FP, FN = self._calculate_metrics(pred, gt)
        self.pos_err += (1 - TP / (TP + FN + np.spacing(1))) * 100
        self.neg_err += (1 - TN / (TN + FP + np.spacing(1))) * 100
        self.count += 1

    def _calculate_metrics(self, pred: np.ndarray, gt: np.ndarray):
        # Ensure pred and gt are boolean arrays
        pred = pred > 0.5
        gt = gt > 0.5

        TP = np.logical_and(gt, pred).sum()
        TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
        FP = np.logical_and(np.logical_not(gt), pred).sum()
        FN = np.logical_and(gt, np.logical_not(pred)).sum()

        return TP, TN, FP, FN

    def get_results(self) -> dict:
        pos_err_avg = self.pos_err / self.count
        neg_err_avg = self.neg_err / self.count
        ber = (pos_err_avg + neg_err_avg) / 2
        return dict(ber=ber)


class WeightedFmeasure(object):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        if np.all(~gt):  # GT가 전부 배경인 경우
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        Dst, Idxt = bwdist(gt == 0, return_indices=True)
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)
        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        return dict(wfm=np.mean(self.weighted_fms))

class ShapeContext:
    def __init__(self, nbins_r=5, nbins_theta=12):
        self.shape_context_similarities = []
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta

    def _extract_contours(self, binary_image):
        contours = measure.find_contours(binary_image, 0.5)
        if contours:
            max_contour = max(contours, key=len)
            return max_contour
        return None

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        pred_contour = self._extract_contours(pred)
        gt_contour = self._extract_contours(gt)
        if pred_contour is None or gt_contour is None:
            self.shape_context_similarities.append(0.0)
            return
        sc_pred = self._compute_shape_context(pred_contour)
        sc_gt = self._compute_shape_context(gt_contour)
        similarity_score = self._shape_context_matching(sc_pred, sc_gt)
        self.shape_context_similarities.append(similarity_score)

    def _compute_shape_context(self, points):
        distances = cdist(points, points)
        angles = np.arctan2(points[:, 1][:, np.newaxis] - points[:, 1],
                            points[:, 0][:, np.newaxis] - points[:, 0])
        log_distances = np.log(distances + np.finfo(float).eps)
        shape_contexts = np.zeros((len(points), self.nbins_r * self.nbins_theta))
        r_bin_edges = np.linspace(np.min(log_distances), np.max(log_distances), self.nbins_r + 1)
        theta_bin_edges = np.linspace(-np.pi, np.pi, self.nbins_theta + 1)
        for i, (log_r, theta) in enumerate(zip(log_distances, angles)):
            r_bin_idx = np.digitize(log_r, r_bin_edges) - 1
            theta_bin_idx = np.digitize(theta, theta_bin_edges) - 1
            valid_idx = (r_bin_idx >= 0) & (r_bin_idx < self.nbins_r) & (theta_bin_idx >= 0) & (theta_bin_idx < self.nbins_theta)
            for r_idx, theta_idx in zip(r_bin_idx[valid_idx], theta_bin_idx[valid_idx]):
                shape_contexts[i, r_idx * self.nbins_theta + theta_idx] += 1
        return shape_contexts

    def _shape_context_matching(self, sc1, sc2):
        distances = np.zeros((len(sc1), len(sc2)))
        for i, sc1_hist in enumerate(sc1):
            for j, sc2_hist in enumerate(sc2):
                diff = (sc1_hist - sc2_hist) ** 2
                sum_sc = (sc1_hist + sc2_hist)
                distances[i, j] = 0.5 * np.sum(diff / (sum_sc + np.finfo(float).eps))
        return np.mean(np.min(distances, axis=1))

    def get_results(self):
        return dict(shape_context=np.mean(self.shape_context_similarities) if self.shape_context_similarities else 0.0)