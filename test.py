import argparse
import os
import torch
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import models
import utils
import datasets

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name):
    """
    Load precomputed FD map and filtered image based on the image name.
    """
    base_name = os.path.splitext(image_name)[0]
    fd_map_path = os.path.join(fd_dir, f"{base_name}.npy")
    filtered_image_path = os.path.join(filtered_image_dir, image_name)

    # Load FD map
    if not os.path.exists(fd_map_path):
        raise FileNotFoundError(f"FD map not found: {fd_map_path}")
    fd_map = torch.tensor(np.load(fd_map_path), dtype=torch.float32).to(device)

    # Load filtered image
    if not os.path.exists(filtered_image_path):
        raise FileNotFoundError(f"Filtered image not found: {filtered_image_path}")
    filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR)
    filtered_image = torch.tensor(filtered_image).permute(2, 0, 1).float() / 255.0
    filtered_image = filtered_image.to(device)

    return fd_map, filtered_image

def make_data_loader(spec):
    """
    Create DataLoader for test dataset.
    """
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    return loader

def test_model(loader, model, save_dir, fd_dir, filtered_image_dir):
    """
    Run inference on the test set and save all predictions.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(total=len(loader), desc="Testing Progress", unit="batch", leave=True)

    for i, batch in enumerate(loader):
        # 데이터 GPU 이동
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        inp, gt, image_name = batch['inp'], batch['gt'], batch['image_name'][0]

        # FD 맵 및 필터링된 이미지 로드
        fd_map, filtered_image = load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name)

        # 모델 예측
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp, image_name, fd_dir, filtered_image_dir))

        # Binarization
        binary_pred = (pred > 0.4).float()

        # 저장 경로
        save_pred_path = os.path.join(save_dir, f"{image_name}_pred.png")
        save_gt_path = os.path.join(save_dir, f"{image_name}_gt.png")

        # 예측 결과 저장
        save_image(binary_pred, save_pred_path)
        save_image(gt, save_gt_path)

        pbar.update(1)

    pbar.close()
    print(f"✅ 모든 테스트 결과가 '{save_dir}'에 저장되었습니다.")

def main(args):
    # Config 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("✅ Config loaded.")

    # 데이터 로더 생성
    test_loader = make_data_loader(config.get('test_dataset'))

    # 모델 로드
    model = models.make(config['model']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Checkpoint '{args.checkpoint}' 로드 완료.")

    # FD 맵 및 필터링된 이미지 경로 로드
    fd_dir = config['fd_dir']
    filtered_image_dir = config['filtered_image_dir']

    # 모델 테스트 실행
    test_model(test_loader, model, args.save_dir, fd_dir, filtered_image_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--save_dir", default="./test_results", help="Directory to save predictions")
    args = parser.parse_args()

    main(args)
