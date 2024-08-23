<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>SAM Decoder fine-tunning</h1>
<h2>Introduction</h2>
<p>What's up! This is wlsl6569 and this page is about using Hugging Face's SAM for fine-tuning.</p>
<p>This code is based on a method by Stefan Todoran. You can read his article <a href="https://towardsdatascience.com/learn-transformer-fine-tuning-and-segment-anything-481c6c4ac802" target="_blank">here</a>.</p>
<p>Essentially, this method involves retraining the mask decoder. We assume that the other components of SAM (the encoder and prompts encoder) are functioning properly. Our focus will be on fixing only the mask decoder.</p>
<p>Therefore, we will freeze all other parts except the mask decoder and retrain it.</p>


<h2>Features</h2>
<ul>
    <li>using huggingface</li>
    <li>using box prompt </li>
    <li>training only decoder part</li>
</ul>

<h2>How to use HuggingFace SAM</h2>
<ul>
  <li>HuggingFcae <a href="https://huggingface.co/docs/huggingface_hub/main/installation" target="_blank">installation</a> for using Hugging Face models and upload and use data within Hugging Face</li> 
  <li>HuggingFcae <a href="https://huggingface.co/docs/huggingface_hub/main/installation" target="_blank">SAM</a></li> 
</ul>
<p>for using HuggingFace SAM, I installed packages below too!</p>

```
pip install opencv-python 
pip install scikit-image
conda install matplotlib && \
pip install -U datasets && \
conda install -y pillow -c conda-forge && \
conda install -y huggingface_hub -c conda-forge && \
conda install -y libjpeg-turbo=2.1.0 && \
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
conda install -y transformers -c conda-forge && \
conda install -y accelerate && \
conda install -y monai && \
pip install fsspec==2023.9.2 
pip install accelerater
pip install ipywidgets && \
conda clean -p -y && \
conda clean -t -y
```

<h2>How to make a HuggingFace DataSet?</h2>
<p>First, collect your datas and organize it like this.</p>

```
my_dataset/
├── train/
│   ├── image
│   ├── label
│   ├── name
└── validation/
│   ├── image
│   ├── label
│   ├── name
```

<p>Upload it!</p>


```
import os
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
import numpy as np

# 경로 설정
train_image_dir = "./image_data/data/train/image"
train_label_dir = "./image_data/data/train/label"
validation_image_dir = "./image_data/data/validation/image"
validation_label_dir = "./image_data/data/validation/label"

def load_and_transform_dataset(image_dir, label_dir, image_ext="jpeg", label_ext="png"):
    # 이미지와 레이블 파일 목록 로드
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(label_ext)])
    
    # 이미지와 레이블 데이터셋 생성
    data = {
        "image": [],
        "label": [],
        "name": []
    }
    
    for image_file in image_files:
        # 이미지 파일 이름에서 확장자를 제거하여 레이블 파일 이름 생성
        label_file = os.path.splitext(image_file)[0] + "." + label_ext
        
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue
        
        try:
            # 이미지 로드 확인
            image = PILImage.open(image_path)
            image_array = np.array(image)
            print(f"Loaded image: {image_path}, shape: {image_array.shape}")
            
            # 레이블 데이터를 PIL을 사용하여 로드하고 (h, w)로 변환
            label_image = PILImage.open(label_path)
            label_array = np.array(label_image)
            if len(label_array.shape) == 3:
                label_array = label_array[:, :, 0]
            label_image = PILImage.fromarray(label_array)
            label_image.save(label_path)
            
            # 라벨 로드 확인
            label_array = np.array(label_image)
            print(f"Loaded label: {label_path}, shape: {label_array.shape}")
            
            data["image"].append(image_path)
            data["label"].append(label_path)
            data["name"].append(os.path.splitext(image_file)[0])
        
        except PILImage.UnidentifiedImageError:
            print(f"Cannot identify image file {label_path}")
        except Exception as e:
            print(f"Error processing file {label_path}: {e}")
        
    return data

# 데이터 로드 및 변환
train_data = load_and_transform_dataset(train_image_dir, train_label_dir, image_ext="jpeg", label_ext="png")
validation_data = load_and_transform_dataset(validation_image_dir, validation_label_dir, image_ext="jpeg", label_ext="png")

# 중간 데이터 확인
print("Train data sample:", train_data["image"][:2], train_data["label"][:2], train_data["name"][:2])
print("Validation data sample:", validation_data["image"][:2], validation_data["label"][:2], validation_data["name"][:2])

# Dataset 생성
train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(validation_data)

# Features 설정
features = Features({
    'image': Image(),
    'label': Image(),
    'name': Value('string')
})

train_dataset = train_dataset.cast(features)
validation_dataset = validation_dataset.cast(features)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

# 데이터셋 구조 확인
print(dataset_dict)

# 데이터셋 허브에 업로드
from huggingface_hub import notebook_login

notebook_login()

dataset_dict.push_to_hub("my_trans-v1")

```

<p>hope you this page helpful, have a good day!</p>
    
</body>
</html>
