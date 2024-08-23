<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>SAM Decoder fine-tunning</h1>

<h2>Introduction</h2>
<p>This code is based on a method by Stefan Todoran. You can read his article <a href="https://towardsdatascience.com/learn-transformer-fine-tuning-and-segment-anything-481c6c4ac802" target="_blank">here</a>.</p>

<h2>Features</h2>
<p> things that I edited </p>
<ul>
    <li>using huggingface</li>
    <li>using box prompt</li>
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


</body>
</html>
