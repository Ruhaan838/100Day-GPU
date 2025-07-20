import torch
import triton
import triton.language as tl

from matplotlib import pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import numpy as np

@triton.jit
def wavelet_kernel(
    input_ptr, 
    output_LL_ptr,
    output_LH_ptr,
    output_HL_ptr,
    output_HH_ptr,
    H:tl.constexpr,
    W:tl.constexpr,
    stride:tl.constexpr,
    Block_Size:tl.constexpr
):
    pid = tl.program_id(0)
    num_cols = W // Block_Size
    row = pid // num_cols
    col = pid % num_cols
    
    if (row * 2 < H) and (col * 2 < W):
        x00 = tl.load(input_ptr + (row * 2) * stride + col * 2)
        x01 = tl.load(input_ptr + (row * 2) * stride + (col * 2 + 1))
        x10 = tl.load(input_ptr + ((row * 2 + 1) * stride) + col * 2)
        x11 = tl.load(input_ptr + ((row * 2 + 1) * stride) + (col * 2 + 1))
        
        LL = (x00 + x01 + x10 + x11) / 4
        LH = (x00 - x01 + x10 - x11) / 4
        HL = (x00 + x01 - x10 - x11) / 4
        HH = (x00 - x01 - x10 + x11) / 4
        
        out_idx = row * num_cols + col
        tl.store(output_LL_ptr + out_idx, LL)
        tl.store(output_LH_ptr + out_idx, LH)
        tl.store(output_HL_ptr + out_idx, HL)
        tl.store(output_HH_ptr + out_idx, HH)
        
def wavelet_transform(image_tensor):
    
    H, W = image_tensor.shape
    assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even."
    
    out_H, out_W = H // 2, W // 2
    
    LL = torch.empty((out_H, out_W), dtype=image_tensor.dtype, device=image_tensor.device)
    LH = torch.empty((out_H, out_W), dtype=image_tensor.dtype, device=image_tensor.device)
    HL = torch.empty((out_H, out_W), dtype=image_tensor.dtype, device=image_tensor.device)
    HH = torch.empty((out_H, out_W), dtype=image_tensor.dtype, device=image_tensor.device)
    
    num_block = out_H * out_W
    grid = (num_block,)
    wavelet_kernel[grid](
        image_tensor, 
        LL, 
        LH, 
        HL, 
        HH, 
        H, W, 
        image_tensor.stride(0), 
        2
    )
    
    return LL, LH, HL, HH

def download_and_process_image(url, size=(256, 256)):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")

    image_url = response.json()["message"]  
    img_response = requests.get(image_url)

    try:
        image = Image.open(BytesIO(img_response.content)).convert('L') 
    
    except Exception as e:
        raise Exception(f"Error processing image from {url}: {e}")
    
    image = image.resize(size)
    img_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)

image_url = "https://dog.ceo/api/breeds/image/random"

dog_image = download_and_process_image(image_url)
dog_image = dog_image.to('cuda')

LL, LH, HL, HH = wavelet_transform(dog_image)
dog_image_cpu = dog_image.cpu().numpy()
LL_cpu = LL.cpu().numpy()
LH_cpu = LH.cpu().numpy()
HL_cpu = HL.cpu().numpy()
HH_cpu = HH.cpu().numpy()

fig, axis = plt.subplots(1, 5, figsize=(20, 4))
axis[0].imshow(dog_image_cpu, cmap='gray')
axis[0].set_title('Original Image')
axis[0].axis('off')

axis[1].imshow(LL_cpu, cmap='gray')
axis[1].set_title('LL (Approx)')
axis[1].axis('off')

axis[2].imshow(LH_cpu, cmap='gray')
axis[2].set_title('LH (Horizontal Detail)')
axis[2].axis('off')

axis[3].imshow(HL_cpu, cmap='gray')
axis[3].set_title('HL (Vertical Detail)')
axis[3].axis('off')

axis[4].imshow(HH_cpu, cmap='gray')
axis[4].set_title('HH (Diagonal Detail)')
axis[4].axis('off')

plt.tight_layout()
plt.show()