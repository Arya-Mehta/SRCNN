import torch
import torch.nn as nn
from models import SRCNN
from utilities import calculate_psnr
from PIL import ImageFilter, Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

device = 'cpu' if torch.mps.is_available() else 'cpu'

def preprocess(img_path):

    if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
        image_path = img_path
        high_res = Image.open(image_path).convert('RGB')
        high_res = high_res.resize(((high_res.width // 2) * 2, (high_res.height // 2) * 2), resample=Image.Resampling.BICUBIC)
        low_res = high_res.resize(((high_res.width // 2) * 2, (high_res.height // 2) * 2), resample=Image.Resampling.BICUBIC)
        low_res = low_res.filter(ImageFilter.GaussianBlur(radius=2))
        print(np.max(high_res))
        high_res = np.asarray(high_res, dtype='float32') / 255
        low_res = np.asarray(low_res, dtype='float32') / 255

        transform = transforms.Compose([transforms.ToTensor()])
        low_res = transform(low_res)
        high_res = transform(high_res)
        
        return low_res, high_res

def load_model():
    model = SRCNN(channels=3)
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    
    return model

def predict(low_res, high_res, model):
    low_res.to(device)
    criterion = nn.MSELoss()
    
     
    model.eval()
    with torch.no_grad():
        preds = model(low_res)
        loss = criterion(preds, high_res)

        result = calculate_psnr(np.float32(preds), np.float32(high_res))
    
    print(f"Loss: {loss:.3f}")
    print(f"PSNR Value: {result:.2f}")

    return low_res.cpu().detach().numpy(), preds.cpu().detach().numpy(), high_res.cpu().detach().numpy(), result

def display(low_res, preds, high_res, result):
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Low-Res Input", f"Model Output\nPSNR: {result:.2f}", "High-Res Target"]


    for ax, img, title in zip(axes, [np.transpose(low_res, (1, 2, 0)), np.transpose(preds, (1, 2, 0)), np.transpose(high_res, (1, 2, 0))], titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")  

    plt.tight_layout()
    
    save_path = 'image_comparison.png'
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"Plot saved as {save_path}")

def save_images(low_res, preds):
    low_res = np.clip(low_res, 0, 1)
    preds = np.clip(preds, 0, 1)
    low_output_file = "low_res_image.png"
    high_output_file = "high_res_image.png"

    plt.imsave(low_output_file, np.transpose(low_res, (1, 2, 0)))
    plt.imsave(high_output_file, np.transpose(preds, (1, 2, 0)))

    print(f"Images saved as {low_output_file} and {high_output_file}")

def main():
    
    test_image = 'datasets/Set5/woman.png'
    lr, hr = preprocess(test_image)
    model = load_model()
    lr, op, hr, psnr = predict(lr, hr, model)
    display(lr, op, hr, psnr)
    save_images(lr, op)


if __name__ == '__main__':
    main()
