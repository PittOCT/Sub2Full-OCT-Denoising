import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms
import numpy as np
import cv2
from unet import UNet


if __name__ == '__main__':
    img_dir = r'./test.png'
    model_dir = r'./best.pth'
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint)
    model.eval()
    input = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    input = trans(input).to(device)
    input = input.view(1, 1, 384, 384)

    with torch.no_grad():
        output = model(input).clamp(0.0, 1.0)


    output = output[0, :, :, :].cpu().numpy()
    output = np.reshape(output, (384, 384))
    output = np.array(output * 255, dtype=np.uint8)
    cv2.imwrite('./output.png', output)


