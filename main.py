import sys
import os
import torch
import cv2
import celldetection as cd
from matplotlib import pyplot as plt
import numpy as np

class CellDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(self.device)
        self.model.eval()

    def detect_cells_and_save(self, input_img_path, output_img_path, new_width=None, new_height=None):
        img = cv2.imread(input_img_path)
        if new_width and new_height:
            img = cv2.resize(img, (new_width, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            x = cd.to_tensor(img, transpose=True, device=self.device, dtype=torch.float32)
            x = x / 255
            x = x[None]
            y = self.model(x)

        contours = y['contours']
        for n in range(len(x)):
            plt.figure(figsize=(16, 9))
            plt.imshow(np.ones((x[n].shape[1], x[n].shape[2], 3), dtype=np.uint8) * 255, cmap='gray')
            cd.plot_contours(contours[n], fill=True)
            for collection in plt.gca().collections:
                collection.set_linewidth(0)
            plt.axis('off')
            plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        return output_img_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py input_image_path output_image_path")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print("Input file not found.")
        sys.exit(1)

    detector = CellDetection()

    output_path = detector.detect_cells_and_save(input_path, output_path, new_width=2500, new_height=2000)
    print("Output image with contours saved at:", output_path)
