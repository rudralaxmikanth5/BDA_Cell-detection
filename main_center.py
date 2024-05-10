import sys
import os
import torch
import cv2
import celldetection as cd
from matplotlib import pyplot as plt
import numpy as np
import time

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

        start_time = time.time()
        with torch.no_grad():
            x = cd.to_tensor(img, transpose=True, device=self.device, dtype=torch.float32)
            x = x / 255  
            x = x[None] 
            y = self.model(x)
        end_time = time.time() 

        execution_time = end_time - start_time
        print("Detection complete")
        print("Total execution time:", execution_time, "seconds")

        contours = y['contours']
        total_contours = 0  


        centers_output_path = output_img_path.replace('.png', '.txt')
        with open(centers_output_path, 'w') as file:
            for n in range(len(x)):
                for contour in contours[n]:
                    contour_np = contour.cpu().numpy()  # Convert tensor to numpy array
                    contour_np = contour_np.astype(np.int32)
                    M = cv2.moments(contour_np)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cyZ =2000
                        file.write(f"{cX}, {cY},{cyZ},{cyZ}\n")
                        total_contours += 1

        print("Total contours:", total_contours)
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
    output_path = detector.detect_cells_and_save(input_path, output_path, new_width=2000, new_height=2000)
    print("Output image and center coordinates saved at:", output_path)
