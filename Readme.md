## System Information

- CUDA Version: 12.1
- Python Version: 3.10.9
- Ubuntu Version:
  - Distributor ID: Ubuntu
  - Description: Ubuntu 22.04.4 LTS
  - Release: 22.04
  - Codename: jammy
- pip Version: 22.3.1



## Installing Dependencies

To install the required dependencies, use this command:

<code>  pip install -r requirements.txt  </code>

## Dependencies

- NumPy (Version: 1.23.3)
- PyTorch (Version: 2.2.1)
- Torchvision (Version: 0.17.1)
- celldetection (Version: 0.4.2)
- Numba (Version: 0.56.4)
- OpenCV-Python (Version: 4.9.80)


## How to Run
  ### To Run a singe file 
    We can run a single file by the following command:
<code>python main.py input_image_path output_image_path</code>

### To Run a multiple files 
    We can run multiple images by adding them to the inputs folder and executing the following commands respectively:
<code> chmod +x process_images.sh
</code>

<code> ./process_images.sh
</code>
