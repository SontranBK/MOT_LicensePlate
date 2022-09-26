# MOT_LicensePlate

## 1. Reproducing this product:

<summary>Setup enviroments</summary>
Step1. Install the dependencies
```shell
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

```
Step2. Install torch and torchvision
```shell
gdown https://drive.google.com/uc?id=1-XmTOEN0z1_-VVCI3DPwmcdC-eLT_-n3
gdown https://drive.google.com/uc?id=1BdvXkwUGGTTamM17Io4kkjIT6zgvf4BJ
sudo -H pip3 install torch-1.8.0a0+37c1f4a-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install torchvision-0.9.0a0+01dfa8e-cp36-cp36m-linux_aarch64.whl
rm torchvision-0.9.0a0+01dfa8e-cp36-cp36m-linux_aarch64.whl
rm torch-1.8.0a0+37c1f4a-cp36-cp36m-linux_aarch64.whl
```
Step3. Installing
```shell
git clone https://github.com/SontranBK/MOT_LicensePlate.git
pip3 install requirements.txt
```
Step4. 
