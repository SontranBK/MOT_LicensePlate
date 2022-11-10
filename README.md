# ***MOT_LicensePlate*** - an Intelligent Transport System with Multiple object tracking and License Plate detection

## 1. Getting Started

This product features an Intelligent Transport System with Multiple object tracking and License Plate detection. Main functions:

- Multiple object tracking with Re-id support
- Object counting and area-contour counting
- License Plate (Vietnam Standard) detection, with text recognition
- IoT system oriented with Jetson Nano deployment as edged devices

## 2. Updates and releases!!!
For all release features and publised product UI, check out [release page](https://github.com/SontranBK/MOT_LicensePlate/releases)

* 【Sep 28, 2022】

## 3. Reproducing this product:

<details>
<summary> 3.1. Jetson Nano Jetpack flash</summary>


</details>

<details>
<summary> 3.2. Setup enviroments for Jetson Nano</summary>

- Step1. Install the dependencies

```shell
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
```

- Step2. Install torch and torchvision

```shell
gdown https://drive.google.com/uc?id=1-XmTOEN0z1_-VVCI3DPwmcdC-eLT_-n3
gdown https://drive.google.com/uc?id=1BdvXkwUGGTTamM17Io4kkjIT6zgvf4BJ
sudo -H pip3 install torch-1.8.0a0+37c1f4a-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install torchvision-0.9.0a0+01dfa8e-cp36-cp36m-linux_aarch64.whl
rm torchvision-0.9.0a0+01dfa8e-cp36-cp36m-linux_aarch64.whl
rm torch-1.8.0a0+37c1f4a-cp36-cp36m-linux_aarch64.whl
```

- Step3. Installing

```shell
git clone https://github.com/SontranBK/MOT_LicensePlate.git
pip3 install requirements.txt
```

- Step4. 

</details>


## 4. Developer guide

<details>
<summary> 4.1. Programming language:</summary>

- This product is written mainly in Python language


</details>


<details>
<summary> 4.2. Deps and source code architecture:</summary>

 - List of dependencies:

    - [Minh huy fill in here]
    - [Minh huy fill in here]
    - [Minh huy fill in here]
</details>