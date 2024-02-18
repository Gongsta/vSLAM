# Visual SLAM

Made to work both on live video feed and stream of images.

Tested with Zed, but easily adaptable to other cameras. Open to testing other USB3.0 cameras if they are sent to me.

Getting started.

The code is make use of all of the NVIDIA hardware (VIC, CUDA, PVA, CPU). Some older hardware might not have access
to VIC or PVA, so you can just use the CUDA backend.

Prereqs for Pangolin (a visualization tool)
```
sudo apt-get install libglew-dev
```

Make sure you have Eigen installed
```
sudo apt install libeigen3-dev
```

### Compile and Build the Project
```
mkdir build
cd build
cmake ..
make
```

Visual Odometry Pipeline

### Step 1: Run Camera Calibration
1. Run Calibration to save calibration file

### Step 2: Run
