## First of all we need to run the generate ArUco markers with different IDs as the following

```bash
python3 generate_aruco_markers.py --count 4 --size 200 --output corner_markers
``` 

## Then we need to run the following command to calibrate the workspace or transform the workspace from camera view into the world view

```bash
./workspace_tool ../data/intrinsic_params.yml test_extrinsic.yaml 0.05 ../data/workspace.yaml visualize.png 0 1 2 3
```


- Prerequisites:
    - The intrinsic parameters of the camera should be provided in the file `intrinsic_params.yml`
    - The extrinsic calibrated file `test_extrinsic.yaml`
    - The physical size of the ArUco marker in meters `0.05`, You have to measure it manually with ruller to see it actual printed out size in meters
    - The output file `workspace.yaml` which contains the transformation matrix from the camera view to the world view
    - The output visualization image `visualize.png` which shows the workspace in the world view 
```