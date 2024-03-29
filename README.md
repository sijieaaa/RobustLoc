# RobustLoc

(AAAI 2023) RobustLoc: Robust Camera Pose Regression in Challenging Driving Environments

[AAAI Proceddings](https://ojs.aaai.org/index.php/AAAI/article/view/25765/25537)

** **

:boom::boom:

- **Requirements**

  Platform

  ```
  CUDA>=11.0
  python>=3.6
  ```

  Pytorch installation (We have tested with Pytorch>=1.10 as well as newly released Pytorch2.0):

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

  Other dependencies:

  ```
  colour_demosaicing
  matplotlib
  numpy
  opencv_python
  Pillow
  scipy
  tqdm
  transforms3d
  torchdiffeq
  ```

- **Dataset**

  We currently provide the Oxford RobotCar dataset that has been pre-processed. 

  https://drive.google.com/file/d/1xewI1Cfq7a-zQfk2oGoJW6zJ8ZhMu_mK/view?usp=share_link
  
  [2023-06-13] The 4Seasons-related datasets&code have been uploaded at: https://drive.google.com/file/d/1H2ujRAd1v3reg31zDHoM1yBI0IUi1Ovz/view?usp=sharing

  [2023-05-09] The 4Seasons-related datasets&code are in preparation. Kindly please tuned.
  

- **Train and test**

  Check in tools/options.py and set your own --data_dir as where you store the Oxford RobotCar dataset.

  ```
  python train.py
  python eval.py
  ```

- **Code reference**

  https://github.com/BingCS/AtLoc

  https://github.com/psh01087/Vid-ODE

- **LiDAR-based Pose Regression**

  Feel free to check out our  CVPR'2023 work, which uses LiDAR point clouds for pose regression

  https://github.com/sijieaaa/HypLiLoc

  https://arxiv.org/abs/2304.00932

- **Citation**

  ```
  @inproceedings{wang2023robustloc,
    title={RobustLoc: Robust camera pose regression in challenging driving environments},
    author={Wang, Sijie and Kang, Qiyu and She, Rui and Tay, Wee Peng and Hartmannsgruber, Andreas and Navarro, Diego Navarro},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={37},
    number={5},
    pages={6209--6216},
    year={2023}
  }
  ```
