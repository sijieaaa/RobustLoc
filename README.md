# RobustLoc

(AAAI 2023) RobustLoc: Robust Camera Pose Regression in Challenging Driving Environments

** ** 
✴️✴️

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

- **Train and test**

  Check in tools/options.py and set your own --data_dir as where you store the Oxford RobotCar dataset.

  ```
  python train.py
  python eval.py
  ```

- **Code reference**

  https://github.com/BingCS/AtLoc

  https://github.com/psh01087/Vid-ODE

- **Citation**

  ```
  @article{wang2022robustloc,
    title={RobustLoc: Robust Camera Pose Regression in Challenging Driving Environments},
    author={Wang, Sijie and Kang, Qiyu and She, Rui and Tay, Wee Peng and Hartmannsgruber, Andreas and Navarro, Diego Navarro},
    journal={arXiv preprint arXiv:2211.11238},
    booktitle={The Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI} 2023},
    pages={online},
    publisher={{AAAI} Press},
    year={2023},
  }
  ```
