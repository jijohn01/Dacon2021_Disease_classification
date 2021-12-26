# Dacon2021_Disease_classification
## 01. 작물 병해 분류 AI 경진대회
 **Private 4등**
 
  병이 걸린 식물의 사진을 입력으로 하여 해당 질병을 분류하는 문제
  
  예를 들어,
  ![image](https://user-images.githubusercontent.com/28197373/147400543-8e2b5e06-30a9-4f8a-8f3e-cc88897435c7.png)
Link : https://dacon.io/competitions/official/235842/overview/description
## 02. 모델 설명
1) 먼저 모델은 Image 분류 문제에서 상위권에 분포된 모델 중 EfficientNet-b7을 기본으로 실험해 나가기로 했습니다. (결과적으로 EfficientNet 이외에 해보지 못함) 모델은 Ref[1]를 통해 ImageNet Dataset으로 Pretrain된 모델을 사용하였습니다.
Ref[1]: https://github.com/lukemelas/EfficientNet-PyTorch

2) 두번째로 부족한 학습데이터는 Data 증강과 Cross Validation을 적용하였습니다. Data 증강은 pytorch에서 제공하는 transform만으로는 부족하다 생각되어 CutMix를 사용하여 부족한 데이터를 보충했습니다. CutMix Code는 Ref[2]를 수정하여 사용하였습니다.
 또한, 고정된 Valid set을 사용하기에는 data 수가 적으므로 5개 fold로 Stratified K-fold를 통해 5개 모델을 학습시켰고 이것들을 Ensemble 하는 것으로 해결하고자 했습니다.
Ref[2]: https://github.com/ildoonet/cutmix

3) 마지막으로 학습데이터의 불균형한 문제가 있었습니다.  평가기준(Macro-f1)을 고려할 때 sample 수가 적은 Class에 잘 대응하는 것이 중요하다고 생각했습니다. 따라서 Loss Function에 Class별로 가중치를 주도록 했습니다(Weighted Cross Entropy). 가중치는 train dataset의 각 class별 sample 개수에 반비례하도록 설정하였습니다.


## 03. Dirs
Effi_CutMix.py

.. ├ cutmix

…. ├ cutmix.py

…. └ utils.py

.. ├ models

.. ├ train_imgs

.. ├ test_imgs

.. ├ train.csv

.. ├ test.csv

.. └ sample_submission.csv
