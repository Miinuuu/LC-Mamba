# LC-Mamba


# conda 환경
conda  create -n lxfi
conda activate lxfi

# pytorch 환경 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# library 환경 
pip install -r requirements.txt


# 체크포인트
https://www.dropbox.com/scl/fo/qqetwthteq18n53p48nwi/ALtsYM3Dse5nW5QfA5KwGEA?rlkey=lhcev43r7ltvv80cr3iszj6nj&dl=0
  
# 데이터셋
본 연구에서는 제안된 모델의 성능을 다양한 조건과 해상도에서 종합적으로 검증하기 위해 여러 데이터셋을 활용하여 실험을 수행하였습니다. 

<ul>
  <li>/경로/datasets
    <ul>
      <li>HD_dataset</li>
      <li>middlebury</li>
      <li>snufilm</li>
      <li>ucf101</li>
      <li>vimeo_setuplet</li>
      <li>vimeo_triplet</li>
      <li>X4K1000FPS</li>
      <li>Xiph</li>
    </ul>
  </li>
</ul>



## 데이터셋 준비:

   * [Vimeo90K dataset](http://toflow.csail.mit.edu/)
   * [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow)
   * [Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py)
   * [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/)
   * [SNU-FILM dataset](https://myungsub.github.io/CAIN/)


1) Vimeo90K 는 448×256 해상도의 프레임 트리플릿으로 구성된 데이터셋으로, 테스트 세트에는 총 3,782개의 트리플릿이 포함되어 있습니다. 
2) UCF101 은 DVF 에서 선정한 256×256 해상도의 프레임 트리플릿 379개로 구성된 테스트 세트를 사용하였습니다.
3) Middlebury 는 해상도가 약 640×480인 OTHER 세트를 테스트용으로 활용하였습니다. 
4) SNU-FILM 은 약 1280×720 해상도의 프레임 트리플릿 1,240개로 이루어진 데이터셋으로, 
   움직임의 크기에 따라 Easy, Medium, Hard, Extreme의 네 가지 난이도로 구분되어 있어 상세한 성능 비교가 가능합니다.
5) Xiph 는 원본 이미지를 2K 해상도로 다운샘플링한 “Xiph-2K”와 중앙을 크롭한 “Xiph-4K” 버전으로 구성하여 테스트에 사용하였습니다. 

# 실행
    Makefile
## 훈련 
    Make train
## 벤치
    Make bench
## 데모
    Make demo