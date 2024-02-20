Install

pip install -r requirements.txt

CUDA 지원

pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

KoBERT 설치

pip install --no-deps git+https://git@github.com/SKTBrain/KoBERT.git@master

/asset/에 saved_model.pt 넣기
saved_model.pt : https://drive.google.com/file/d/1a9U0X63b8BSlnrYnYtgaVLVtRvgayDdJ/view?usp=sharing

test acc 0.7038596524997434 test loss 0.6935189366340637

1. 정확도를 희생해서라도 빠른 반응성과 유지비 절감을 위해 BERT 모델 대신 DistilBERT 모델 사용
2. DistilBERT가 BERT 대비 정확도는 3% 떨어지나 모델 크기가 2/3
3. KoBERT의 Dependency 문제 해결을 위해 모듈들의 버전을 제한함. Python 3.7.x 이외에서 사용을 권장하지 않음