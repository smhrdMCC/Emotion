KoBERT - DistilBERT based python model

-- Install --

    pip install -r requirements.txt

- CUDA 지원 설치

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

- KoBERT 설치

    pip install --no-deps git+https://git@github.com/SKTBrain/KoBERT.git@master

- Model 다운로드
./asset/saved_model.pt : 

    https://drive.google.com/file/d/1a9U0X63b8BSlnrYnYtgaVLVtRvgayDdJ/view?usp=sharing

(test model acc 0.7038596524997434, test model loss 0.6935189366340637)


1. 정확도를 희생해서라도 빠른 반응성과 유지비 절감을 위해 BERT 모델 대신 DistilBERT 모델 사용
2. DistilBERT가 BERT 대비 정확도는 3% 떨어지나 모델 크기가 2/3
3. KoBERT의 Dependency 문제 해결을 위해 모듈들의 버전과 설치 순서를 제한함. Python 3.7.x 이외에서 사용을 권장하지 않음
4. 일정 크기 이상의 파일은 Github에 업로드되지 않으므로 개별 설치