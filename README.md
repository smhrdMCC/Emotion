1. 정확도를 희생해서라도 빠른 반응성과 유지비 절감을 위해 BERT 모델 대신 DistilBERT 모델 사용
2. DistilBERT가 BERT 대비 정확도는 3% 떨어지나 모델 크기가 2/3
3. KoBERT의 Dependency 문제 해결을 위해 모듈들의 버전을 제한함. Python 3.7.x 이외에서 사용을 권장하지 않음
4. dataset_maker나 model_maker는 실제 서비스시에는 사용하지 않음(혹시 몰라 BASE64 인코딩 문자열을 경로에 추가)

5. saved_model.pt : https://drive.google.com/file/d/1a9U0X63b8BSlnrYnYtgaVLVtRvgayDdJ/view?usp=sharing
6. test acc 0.7038596524997434 test loss 0.6935189366340637

