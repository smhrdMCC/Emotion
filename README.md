KoBERT - DistilBERT based python model

-- Install --

    pip install -r requirements.txt

- CUDA 지원 설치

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

- KoBERT 설치

    pip install --no-deps git+https://git@github.com/SKTBrain/KoBERT.git@master

- Model 다운로드
./asset/saved_model.pt : 

(test model acc 0.7038596524997434, test model loss 0.6935189366340637)

1. 정확도를 희생해서라도 빠른 반응성과 유지비 절감을 위해 BERT 모델 대신 DistilBERT 모델 사용
2. DistilBERT가 BERT 대비 정확도는 3% 떨어지나 모델 크기가 2/3
3. KoBERT의 Dependency 문제 해결을 위해 모듈들의 버전과 설치 순서를 제한함. Python 3.7.x 이외에서 사용을 권장하지 않음
4. 일정 크기 이상의 파일은 Github에 업로드되지 않으므로 개별 설치

-- TroubleShooting --

1. BERT와 KoBERT 모델의 Dependency 문제
  - 최신 파이썬 버전, 최신 모듈에서는 동작하지 않고, pip dependency check도 불가능하기 때문에 KoBERT 설치시 --no-deps 명령어를 사용하여 설치 후, Dependency 문제가 발생할 때마다 동작하는 모듈 버전 조합을 https://pypi.org/을 참고하여 직접 찾고, 최종 동작 확인 후 requirements.txt에 작성.
  
2. BERT 모델의 응답성과 모듈의 용량 문제
  - 경량화 버전의 DistilBERT를 사용하여 정확도를 3~5%정도 희생하여 모델 크기는 1/3 이상 감소, 응답성은 AWS 프리티어에서도 충분할 정도로 개선됨

3. Github로 업로드 불가 현상
  - 30메가 이상의 파일을 Git을 활용하여 Github로 업로드시 용량 제한에 걸려 그 세션에서의 Git 활용이 불가능해지므로 .gitignore에 작성. .cash폴더는 KoBERT 모델이 huggingface 등에서 필요 모듈을 다운로드받는 데 사용하기 때문에 역시 .gitignore에 작성.

4. App과 json 방식으로 통신 불가
  - App과 Flask 서버가 Json 형식으로 소통시 json 형식이 뭉게지는 현상이 발생하여 String 형식으로 통신
  - 버전에 따른 호환성 문제인 것으로 추정