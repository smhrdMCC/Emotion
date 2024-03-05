import pandas as pd

ckpt_path="./assets/"

# ----------------------------------------------------------------------------------------------
# 한국어 감정 정보 연속적 대화 데이터셋 
# https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271

# Read .xlsx
keti_constant=pd.read_excel(ckpt_path+'한국어_연속적_대화_데이터셋.xlsx', engine='openpyxl')

print(keti_constant.shape)