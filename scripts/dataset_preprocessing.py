import pandas as pd
import numpy as np

ckpt_path="../assets/"
ckpt_name=ckpt_path+"sentiment_dialogues.csv"

# ---------------------------------------------------------------------------------------------
# 한국어 감정 정보가 포함된 단발성 대화 데이터셋
# https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270

# Read .xlsx
keti_single=pd.read_excel(ckpt_path+'한국어_단발성_대화_데이터셋.xlsx', engine='openpyxl')

# Delete other column
keti_single=keti_single.iloc[:,:2]

# Rename columns to fit other dataset
keti_single.rename(columns={'Sentence':"발화",'Emotion':'감정'},inplace=True)
# Delete data to fit other dataset
keti_single=keti_single[keti_single["감정"]!="놀람"]
# Merge data fo fit other dataset
keti_single["감정"]=np.where(keti_single["감정"].str.match("공포"),"불안",keti_single["감정"])


# ----------------------------------------------------------------------------------------------
# 한국어 감정 정보 연속적 대화 데이터셋 
# https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271

# Read .xlsx
keti_constant=pd.read_excel(ckpt_path+'한국어_연속적_대화_데이터셋.xlsx', engine='openpyxl')

# Correct data to fit other dataset
new_columns=keti_constant.iloc[0]
keti_constant=keti_constant.rename(columns=new_columns).iloc[:,1:3]
keti_constant.drop(index=0,axis=0,inplace=True)

# Drop the data to fit other dataset
keti_constant=keti_constant[keti_constant["감정"].isin(["중립","놀람","분노","슬픔","행복","혐오","공포"])]

# Rename the column to fit other dataset
keti_constant["감정"]=np.where(keti_constant["감정"].str.match("놀람"),"당황",keti_constant["감정"])
keti_constant["감정"]=np.where(keti_constant["감정"].str.match("공포"),"불안",keti_constant["감정"])


# ----------------------------------------------------------------------------------------------
# 감성대화 말뭉치
# https://aihub.or.kr/aidata/7978

# Read .xlsx
fin_t=pd.read_excel(ckpt_path+'감성대화말뭉치(최종데이터)_Training.xlsx', engine='openpyxl')
fin_v=pd.read_excel(ckpt_path+'감성대화말뭉치(최종데이터)_Validation.xlsx', engine='openpyxl')

# Fill the N/A
fin_t=fin_t.fillna("")
fin_v=fin_v.fillna("")

# Merge the scattered data 
fin_t["사람문장"]=fin_t["사람문장1"].astype(str)+fin_t["사람문장2"].astype(str)+fin_t["사람문장3"].astype(str)
fin_v["사람문장"]=fin_v["사람문장1"].astype(str)+fin_v["사람문장2"].astype(str)+fin_v["사람문장3"].astype(str)
df_concat=pd.concat([fin_t,fin_v])

# Drop other column to merge other dataset
final_df=df_concat[["사람문장","감정_대분류"]]
# Rename column
final_df=final_df.rename({"감정_대분류":"감정"},axis=1)

# Stripping whitespace on "감정" column
final_df["감정"]=final_df["감정"].apply(lambda x:x.strip())

# Rename data to merge other dataset
# np.where(condition, value when true, value when false)
final_df["감정"]=np.where(final_df["감정"].str.match("상처"),"슬픔",final_df["감정"])
final_df["감정"]=np.where(final_df["감정"].str.match("기쁨"),"행복",final_df["감정"])
final_df=final_df.reset_index(drop=True)

# Rename columns to merge other dataset
final_df.rename({"사람문장":"발화"},axis=1,inplace=True)

# -----------------------------------------------------------------------------------------------
# Merge the data
real_final_df=pd.concat([final_df,keti_constant,keti_single])
real_final_df.reset_index(drop=True,inplace=True)

# Save the data
real_final_df.to_csv(ckpt_name)

# Checking
keti_single[keti_single["감정"]=="중립"].sample(50)