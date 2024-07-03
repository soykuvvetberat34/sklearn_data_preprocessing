import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
SI=SimpleImputer(missing_values=np.nan,strategy="mean")#hangi verileri değiştirecek,hangi strateji ile

data=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\deneme.csv")
df=pd.DataFrame(data)

veriler=df.iloc[:,1:4].values
SI=SI.fit(veriler)#imputer ı veriler üzerinde eğitir
df_last=SI.transform(veriler)#nan verileri doldurur

datas=df.iloc[:,0]
OneHotEncoder=preprocessing.OneHotEncoder()#yapı oluşturduk
datas_reshaped=datas.values.reshape(-1,1)#verileri 2D diziye dönüştürdük
#reshape(-1,1) diziyi tek sütun içeren 2D diziye dönüştürür,
#reshape(1,-1) diziyi tek satır içeren 2D diziye dönüştürür,
datas_ohe=OneHotEncoder.fit_transform(datas_reshaped).toarray()


country=pd.DataFrame(datas_ohe,columns=["tr","us","fr"])
datas_1=pd.DataFrame(df_last,columns=["boy","kilo","yaş"])
gender=df["cinsiyet"]
gender_df=pd.DataFrame(gender,columns=["cinsiyet"])
s=pd.concat([country,datas_1],axis=1)
s2=pd.concat([s,gender_df],axis=1)


#verileri train ve test verisi olarak ayırmak
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,gender_df,test_size=0.33,random_state=0)
"""
s: Bu genellikle bağımsız değişkenleri (özellikleri) içeren bir veri setidir. 
Makine öğrenmesi projelerinde X olarak da adlandırılır. Bu veri seti, 
modelin eğitilmesi ve performansının değerlendirilmesi için kullanılır.

gender_df: Bu ise genellikle bağımlı değişkeni (etiketi) içeren bir veri setidir. 
Makine öğrenmesi projelerinde y olarak da adlandırılır. Modelin öğrenmesi gereken 
hedef değişkeni veya sonuçları içerir.

 X veri seti, modelin özellikleri anlamasına ve öğrenmesine yardımcı olurken, 
 y veri seti modelin hangi sonuçları tahmin etmesi gerektiğini belirler. 
 Bu nedenle, model eğitimi ve testi için her iki veri setine de ihtiyaç vardır.
"""


#öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
X_train=SC.fit_transform(x_train)
X_test=SC.fit_transform(x_test)
print(X_train)
print(x_train)
print("\n")
print(X_test)
print(x_test)
