import os, sys, pickle, time, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
from cv_lib import *
df=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados", "samya_df.pkl"),"rb")).dropna(subset=["uso","cdd","elev","roaddist"])
df=df[df.uso.astype(str)!="NA"].reset_index(drop=True)
df["ano_real"]=df.year-1   # rótulo corrigido
keep={"3 - Formacao Florestal":"floresta","4 - Formacao Savanica":"savana","15 - Pastagem":"pastagem","39 - Soja":"soja","41 - Outras Lavouras Temporarias":"outras_temp","11 - Campo Alagado e Area Pantanosa":"alagado","12 - Formacao Campestre":"campestre"}
df["uso2"]=df.uso.astype(str).map(keep).fillna("outros")
X=pd.get_dummies(df[["uso2","bioma"]].astype(str),drop_first=False).astype(float).drop(columns=["uso2_pastagem","bioma_Amazonia"])
for c in ["cdd","elev","slope","roaddist"]: X[c]=(df[c]-df[c].mean())/df[c].std()
X["lpop"]=np.log1p(df.denspop*1000)
y=df.focos.values
from glum import GeneralizedLinearRegressor
def glm_nb(Xtr,ytr,Xte):
    m=GeneralizedLinearRegressor(family="negative.binomial",alpha=1e-4,l1_ratio=0,fit_intercept=True)
    m.fit(Xtr.values,ytr); return m.predict(Xte.values)
def gbm(Xtr,ytr,Xte):
    m=lgb.LGBMRegressor(objective="poisson",n_estimators=400,learning_rate=0.05,num_leaves=63,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
    m.fit(Xtr,ytr); return m.predict(Xte)
res=[]
for name,fp in [("NB-GLM",glm_nb),("LightGBM",gbm)]:
    for lab,fo in [("aleatória",folds_random(len(y))),("blocos 1°",folds_spatial(df.x.values,df.y.values,1.0)),("temporal (teste 2017–19)",folds_temporal(df.ano_real.values,2017))]:
        t0=time.time(); r=run_cv(fp,X,y,fo,lab); r["modelo"]=name; res.append(r); print(name,lab,round(time.time()-t0),"s",flush=True)
R=pd.concat(res); s=R.groupby(["modelo","esquema"])[["D2","AUC_ocorr","Spearman","MAE"]].mean().round(3)
print(s.to_string()); s.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados", "teste_cv_samya.csv"))
