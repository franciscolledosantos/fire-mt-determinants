import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(W,"resultados"); F=os.path.join(W,"figuras")
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":BASE,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":MUTED,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"savefig.dpi":300})
R=pd.read_csv(os.path.join(D,"cv_resultados_folds_final.csv"))
ESQ=[("aleatória (5 folds)","Random\n5-fold"),("blocos espaciais 1° (5 folds)","Spatial\n1° blocks"),("blocos espaciais 2° (5 folds)","Spatial\n2° blocks"),("temporal (teste a partir de T_TESTE)","Temporal\n2020–24")]
MOD=[("NB-GLM","NB GLM","#2a78d6"),("LightGBM (padrão)","LightGBM, default","#eb6834"),("LightGBM (ajustado, CV interna aleatória)","LightGBM, tuned (random inner CV)","#1baf7a"),("LightGBM (ajustado, CV interna espacial)","LightGBM, tuned (spatial inner CV)","#eda100")]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={"width_ratios":[1.35,1]})
ax=axs[0]; off=np.linspace(-0.15,0.15,len(MOD))
for k,(m,lab,col) in enumerate(MOD):
    xs=[];ys=[];es=[]
    for i,(e,_) in enumerate(ESQ):
        d=R[(R.modelo==m)&(R.esquema==e)].D2
        xs.append(i+off[k]); ys.append(d.mean() if len(d) else np.nan); es.append(d.std() if len(d)>1 else 0)
    ax.errorbar(xs,ys,yerr=es,color=col,lw=2,marker="o",ms=5,capsize=0,elinewidth=1,label=lab,mec="white",mew=0.8)
ax.set_xticks(range(len(ESQ))); ax.set_xticklabels([l for _,l in ESQ]); ax.set_ylabel("Deviance explained (D²), test folds"); ax.set_ylim(0,0.42)
ax.set_title("a  Skill by validation design",loc="left",color=INK,fontsize=9); ax.grid(axis="x",visible=False)
h,l=ax.get_legend_handles_labels()
# b: inflação = random - spatial1 para cada modelo (barras horizontais)
ax=axs[1]; rows=[]
for m,lab,col in MOD:
    a=R[(R.modelo==m)&(R.esquema==ESQ[0][0])].D2.mean(); s=R[(R.modelo==m)&(R.esquema==ESQ[1][0])].D2.mean()
    rows.append((lab,col,(a-s)/a*100))
for i,(lab,col,v) in enumerate(rows):
    ax.barh(i,v,color=col,height=0.55); ax.text(v+0.6,i,f"{v:.0f}%",va="center",color=INK2,fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y",visible=False)
ax.set_xlabel("Drop in D², random → spatial 1° (%)"); ax.set_title("b  Optimism of random CV",loc="left",color=INK,fontsize=9)
fig.legend(h,l,frameon=False,fontsize=7,loc="lower center",ncol=4,labelcolor=INK2,bbox_to_anchor=(0.5,-0.04)); fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(os.path.join(F,"fig4_validacao.png"),facecolor="white",bbox_inches="tight",pad_inches=0.05)
