import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(W,"resultados"); F=os.path.join(W,"figuras")
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
BIO={"Amazonia":"#2a78d6","Cerrado":"#eb6834","Pantanal":"#1baf7a"}; NOME={"Amazonia":"Amazon","Cerrado":"Cerrado","Pantanal":"Pantanal"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":BASE,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":INK2,
    "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"savefig.dpi":300})
ca=pd.read_csv(os.path.join(D,"glmm_efeitos_com_ano.csv")); sa=pd.read_csv(os.path.join(D,"glmm_efeitos_sem_ano.csv"))
clim=[("p_jjas_anom","Dry-season rain\n(Jun–Sep), −1 SD",-1),("p_chuvosa_pre_anom","Preceding wet-\nseason rain, −1 SD",-1),("cdd_anom","Consecutive dry\ndays, +1 SD",1)]
lu=[("conv","Converted native veg.\n(prev. year), 0→10 p.p.",np.log(11)),("floresta","Forest, +10 p.p.",.1),("savana","Savanna, +10 p.p.",.1),("campestre","Grassland, +10 p.p.",.1),
    ("alagado","Flooded grassland, +10 p.p.",.1),("soja","Soybean, +10 p.p.",.1),("outras_temp","Other temporary crops, +10 p.p.",.1)]
fig,axs=plt.subplots(1,2,figsize=(7.4,4.4),gridspec_kw={"width_ratios":[1,1]})
def panel(ax,items,src_full,src_alt,title):
    yt=[]; yl=[]
    for i,(v,lab,s) in enumerate(items):
        for j,b in enumerate(BIO):
            yy=i*4+j
            r=src_full[(src_full.variavel==v)&(src_full.bioma==b)].iloc[0]
            lo,hi=sorted([np.exp(s*r.q025),np.exp(s*r.q975)]); m=np.exp(s*r.media)
            ax.plot([lo,hi],[yy,yy],color=BIO[b],lw=2,solid_capstyle="round"); ax.plot(m,yy,"o",color=BIO[b],ms=5,mec="white",mew=0.8)
            if src_alt is not None:
                r2=src_alt[(src_alt.variavel==v)&(src_alt.bioma==b)].iloc[0]; ax.plot(np.exp(s*r2.media),yy,"o",mfc="white",mec=BIO[b],ms=5,mew=1.2)
        yt.append(i*4+1); yl.append(lab)
    ax.axvline(1,color=BASE,lw=1); ax.set_xscale("log"); ax.set_yticks(yt); ax.set_yticklabels(yl); ax.invert_yaxis(); ax.grid(axis="y",visible=False)
    ax.set_xlabel("Rate ratio (log scale)"); ax.set_title(title,loc="left",color=INK,fontsize=9)
panel(axs[0],clim,sa,ca,"a  Climate")
panel(axs[1],lu,ca,None,"b  Land use and conversion")
for b in BIO: axs[0].plot([],[],"o",color=BIO[b],label=NOME[b])
axs[0].plot([],[],"o",mfc="white",mec=MUTED,label="within-year only"); fig.legend(frameon=False,fontsize=7,loc="lower center",ncol=4,labelcolor=INK2,bbox_to_anchor=(0.5,-0.01))
xt=[0.5,1,2,4,8,16]; axs[1].set_xticks(xt); axs[1].set_xticklabels([str(x) for x in xt]); axs[0].set_xticks([0.8,1,1.5,2,2.5]); axs[0].set_xticklabels(["0.8","1","1.5","2","2.5"])
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig(os.path.join(F,"fig3_efeitos_glmm.png"),facecolor="white",bbox_inches="tight",pad_inches=0.05); plt.close(fig)
