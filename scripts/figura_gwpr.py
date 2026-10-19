import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(W,"resultados"); F=os.path.join(W,"figuras")
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; BASE="#c3c2b7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":BASE,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,"savefig.dpi":300})
div=LinearSegmentedColormap.from_list("div",["#1c5cab","#5598e7","#b7d3f6","#f0efec","#f6b8a8","#e34948","#a32524"])
c=pd.read_csv(os.path.join(D,"gwpr_coef_locais.csv"))
itens=[("conv","a  Recent native-vegetation conversion"),("floresta","b  Forest fraction"),("p_chuvosa_pre","c  Wet-season rainfall (climatology)"),("cdd","d  Consecutive dry days (climatology)")]
fig,axs=plt.subplots(2,2,figsize=(7.2,6.6))
for ax,(v,t) in zip(axs.ravel(),itens):
    b=c[f"b_{v}"]; sig=np.ones(len(b),bool); lim=np.quantile(np.abs(b),0.98)
    ax.scatter(c.x[~sig],c.y[~sig],c="#e1e0d9",s=9,marker="s",linewidths=0)
    sc=ax.scatter(c.x[sig],c.y[sig],c=b[sig],cmap=div,norm=TwoSlopeNorm(0,-lim,lim),s=9,marker="s",linewidths=0)
    cb=fig.colorbar(sc,ax=ax,fraction=0.04,pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED,labelsize=7); cb.set_label("local coefficient (per SD)",color=INK2,fontsize=7)
    ax.set_title(t,loc="left",color=INK,fontsize=9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.text(0.5,0.01,"0.25° cells, adaptive bisquare kernel (151 neighbours). Significance not mapped: Poisson standard errors understate uncertainty under overdispersion.",ha="center",color=MUTED,fontsize=7)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(os.path.join(F,"fig5_gwpr_coeficientes.png"),facecolor="white",bbox_inches="tight",pad_inches=0.05)
