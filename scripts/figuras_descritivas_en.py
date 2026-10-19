import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(W,"resultados"); F=os.path.join(W,"figuras")
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"; SURF="#ffffff"
BIO={"Amazonia":"#2a78d6","Cerrado":"#eb6834","Pantanal":"#1baf7a"}; NOME={"Amazonia":"Amazon","Cerrado":"Cerrado","Pantanal":"Pantanal"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":BASE,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"figure.dpi":300,"savefig.dpi":300})
pan=pd.read_parquet(os.path.join(D,"painel_MT_2010_2024.parquet"))
# Fig 1: mapa densidade média anual de focos (por 100 km²) + contorno dos biomas
area=(0.05*111.32)*(0.05*111.32*np.cos(np.deg2rad(pan.y)))  # km²
pan["dens"]=pan.focos/area*100
m=pan.groupby(["x","y","bioma"],as_index=False).dens.mean()
seq=LinearSegmentedColormap.from_list("seq",["#fdf0e6","#f6c3a4","#ef9566","#eb6834","#c24e1c","#8a3511"])
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6),gridspec_kw={"width_ratios":[1,1]})
ax=axs[0]; bcol={"Amazonia":0,"Cerrado":1,"Pantanal":2}
sc=ax.scatter(m.x,m.y,c=m.bioma.map(bcol),cmap=ListedColormap(list(BIO.values())),s=0.6,marker="s",linewidths=0,rasterized=True)
for b,c in BIO.items(): ax.scatter([],[],c=c,marker="s",s=20,label=NOME[b])
ax.legend(frameon=False,loc="lower left",fontsize=7,labelcolor=INK2); ax.set_title("a  Biomes",loc="left",color=INK,fontsize=9)
ax=axs[1]; v=np.clip(m.dens,0,np.quantile(m.dens,0.98))
sc=ax.scatter(m.x,m.y,c=v,cmap=seq,s=0.6,marker="s",linewidths=0,rasterized=True)
cb=fig.colorbar(sc,ax=ax,fraction=0.04,pad=0.02); cb.set_label("active fires (100 km²)⁻¹ yr⁻¹",color=INK2,fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED,labelsize=7)
ax.set_title("b  Mean annual active-fire density, 2010–2024",loc="left",color=INK,fontsize=9)
for ax in axs: ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
fig.tight_layout(); fig.savefig(os.path.join(F,"fig1_area_densidade.png"),facecolor=SURF); plt.close(fig)
# Fig 2: focos por bioma × ano (painel a) e anomalias de chuva antecedente e CDD (painéis b,c)
t=pan.groupby(["year","bioma"]).focos.sum().unstack()/1000
c=pan.groupby(["year","bioma"])[["p_chuvosa_pre_anom","cdd_anom"]].mean()
fig,axs=plt.subplots(3,1,figsize=(7.2,6.2),sharex=True,gridspec_kw={"height_ratios":[1.4,1,1]})
for b in BIO:
    axs[0].plot(t.index,t[b],color=BIO[b],lw=2,marker="o",ms=4,label=NOME[b])
    axs[1].plot(c.xs(b,level=1).index,c.xs(b,level=1).p_chuvosa_pre_anom,color=BIO[b],lw=2,marker="o",ms=4)
    axs[2].plot(c.xs(b,level=1).index,c.xs(b,level=1).cdd_anom,color=BIO[b],lw=2,marker="o",ms=4)
axs[0].set_ylabel("Active fires (thousand)"); axs[0].legend(frameon=False,ncol=3,loc="upper center",labelcolor=INK2)
axs[1].set_ylabel("Preceding wet-season\nrainfall anomaly (SD)"); axs[2].set_ylabel("Consecutive dry days\nanomaly (SD)")
for ax in axs[1:]: ax.axhline(0,color=BASE,lw=1)
for i,l in enumerate("abc"): axs[i].set_title(f"{l}",loc="left",color=INK,fontsize=9,fontweight="bold")
axs[0].annotate("Pantanal 2020",xy=(2020,t.loc[2020,"Pantanal"]),xytext=(2016.3,15),color=INK2,fontsize=7,arrowprops=dict(arrowstyle="-",color=MUTED,lw=0.6))
axs[2].set_xticks(range(2010,2025,2)); axs[2].set_xlabel("Year")
fig.tight_layout(); fig.savefig(os.path.join(F,"fig2_series_bioma.png"),facecolor=SURF); plt.close(fig)
t.round(2).to_csv(os.path.join(D,"tab_focos_bioma_ano.csv")); print(t.round(1).to_string())
