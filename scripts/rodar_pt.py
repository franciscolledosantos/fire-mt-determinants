"""Roda um script de figura e grava a versão em português (sufixo _pt): traduz os textos da figura e usa
vírgula decimal nos rótulos numéricos. Uso: python rodar_pt.py scripts/figura_x.py"""
import os, re, sys, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.text as mtext
import matplotlib.figure as mfig

PT = {
    "Amazon": "Amazônia", "Year": "Ano", "Latitude (°)": "Latitude (°)", "Longitude (°)": "Longitude (°)",
    "Consecutive dry days\nanomaly (SD)": "Anomalia de dias secos\nconsecutivos (DP)",
    "Hotspots (thousand)": "Focos (mil)",
    "Preceding wet-season\nrainfall anomaly (SD)": "Anomalia da chuva da estação\nchuvosa anterior (DP)",
    "a  Biomes": "a  Biomas",
    "b  Mean annual hotspot density, 2010–2024": "b  Densidade média anual de focos, 2010–2024",
    "hotspots · 100 km⁻² · yr⁻¹": "focos · 100 km⁻² · ano⁻¹",
    "Consecutive dry\ndays, +1 SD": "Dias secos\nconsecutivos, +1 DP",
    "Converted native veg.\n(prev. year), 0→10 p.p.": "Vegetação nativa convertida\n(ano anterior), 0→10 p.p.",
    "Dry-season rain\n(Jun–Sep), −1 SD": "Chuva da estação seca\n(jun–set), −1 DP",
    "Flooded grassland, +10 p.p.": "Campo alagado, +10 p.p.", "Forest, +10 p.p.": "Floresta, +10 p.p.",
    "Grassland, +10 p.p.": "Campo, +10 p.p.", "Other temporary crops, +10 p.p.": "Outras lavouras temporárias, +10 p.p.",
    "Preceding wet-\nseason rain, −1 SD": "Chuva da estação\nchuvosa anterior, −1 DP",
    "Rate ratio (log scale)": "Razão de taxas (escala log)", "Savanna, +10 p.p.": "Savana, +10 p.p.",
    "Soybean, +10 p.p.": "Soja, +10 p.p.", "a  Climate": "a  Clima", "b  Land use and conversion": "b  Uso do solo e conversão",
    "within-year only": "só variação dentro do ano",
    "Deviance explained (D²), test folds": "Desviância explicada (D²), folds de teste",
    "Drop in D², random → spatial 1° (%)": "Queda do D², aleatória → blocos de 1° (%)",
    "LightGBM, default": "LightGBM, padrão", "LightGBM, tuned (random inner CV)": "LightGBM, ajustado (CV interna aleatória)",
    "LightGBM, tuned (spatial inner CV)": "LightGBM, ajustado (CV interna espacial)", "NB GLM": "GLM binomial negativo",
    "Random\n5-fold": "Aleatória\n5 folds", "Spatial\n1° blocks": "Blocos\nde 1°", "Spatial\n2° blocks": "Blocos\nde 2°",
    "Temporal\n2020–24": "Temporal\n2020–24", "a  Skill by validation design": "a  Desempenho por desenho de validação",
    "b  Optimism of random CV": "b  Otimismo da validação aleatória",
    "a  Recent native-vegetation conversion": "a  Conversão recente de vegetação nativa", "b  Forest fraction": "b  Fração de floresta",
    "c  Wet-season rainfall (climatology)": "c  Chuva da estação chuvosa (climatologia)",
    "d  Consecutive dry days (climatology)": "d  Dias secos consecutivos (climatologia)",
    "local coefficient (per SD)": "coeficiente local (por DP)",
    "CDD anomaly (SD)": "Anomalia de CDD (DP)", "Dry-season rain anomaly (SD)": "Anomalia da chuva da\nestação seca (DP)",
    "Forest fraction": "Fração de floresta", "Preceding wet-season rain anomaly (SD)": "Anomalia da chuva da estação\nchuvosa anterior (DP)",
    "SHAP (log rate)": "SHAP (log da taxa)", "Soybean fraction": "Fração de soja", "log(1 + p.p. converted)": "log(1 + p.p. convertidos)",
    "Deviance explained in held-out blocks (D²)": "Desviância explicada nos blocos de teste (D²)",
    "ZINB + BYM spatial effect": "ZINB + efeito espacial BYM", "NB-GLM": "GLM binomial negativo", "NB global": "NB global",
    "RF global": "Floresta aleatória global", "GWRF (a = 0.5)": "GWRF (a = 0,5)",
    "a  0.05° cell-years, 1° spatial blocks": "a  Células-ano de 0,05°, blocos de 1°",
    "b  0.25° cells, 2° spatial blocks": "b  Células de 0,25°, blocos de 2°",
    "Bars: median across folds; points: individual folds (values below −0.2 clipped). Orange: geographically weighted models.":
        "Barras: mediana dos folds; pontos: folds individuais (valores abaixo de −0,2 truncados). Laranja: modelos geograficamente ponderados.",
}
REGEX = [  # textos montados com f-strings
    (re.compile(r"GWNBR, 0\.25° cells, adaptive bisquare kernel \((\d+) neighbours, AICc\)\. Grey: \|t\| < ([\d.]+) \(multiple-testing adjusted\)\."),
     lambda m: f"GWNBR, células de 0,25°, núcleo bisquare adaptativo ({m.group(1)} vizinhos, AICc). Cinza: |t| < {m.group(2).replace('.', ',')} (ajuste para testes múltiplos)."),
]
NUM = re.compile(r"^([−\-]?\d+)\.(\d+)(%?)$")
_orig = mtext.Text.set_text


def set_text(self, s):
    if isinstance(s, str):
        if s in PT:
            s = PT[s]
        else:
            for rx, fn in REGEX:
                if rx.fullmatch(s): s = rx.sub(fn, s)
            m = NUM.match(s)
            if m: s = f"{m.group(1)},{m.group(2)}{m.group(3)}"
    return _orig(self, s)


mtext.Text.set_text = set_text
_save = mfig.Figure.savefig


def savefig(self, fname, *a, **k):
    if isinstance(fname, str):
        r, e = os.path.splitext(fname); d, b = os.path.split(r); os.makedirs(os.path.join(d, "pt"), exist_ok=True)
        fname = os.path.join(d, "pt", b + "_pt" + e)
    return _save(self, fname, *a, **k)


mfig.Figure.savefig = savefig
script = os.path.abspath(sys.argv[1]); sys.path.insert(0, os.path.dirname(script)); sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
