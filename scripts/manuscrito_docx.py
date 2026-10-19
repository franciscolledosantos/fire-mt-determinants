"""Gera o .docx do manuscrito a partir da exportação em Markdown do doc (Claude Docs) e das figuras locais em português.
Uso: python manuscrito_docx.py <export.json> <saida.docx>"""
import sys, json, base64, re, os, subprocess
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); F = os.path.join(W, "figuras", "pt")
FIGS = {1: "fig1_area_densidade_pt.png", 2: "fig2_series_bioma_pt.png", 3: "fig3_efeitos_glmm_pt.png", 4: "fig5_gwnbr_coeficientes_pt.png",
        5: "fig6_shap_bioma_pt.png", 6: "fig4_validacao_pt.png", 7: "fig7_validacao_avancados_pt.png", 8: "fig8_focos_area_queimada_pt.png"}
d = json.load(open(sys.argv[1]))["data"]; md = base64.b64decode(d["bytes_b64"]).decode("utf-8")
md = re.sub(r"&#91;image: Figura (\d+)\\\]", lambda m: f"![]({os.path.join(F, FIGS[int(m.group(1))])}){{width=16cm}}", md)
md = re.sub(r"^.*@Francisco Lledo dos Santos\s*$", "", md, flags=re.M)
md = re.sub(r"```latex\n(.*?)\n```", lambda m: "$$" + m.group(1).strip() + "$$", md, flags=re.S)
tmp = os.path.join(W, "data", "manuscrito_pt.md"); open(tmp, "w").write(md)
subprocess.run(["pandoc", tmp, "-f", "markdown+tex_math_dollars+pipe_tables", "-o", sys.argv[2], "--resource-path", F], check=True)
print("ok", sys.argv[2], os.path.getsize(sys.argv[2]))
