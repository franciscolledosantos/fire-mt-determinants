"""Monta os .docx da submissão à JEM a partir dos .md (manuscrito, suplementar, highlights, carta) e um PDF de conferência.
Figuras: fogo_final/jem/figures/Fig1–Fig5.png (numeração da JEM, arquivos separados) e FigS1–FigS2.png (no suplementar)."""
import os, re, sys, subprocess
J = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(J), "figuras")
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def formatar(out, numerar=True):
    doc = Document(out)
    for st in doc.styles:
        try:
            if st.type == 1:
                st.font.name = "Times New Roman"; st.font.size = Pt(12) if st.name not in ("Title",) else Pt(14)
                rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
                if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
                for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), "Times New Roman")
                st.paragraph_format.line_spacing = 2.0 if numerar else 1.15
                if st.name.startswith("Heading") or st.name == "Title":
                    st.font.color.rgb = RGBColor(0, 0, 0); st.font.bold = True
        except Exception:
            pass
    for p in doc.paragraphs:
        p.paragraph_format.line_spacing = 2.0 if numerar else 1.15
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    for r in p.runs: r.font.size = Pt(10)
    for s in doc.sections:
        sp = s._sectPr
        if numerar:
            ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); sp.append(ln)
        fp = s.footer.paragraphs[0]
        if "PAGE" not in fp._p.xml:
            fp.alignment = 1
            r = fp.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r._r.append(f1)
            r2 = fp.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"; r2._r.append(it)
            r3 = fp.add_run(); f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end"); r3._r.append(f2)
    doc.save(out)


def pandoc(src_md, out, resource=None):
    cmd = ["pandoc", src_md, "-f", "markdown+tex_math_dollars+pipe_tables", "-o", out]
    if resource: cmd += ["--resource-path", resource]
    subprocess.run(cmd, check=True)


# 1) manuscrito
md = open(os.path.join(J, "manuscript_JEM.md"), encoding="utf-8").read()
md = re.sub(r"```latex\n(.*?)\n```", lambda m: "$$" + m.group(1).strip() + "$$", md, flags=re.S)
tmp = os.path.join(J, "_full.md"); open(tmp, "w", encoding="utf-8").write(md)
out = os.path.join(J, "Manuscript_JEM_fire_MatoGrosso.docx"); pandoc(tmp, out); formatar(out)
for fx in ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5"]:
    assert os.path.exists(os.path.join(FIG, fx + ".png")), fx
# contagens para a revista
corpo = md.split("## Introduction")[1].split("## CRediT")[0]
resumo = md.split("## Abstract")[1].split("**Keywords")[0]
refs = md.split("## References")[1].split("## Tables")[0]
tabs = md.split("## Tables")[1].split("## Figure captions")[0]
legs = md.split("## Figure captions")[1]
n = lambda s: len(s.split())
print(f"resumo {n(resumo)} palavras; corpo Intro→Conclusões {n(corpo)}; referências {n(refs)}; tabelas {n(tabs)}; legendas {n(legs)}; "
      f"total md {n(md)}")
# 2) suplementar
sup = os.path.join(J, "Supplementary_JEM_fire_MatoGrosso.docx"); pandoc(os.path.join(J, "supplementary_JEM.md"), sup, FIG); formatar(sup, numerar=False)
# 3) highlights e carta
hl = os.path.join(J, "Highlights_JEM.docx"); pandoc(os.path.join(J, "highlights.txt"), hl); formatar(hl, numerar=False)
for l in open(os.path.join(J, "highlights.txt"), encoding="utf-8"):
    if l.startswith("•"): print("highlight", len(l.strip("• \n")), "caracteres")
cl = os.path.join(J, "Cover_letter_JEM.docx"); pandoc(os.path.join(J, "cover_letter_JEM.md"), cl); formatar(cl, numerar=False)
# 4) PDF de conferência do manuscrito
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", J, out], check=True, capture_output=True)
print("ok", out, os.path.getsize(out))
