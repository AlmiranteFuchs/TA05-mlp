#!/usr/bin/env python3
"""Builds report/relatorio.html from relatorio.md + figs, for Chrome print-to-pdf."""
import re, markdown

src = open("report/relatorio.md", encoding="utf-8").read()
figno = 0

def single(name, caption, w="88%"):
    global figno
    figno += 1
    return (f'<figure><img style="width:{w}" src="figs/{name}">'
            f'<figcaption>Figura {figno}: {caption}</figcaption></figure>')

def pair(a, b, caption):
    global figno
    figno += 1
    return (f'<figure><div class="pair"><img src="figs/{a}"><img src="{b}"></div>'
            f'<figcaption>Figura {figno}: {caption}</figcaption></figure>')

subs = {
    r"\[FIGURA: dataset\]": lambda: single("dataset.png",
        "as 3 classes do Iris: sepas (esq.) quase inseparaveis, petalas (dir.) bem melhores"),
    r"\[FIGURA: debug\]": lambda: single("debug_baseline.png",
        "esq.: o 'erro' que o codigo de referencia mostra vs. o MSE real da mesma rede; "
        "dir.: MSE real da referencia vs. versao corrigida (dados crus)"),
    r"\[FIGURA: norm\]": lambda: single("normalizacao.png",
        "normalizacao z-score: mesma rede, mesmos pesos iniciais, convergencia muito mais rapida"),
    r"\[FIGURA: lr\]": lambda: single("lr_sweep.png",
        "sweep de learning rate: curvas de MSE por epoca (acc de teste na legenda)"),
    r"\[FIGURA: hidden\]": lambda: single("hidden_sweep.png",
        "acuracia de teste vs. neuronios na camada escondida (media +- desvio, 3 seeds)"),
    r"\[FIGURA: overfit\]": lambda: single("overfitting.png",
        "dados limpos: nem H=64 overfitta, val acompanha treino (Iris e separavel demais)"),
    r"\[FIGURA: ruido\]": lambda: single("overfitting_ruido.png",
        "com 20% de ruido nos rotulos: H=64 memoriza o ruido (treino* sobe, val cai; "
        "val_loss diverge a partir de ~epoca 1000)"),
    r"\[FIGURA: best\]": lambda: single("best_model.png",
        "modelo final (4-8-3, sigmoid, lr=0.05): matriz de confusao e acuracia por classe"),
    r"\[FIGURA: regioes\]": lambda: single("decision_regions.png",
        "regioes de decisao de uma rede 2-8-3 treinada so nas features de petala "
        "(pontos grandes = teste, pequenos = treino)"),
}
for pat, fn in subs.items():
    src = re.sub(pat, lambda m, f=fn: f(), src)

body = markdown.markdown(src, extensions=["extra"])

html = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<style>
body {{ font-family: Georgia, serif; max-width: 720px; margin: 40px auto;
       line-height: 1.45; font-size: 14px; color: #111; padding: 0 12px; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 1.6em; }}
h3 {{ font-size: 14px; }}
code {{ font-family: Menlo, monospace; font-size: 12px; background: #f4f4f4; }}
pre code {{ display: block; padding: 10px; }}
a {{ color: #0645ad; }}
figure {{ margin: 18px 0; text-align: center; }}
.pair img {{ width: 49%; }}
figcaption {{ font-size: 12px; color: #555; margin-top: 6px; }}
li {{ margin: 4px 0; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #ccc; padding: 3px 8px; }}
</style></head><body>{body}</body></html>"""

open("report/relatorio.html", "w", encoding="utf-8").write(html)
print("report/relatorio.html written,", figno, "figures")
