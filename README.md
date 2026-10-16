# TA05 - Experimentos com o MLP from scratch

CI1026 - Visão Computacional e Percepção (UFPR, Prof. Eduardo Todt)

Estudo, depuração e experimentos sobre o notebook de referência
"MLP from scratch" (Iris, 3 classes): port fiel do código original com os
bugs preservados, versão corrigida, e sweeps de convergência, capacidade,
ativação e overfitting.

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib scikit-learn pandas markdown jupyter nbconvert
```

## Usage

```sh
python src/run_experiments.py    # roda tudo -> report/figs/*.png + out/results.json
python report/build.py           # report/relatorio.md -> report/relatorio.html
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
```

## Layout

- `ref_mlp_from_scratch.ipynb` — notebook de referência da disciplina (baixado)
- `src/mlp_ref.py` — port fiel da referência, **bugs preservados** (comparação)
- `src/mlp.py` — MLP corrigido (numpy puro), com histórico treino/val por época
- `src/run_experiments.py` — todos os experimentos do relatório
- `notebook.ipynb` — notebook entregável (narrativa + experimentos executados)
- `report/` — relatório (md -> html -> pdf) e figuras
- `out/` — `results.json` com os números dos sweeps

## Resumo dos achados

- o `predict` da referência não aplica ativação → a rede aprende (MSE 0.02)
  mas acerta só 33% no teste; com ativação no predict: 100%
- bias de saída atualizado dentro do loop errado; biases int64 truncam os
  updates (nunca aprendem); erro reportado ~4x menor que o MSE real;
  `params=None` quebra (typo `iasHiddenValue`)
- receita p/ 100%: normalizar + sigmoid/tanh + lr~0.05 + H>=2 + early stop
- relu com bias=-1 = rede morta (33%); com bias=0 revive instável (0.86)
- Iris limpo não overfitta nem com H=64; com 20% de ruído nos rótulos o
  H=64 memoriza (treino* 0.92 ↑, val 1.0 -> 0.97 ↓, val_loss diverge)
