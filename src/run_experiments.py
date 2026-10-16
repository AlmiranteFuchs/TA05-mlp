"""TA05 - experimentos com o MLP from scratch (Iris, 3 classes).

Gera report/figs/*.png e out/results.json:
  1. reproducao do baseline (implementacao de referencia, bugs inclusos)
  2. debug: erros encontrados no codigo original
  3. efeito da normalizacao
  4. sweeps: learning rate, neuronios escondidos, funcao de ativacao
  5. overfitting: curvas treino vs validacao, early stopping
  6. melhor config: matriz de confusao, acuracia por classe, regioes de decisao
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MultiLayerPerceptron
from mlp_ref import MLPReferencia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'report', 'figs')
OUT = os.path.join(ROOT, 'out')
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .3})
CLASS_COLORS = ['#1f77b4', '#2ca02c', '#d62728']
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
results = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print('  fig:', name)


# ---------------- dados ----------------
iris = load_iris()
X, y = iris.data, iris.target
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=11, stratify=y)   # mesmo split do notebook

scaler = StandardScaler().fit(train_X)
train_Xn, test_Xn = scaler.transform(train_X), scaler.transform(test_X)

# split interno treino/validacao p/ analise de overfitting
tr_X, val_X, tr_y, val_y = train_test_split(
    train_Xn, train_y, test_size=0.25, random_state=7, stratify=train_y)

BASE = dict(InputLayer=4, HiddenLayer=5, OutputLayer=3,
            Epocas=3000, LearningRate=0.005, BiasHiddenValue=-1,
            BiasOutputValue=-1, ActivationFunction='sigmoid')

# ---------------- fig: dataset ----------------
print('[1] dataset')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, (feats, title) in enumerate([((0, 1), 'sepas'), ((2, 3), 'petalas')]):
    for c in range(3):
        m = y == c
        ax[k].scatter(X[m, feats[0]], X[m, feats[1]], s=22, alpha=.75,
                      c=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax[k].set_xlabel(iris.feature_names[feats[0]])
    ax[k].set_ylabel(iris.feature_names[feats[1]])
    ax[k].set_title(f'features da {title}')
ax[0].legend(fontsize=8)
save(fig, 'dataset.png')

# ---------------- 1. baseline + debug ----------------
print('[2] baseline (referencia, bugs inclusos) + MLP corrigido')

# demo do bug: params=None estoura AttributeError ('iasHiddenValue' != 'BiasHiddenValue')
try:
    MLPReferencia()
    results['bug_default_ctor'] = 'nao quebrou?!'
except AttributeError as e:
    results['bug_default_ctor'] = f'AttributeError: {e}'
print('  params=None ->', results['bug_default_ctor'])

ref = MLPReferencia(BASE, seed=0)
h_ref = ref.fit(train_X, train_y, log_every=50)
acc_ref = ref.score(test_X, test_y)
# acuracia se o predict aplicasse a ativacao (prova de que a rede aprendeu
# e quem estraga e o predict "linear")
results['ref_test_acc'] = acc_ref
results['ref_test_acc_if_activated'] = float(
    np.mean(ref.predict_activated(test_X) == test_y))
print(f'  ref: test acc = {acc_ref:.3f}  |  se predict usasse ativacao: '
      f'{results["ref_test_acc_if_activated"]:.3f}  |  MSE real(final) = '
      f'{h_ref["train_loss"][-1]:.3f}')

fix = MultiLayerPerceptron(BASE, seed=0)
h_fix = fix.fit(train_X, train_y, log_every=50, verbose=False)
acc_fix_raw = fix.score(test_X, test_y)
results['fix_raw_test_acc'] = acc_fix_raw
print(f'  fixed (sem normalizar): test acc = {acc_fix_raw:.3f}')

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].plot(h_ref['epoch'], h_ref['reported_error'], c='#d62728',
           label='"total error" reportado')
ax[0].plot(h_ref['epoch'], h_ref['train_loss'], c='#1f77b4', ls='--',
           label='MSE real (mesma rede)')
ax[0].set_title('referencia: o que se ve vs. o que acontece')
ax[0].set_xlabel('epoca'); ax[0].set_ylabel('erro'); ax[0].legend(fontsize=8)
ax[1].plot(h_fix['epoch'], h_fix['train_loss'], c='#2ca02c',
           label=f'corrigido (acc={acc_fix_raw:.2f})')
ax[1].plot(h_ref['epoch'], h_ref['train_loss'], c='#d62728',
           label=f'referencia (acc={acc_ref:.2f})')
ax[1].set_title('MSE real por epoca, dados crus')
ax[1].set_xlabel('epoca'); ax[1].set_ylabel('MSE'); ax[1].legend(fontsize=8)
save(fig, 'debug_baseline.png')

# ---------------- 3. normalizacao ----------------
print('[3] efeito da normalizacao (z-score)')
fixN = MultiLayerPerceptron(BASE, seed=0)
h_fixN = fixN.fit(train_Xn, train_y, log_every=50, verbose=False)
acc_fixN = fixN.score(test_Xn, test_y)
results['fix_norm_test_acc'] = acc_fixN
print(f'  fixed normalizado: test acc = {acc_fixN:.3f}')

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(h_fix['epoch'], h_fix['train_loss'], c='#d62728',
        label=f'features crus (acc={acc_fix_raw:.2f})')
ax.plot(h_fixN['epoch'], h_fixN['train_loss'], c='#1f77b4',
        label=f'features normalizados (acc={acc_fixN:.2f})')
ax.set_xlabel('epoca'); ax.set_ylabel('MSE de treino')
ax.set_title('normalizacao z-score destrava a sigmoide')
ax.legend(fontsize=8)
save(fig, 'normalizacao.png')

# ---------------- 4. sweeps ----------------
print('[4] sweep de learning rate')
LRS = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
lr_table, lr_curves = {}, {}
for lr in LRS:
    p = dict(BASE, LearningRate=lr, Epocas=1200)
    accs, hist = [], None
    for s in range(3):
        m = MultiLayerPerceptron(p, seed=s)
        h = m.fit(train_Xn, train_y, log_every=25, verbose=False)
        accs.append(m.score(test_Xn, test_y))
        if s == 0:
            hist = h
    lr_table[lr] = (float(np.mean(accs)), float(np.std(accs)))
    lr_curves[lr] = hist
    print(f'  lr={lr:<6} acc={np.mean(accs):.3f} ± {np.std(accs):.3f}')
results['lr_sweep'] = {str(k): v for k, v in lr_table.items()}

fig, ax = plt.subplots(figsize=(6.5, 3.8))
cmap = plt.get_cmap('viridis')
for i, lr in enumerate(LRS):
    h = lr_curves[lr]
    ax.plot(h['epoch'], h['train_loss'], c=cmap(i / (len(LRS) - 1)),
            label=f'lr={lr} (acc={lr_table[lr][0]:.2f})')
ax.set_xlabel('epoca'); ax.set_ylabel('MSE de treino')
ax.set_title('sweep de learning rate (normalizado, hidden=5, sigmoid)')
ax.legend(fontsize=7.5)
save(fig, 'lr_sweep.png')

print('[5] sweep de neuronios escondidos')
HS = [1, 2, 3, 4, 5, 8, 16, 32]
h_table = {}
for h_ in HS:
    p = dict(BASE, HiddenLayer=h_, LearningRate=0.05, Epocas=1200)
    accs = []
    for s in range(3):
        m = MultiLayerPerceptron(p, seed=s)
        m.fit(train_Xn, train_y, log_every=999, verbose=False)
        accs.append(m.score(test_Xn, test_y))
    h_table[h_] = (float(np.mean(accs)), float(np.std(accs)))
    print(f'  H={h_:<3} acc={np.mean(accs):.3f} ± {np.std(accs):.3f}')
results['hidden_sweep'] = {str(k): v for k, v in h_table.items()}

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.errorbar(list(h_table), [v[0] for v in h_table.values()],
            yerr=[v[1] for v in h_table.values()], marker='o', capsize=3)
ax.set_xlabel('neuronios na camada escondida'); ax.set_ylabel('acc de teste')
ax.set_title('capacidade vs. acuracia (lr=0.05, sigmoid, 1200 epocas)')
ax.set_ylim(0.5, 1.02); ax.set_xscale('log', base=2)
save(fig, 'hidden_sweep.png')

print('[6] sweep de funcao de ativacao (+ ablacao de bias p/ relu)')
a_table = {}
for act, lr, bh in [('sigmoid', 0.05, -1), ('tanh', 0.05, -1),
                    ('relu', 0.05, -1), ('relu', 0.05, 0.0)]:
    accs = []
    for s in range(3):
        p = dict(BASE, LearningRate=lr, Epocas=1200, ActivationFunction=act,
                 BiasHiddenValue=bh, BiasOutputValue=bh)
        m = MultiLayerPerceptron(p, seed=s)
        m.fit(train_Xn, train_y, log_every=999, verbose=False)
        accs.append(m.score(test_Xn, test_y))
    key = f'{act} (bias={bh})'
    a_table[key] = (float(np.mean(accs)), float(np.std(accs)))
    print(f'  {key:<18} acc={np.mean(accs):.3f} ± {np.std(accs):.3f}')
results['act_sweep'] = a_table

# ---------------- 5. overfitting ----------------
print('[7] overfitting: treino vs validacao')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for k, (h_, col) in enumerate([(4, '#1f77b4'), (64, '#d62728')]):
    p = dict(BASE, HiddenLayer=h_, LearningRate=0.05, Epocas=2500)
    m = MultiLayerPerceptron(p, seed=0)
    h = m.fit(tr_X, tr_y, X_val=val_X, y_val=val_y, log_every=25, verbose=False)
    axes[0].plot(h['epoch'], h['train_loss'], c=col, label=f'H={h_} treino')
    axes[0].plot(h['epoch'], h['val_loss'], c=col, ls='--', label=f'H={h_} val')
    axes[1].plot(h['epoch'], h['train_acc'], c=col, label=f'H={h_} treino')
    axes[1].plot(h['epoch'], h['val_acc'], c=col, ls='--', label=f'H={h_} val')
    if h_ == 64:
        best_ep = h['epoch'][int(np.argmax(h['val_acc']))]
        results['overfit'] = {
            'H': h_, 'best_val_epoch': best_ep,
            'val_acc_best': max(h['val_acc']),
            'val_acc_final': h['val_acc'][-1],
            'train_acc_final': h['train_acc'][-1],
        }
        axes[1].axvline(best_ep, c='k', ls=':', alpha=.6)
axes[0].set_xlabel('epoca'); axes[0].set_ylabel('MSE')
axes[0].set_title('erro'); axes[0].legend(fontsize=7.5)
axes[1].set_xlabel('epoca'); axes[1].set_ylabel('acuracia')
axes[1].set_title('acuracia (linha pontilhada = melhor val de H=64)')
axes[1].legend(fontsize=7.5)
save(fig, 'overfitting.png')
print('  overfit (dados limpos):', results['overfit'])

# Iris limpo e separavel demais p/ overfit classico -> repete com 20% de
# ruido nos rotulos de treino: a rede grande memoriza o ruido e a val cai
print('[7b] overfitting com 20% de ruido nos rotulos')
rng = np.random.default_rng(3)
tr_y_noise = tr_y.copy()
idx_noise = rng.choice(len(tr_y), size=int(0.2 * len(tr_y)), replace=False)
tr_y_noise[idx_noise] = rng.integers(0, 3, size=len(idx_noise))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for h_, col in [(4, '#1f77b4'), (64, '#d62728')]:
    p = dict(BASE, HiddenLayer=h_, LearningRate=0.05, Epocas=2500)
    m = MultiLayerPerceptron(p, seed=0)
    h = m.fit(tr_X, tr_y_noise, X_val=val_X, y_val=val_y,
              log_every=25, verbose=False)
    axes[0].plot(h['epoch'], h['train_loss'], c=col, label=f'H={h_} treino*')
    axes[0].plot(h['epoch'], h['val_loss'], c=col, ls='--', label=f'H={h_} val')
    axes[1].plot(h['epoch'], h['train_acc'], c=col, label=f'H={h_} treino*')
    axes[1].plot(h['epoch'], h['val_acc'], c=col, ls='--', label=f'H={h_} val')
    if h_ == 64:
        ib = int(np.argmax(h['val_acc']))
        results['overfit_noise'] = {
            'best_val_epoch': h['epoch'][ib],
            'val_acc_best': h['val_acc'][ib],
            'val_acc_final': h['val_acc'][-1],
            'train_acc_final': h['train_acc'][-1],
        }
        axes[1].axvline(h['epoch'][ib], c='k', ls=':', alpha=.6)
axes[0].set_xlabel('epoca'); axes[0].set_ylabel('MSE')
axes[0].set_title('erro (* = medido contra rotulos ruidosos)')
axes[0].legend(fontsize=7.5)
axes[1].set_xlabel('epoca'); axes[1].set_ylabel('acuracia')
axes[1].set_title('acuracia — H=64 memoriza ruido, val cai')
axes[1].legend(fontsize=7.5)
save(fig, 'overfitting_ruido.png')
print('  overfit (ruido):', results['overfit_noise'])

# ---------------- 6. melhor modelo ----------------
print('[8] melhor config + avaliacao final')
BEST = dict(BASE, HiddenLayer=8, LearningRate=0.05, Epocas=800,
            ActivationFunction='sigmoid')
best = MultiLayerPerceptron(BEST, seed=0)
h_best = best.fit(train_Xn, train_y, X_val=test_Xn, y_val=test_y,
                  log_every=25, verbose=False)
acc_best = best.score(test_Xn, test_y)
results['best'] = {'params': BEST, 'test_acc': acc_best}
print(f'  best: test acc = {acc_best:.3f}')

# matriz de confusao + acuracia por classe
pred = best.predict(test_Xn)
cm = np.zeros((3, 3), dtype=int)
for t, p_ in zip(test_y, pred):
    cm[t, p_] += 1
per_class = cm.diagonal() / cm.sum(axis=1)
results['confusion'] = cm.tolist()
results['per_class_acc'] = per_class.tolist()

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
im = ax[0].imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, cm[i, j], ha='center', va='center',
                   c='white' if cm[i, j] > cm.max() / 2 else 'black')
ax[0].set_xticks(range(3), CLASS_NAMES, rotation=20)
ax[0].set_yticks(range(3), CLASS_NAMES)
ax[0].set_xlabel('predito'); ax[0].set_ylabel('real')
ax[0].set_title(f'matriz de confusao (acc={acc_best:.2f})')
ax[1].bar(CLASS_NAMES, per_class * 100, color=CLASS_COLORS)
ax[1].set_ylim(0, 105); ax[1].set_ylabel('acuracia %')
ax[1].set_title('acuracia por classe')
for i, v in enumerate(per_class):
    ax[1].text(i, v * 100 + 2, f'{v * 100:.0f}%', ha='center', fontsize=9)
save(fig, 'best_model.png')

# regioes de decisao nas 2 features de petala (rede treinada so com elas)
p2 = dict(BEST, InputLayer=2, HiddenLayer=8)
sc2 = StandardScaler().fit(train_X[:, 2:4])
m2 = MultiLayerPerceptron(p2, seed=0)
m2.fit(sc2.transform(train_X[:, 2:4]), train_y, log_every=999, verbose=False)
acc2 = m2.score(sc2.transform(test_X[:, 2:4]), test_y)
results['petal2_test_acc'] = acc2

xx, yy = np.meshgrid(
    np.linspace(X[:, 2].min() - .4, X[:, 2].max() + .4, 220),
    np.linspace(X[:, 3].min() - .4, X[:, 3].max() + .4, 220))
Z = m2.predict(sc2.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.contourf(xx, yy, Z, levels=2, colors=CLASS_COLORS, alpha=.18)
for c in range(3):
    m = test_y == c
    ax.scatter(test_X[m, 2], test_X[m, 3], s=45, c=CLASS_COLORS[c],
               edgecolors='k', linewidths=.5, label=CLASS_NAMES[c])
    mtr = train_y == c
    ax.scatter(train_X[mtr, 2], train_X[mtr, 3], s=14, c=CLASS_COLORS[c],
               alpha=.4)
ax.set_xlabel(iris.feature_names[2]); ax.set_ylabel(iris.feature_names[3])
ax.set_title(f'regioes de decisao (rede 2->8->3 nas petalas, acc={acc2:.2f})')
ax.legend(fontsize=8, loc='upper left')
save(fig, 'decision_regions.png')

# ---------------- sklearn de referencia ----------------
from sklearn.neural_network import MLPClassifier
sk = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000,
                   random_state=0).fit(train_Xn, train_y)
results['sklearn_test_acc'] = sk.score(test_Xn, test_y)
print(f'  sklearn MLPClassifier(8): test acc = {results["sklearn_test_acc"]:.3f}')

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nresults ->', os.path.join(OUT, 'results.json'))
