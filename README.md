# Inferência Estatística
Trabalho de Inferência Estatística referente a primeira prova.
# AV1 — Introdução à Inferência I

Estudo de Monte Carlo comparando o **Estimador de Máxima Verossimilhança (EMV)** e o
**Estimador de Momentos (EMM)** dos parâmetros da distribuição **Lomax(α, λ)**.

Universidade Federal do Amazonas — Prof. Jeremias

---

## Estrutura

```
.
├── main.py          # script da simulação
├── requirements.txt     # dependências
├── README.md
└── (gerados após a execução)
    ├── resultados_lomax.csv
    └── vies_lomax.png
```

---

## 1. Pré-requisitos

- Python 3.10 ou superior

Verifique com:

```bash
python3 --version
```

---

## 2. Criar o ambiente virtual

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Com o ambiente ativo, o prompt passa a exibir o prefixo `(.venv)`.

---

## 3. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Ou, sem o arquivo de requisitos:

```bash
pip install numpy scipy pandas matplotlib
```

## 4. Executar

```bash
python lomax_mc.py
```

Tempo aproximado: **3 a 8 minutos** (5000 réplicas × 3 tamanhos amostrais × 3 valores
de α). Para um teste rápido, edite a linha `R = 5000` no script para `R = 200`.

Para desativar o ambiente ao final:

```bash
deactivate
```

---

## 5. O que é gerado

### Saída no terminal

Tabela com uma linha por combinação (método × parâmetro × α × n), contendo:

| coluna | significado |
|---|---|
| `metodo` | `EMV` ou `EMM` |
| `param` | `alpha` ou `lambda` |
| `alpha_real` | valor verdadeiro de α usado na geração |
| `n` | tamanho amostral (50, 200, 1000) |
| `media` | média das estimativas válidas nas réplicas |
| `vies` | `media` − valor verdadeiro |
| `eqm` | erro quadrático médio |
| `taxa_valida` | fração de réplicas em que o estimador existiu |

### `resultados_lomax.csv`

A mesma tabela em CSV, para uso em relatório ou pós-processamento.

### `vies_lomax.png`

Painel 2 × 3 com o viés em função de *n* (eixo em escala logarítmica):

- **linha superior** — viés de α̂; **linha inferior** — viés de λ̂
- **colunas** — α = 0,5 / α = 2 / α = 10
- curva azul (círculos) = EMV; curva laranja (quadrados, tracejada) = EMM
- a linha horizontal em zero marca a ausência de viés

---

## 6. Notas de implementação

**Geração das amostras.** Método da inversa: como
`F(x) = 1 − (1 + x/λ)^(−α)`, basta tomar `X = λ (U^(−1/α) − 1)` com `U ~ Uniforme(0,1)`.

**EMV por verossimilhança perfilada.** A log-verossimilhança da Lomax possui uma
crista quase plana (quando α → ∞ e λ → ∞ com λ/α fixo, a Lomax degenera numa
Exponencial), o que faz um BFGS bidimensional ingênuo divergir. O script substitui
α̂(λ) = n / S(λ), com S(λ) = Σ log(1 + xᵢ/λ), obtendo

```
ℓp(λ) = n·log(n) − n·log(S(λ)) − n·log(λ) − S(λ) − n
```

e otimiza em **uma** dimensão sobre log λ, com busca ancorada na **mediana** amostral
(a média não existe quando α ≤ 1). Réplicas cujo ótimo cai na fronteira do intervalo
são marcadas como não convergidas e entram em `taxa_valida`.

**EMM.** Resolvendo o sistema de momentos:

```
α̃ = 2(m₂ − x̄²) / (m₂ − 2x̄²)        λ̃ = x̄ (α̃ − 1)
```

que só é válido quando `m₂ > 2x̄²`, isto é, coeficiente de variação amostral maior
que 1. Fora dessa condição a réplica é descartada — o mesmo critério que governa a
existência do máximo interior da verossimilhança.

**Semente.** `np.random.default_rng(42)`, garantindo reprodutibilidade dos resultados.