# A1: Atenção Temporal Condicionada ao Horizonte

## Objetivo

`A1-attn` testa se agregação temporal adaptativa melhora a correção residual
da persistência. A comparação usa os mesmos dados, seeds, loss e protocolo de
`A0` e `A0-cap`; somente o adaptador entre o bloco espacial e o gerador muda.

As variantes são:

- `A0`: `--model stconvs2s-c`;
- `A0-cap`: `--model stconvs2s-c-cap`;
- `A1-attn`: `--model stconvs2s-c-attn`.

## Arquitetura

Depois do bloco espacial, cada célula da grade produz uma sequência de cinco
vetores com `C=32` canais. A1 aplica LayerNorm e atenção temporal multi-head de
quatro cabeças independentemente em cada célula. Uma consulta aprendida para
cada horizonte é somada à query, enquanto keys e values vêm dos estados
normalizados:

```text
q_t = LN(z_t) + e_t
k_t = v_t = LN(z_t)
z_attn = z + gamma * MHA(q, k, v)
```

`gamma` é treinável e começa em zero. Assim, A0 e A1 têm a mesma saída antes
da primeira atualização e os parâmetros comuns recebem a mesma inicialização
para uma mesma seed. Não há atenção espacial global: células diferentes não
são transformadas em uma sequência, evitando custo quadrático na área.

O experimento está definido para cinco entradas e cinco horizontes. Todos os
frames de entrada já estão disponíveis no instante da previsão; a atenção não
recebe targets nem frames futuros.

## Controle de parâmetros

No protocolo atual:

| Variante | Parâmetros totais | Acréscimo sobre A0 |
|---|---:|---:|
| A0 | 425.697 | 0 |
| A0-cap | 429.954 | 4.257 |
| A1-attn | 430.146 | 4.449 |

A1 adiciona 192 parâmetros a mais que A0-cap, diferença de 4,51% entre os
orçamentos adicionais. Isso satisfaz o limite pré-definido de 5%.

## Smoke test

Na workstation02, depois de atualizar a branch e o submódulo:

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate ailab

dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv
run="smoke-A1-attn-$(date +%Y%m%d-%H%M%S)"

PYTHONPATH=src python -u -m nowcasting.cli.train \
  --dataset-root "$dataset" \
  --train-years 2021 --val-years 2022 --test-years 2023 \
  --target-source alertario \
  --crop-stations --crop-margin-pixels 20 \
  --station-mapping "$mapping" \
  --mapping-height-orig 654 --mapping-width-orig 656 \
  --input-stations --forecast-formulation residual-persistence \
  --model stconvs2s-c-attn \
  --batch-size 2 --epochs 1 --patience 1 \
  --step 5 --stride 5 --workers 2 --persistent-workers \
  --loss masked-huber --cuda 0 --seed 1000 \
  --skip-test --run-name "$run"
```

O log deve registrar `Model | parameters=430146` e terminar com
`test=skipped`. Em CUDA, a linha final e o `summary.json` também registram os
picos de memória alocada e reservada.

## Seleção em 2022

Execute somente A1 para as mesmas seeds já usadas em A0 e A0-cap:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

common=(
  --dataset-root "$dataset"
  --train-years 2012-2021 --val-years 2022 --test-years 2023-2024
  --target-source alertario
  --crop-stations --crop-margin-pixels 20
  --station-mapping "$mapping"
  --mapping-height-orig 654 --mapping-width-orig 656
  --input-stations --forecast-formulation residual-persistence
  --model stconvs2s-c-attn
  --batch-size 2 --epochs 30 --patience 10
  --step 5 --stride 5 --workers 2 --persistent-workers
  --log-interval 250 --loss masked-huber
  --cuda 0 --checkpoint-every 1 --skip-test
)

for seed in 1000 1001 1002; do
  PYTHONPATH=src python -u -m nowcasting.cli.train \
    "${common[@]}" --seed "$seed" \
    --run-name "A1-attn-selection-seed-$seed"
done
```

## Decisão

A1 só avança para 2023--2024 se a validação apresentar ganho que não dependa
de uma única seed. O mínimo é vencer A0 e A0-cap em ao menos duas das três
seeds, além de reduzir média e mediana. A análise deve registrar também tempo
por época e pico de memória da GPU. Até essa decisão, `--skip-test` permanece
obrigatório.
