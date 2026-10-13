# C6R: Supervisao Auxiliar WebSirene

## Objetivo

Avaliar se os pluviometros WebSirene auditados melhoram o treinamento do
STConvS2S sem substituir o AlertaRio como referencia experimental. O modelo
recebe as mesmas entradas do C3R: cinco frames RGB de radar e cinco valores
passados com mascaras das estacoes AlertaRio. O WebSirene entra somente na
funcao de perda durante o treino.

Para cada batch de treino, a otimizacao e:

```text
loss total = loss mascarada AlertaRio + peso WebSirene * loss mascarada WebSirene
```

Cada termo e normalizado pela soma de sua propria mascara. Assim, a maior
quantidade de leituras WebSirene nao aumenta implicitamente sua contribuicao.
Quando as duas redes observam uma mesma celula e instante, ambas contribuem
com seu proprio erro; os valores nao sao combinados em um target unico.

Validacao, early stopping, selecao de peso e teste usam exclusivamente o
AlertaRio. Portanto, a comparacao permanece diretamente comparavel aos
experimentos C3R.

## Pre-requisitos

- Branch `feature/websirene-auxiliary-supervision`.
- Targets AlertaRio e WebSirene presentes para todos os anos usados.
- Targets WebSirene gerados da whitelist auditada V1, com offset temporal zero
  e agregacao `median` quando houver mais de uma estacao por celula.

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3

for year in $(seq 2012 2024); do
  test -f "$dataset/year=$year/targets_alertario_sparse.npz" &&
  test -f "$dataset/year=$year/targets_websirene_sparse.npz" &&
  echo "$year: OK" || echo "$year: AUSENTE"
done
```

## Controle Pareado e Busca de Peso

Use 2012--2021 para treino, 2022 para validacao e 2023--2024 somente para o
teste final. Primeiro, execute o controle C3R na mesma configuracao. Depois,
para a seed 1000, compare pesos `0.25`, `0.5` e `1.0` unicamente por
`best_val_loss` de 2022. Nao consulte 2023--2024 para escolher o peso.

```bash
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

common=(
  --dataset-root "$dataset"
  --train-years 2012-2021 --val-years 2022 --test-years 2023-2024
  --target-source alertario
  --crop-stations --crop-margin-pixels 20
  --station-mapping "$mapping" --mapping-height-orig 654 --mapping-width-orig 656
  --input-stations
  --forecast-formulation residual-persistence
  --model stconvs2s-c
  --batch-size 2 --epochs 30 --patience 10
  --step 5 --stride 5
  --workers 2 --persistent-workers
  --log-interval 250 --loss masked-huber
  --cuda 0 --checkpoint-every 1
  --seed 1000
)

PYTHONPATH=src python -u -m nowcasting.cli.train \
  "${common[@]}" \
  --run-name C6R-alertario-control-seed-1000

for weight in 0.25 0.5 1.0; do
  PYTHONPATH=src python -u -m nowcasting.cli.train \
    "${common[@]}" \
    --auxiliary-target-source websirene \
    --auxiliary-loss-weight "$weight" \
    --run-name "C6R-websirene-aux-w${weight}-seed-1000"
done
```

O historico de cada run registra `train_primary_loss` e
`train_auxiliary_loss`. `val_loss` continua sendo apenas a loss mascarada do
AlertaRio.

Depois de fixar o peso em 2022, repita controle e candidata com as seeds 1000,
1001 e 1002. Exporte as previsoes e aplique o mesmo avaliador pareado usado
nos experimentos C3R, sempre contra os targets AlertaRio de 2023--2024.

## Interpretacao

Aceite o WebSirene apenas se o ganho for reproduzivel entre seeds e sustentado
por metricas de alerta, horizonte e intensidade no teste cronologico. Uma
reducao isolada de MAE global nao basta, pois a maior parte das observacoes e
seca ou fraca.
