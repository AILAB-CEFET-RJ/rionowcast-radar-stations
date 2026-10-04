# Baseline B2a: Fluxo Óptico do Radar

`nowcasting-evaluate-optical-flow` implementa uma baseline CPU causal para
comparar a informação advectiva do radar com os modelos profundos. Ela usa
somente os cinco frames RGB de radar anteriores a cada instante previsto.

## Método

1. Converte RGB em um proxy visual de eco: `max(R,G,B)/255`.
2. Estima o movimento com Lucas-Kanade do `pysteps` e extrapola cinco passos
   de 15 minutos pelo método semilagrangiano.
3. Em cada horizonte e estação, extrai média, máximo e fração não nula em uma
   vizinhança de raio dois pixels.
4. Ajusta uma Ridge por horizonte no split de treino. As features incluem o
   identificador da estação; os alvos são `log1p(mm/15min)`.
5. Escolhe `alpha` pelo MAE da validação e reporta teste global, por horizonte
   e por intensidade.

O fluxo, as features e a predição usam exclusivamente radar anterior ao
instante previsto. Targets futuros são usados apenas para ajustar a Ridge no
treino e avaliar validação/teste. O proxy RGB não é refletividade calibrada.

## Dependências

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`pysteps` requer OpenCV; ambos constam de `requirements.txt`.

## Experimento multianual

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
run=B2a-optical-flow-audited-v3-$(date +%Y%m%d)

CUDA_VISIBLE_DEVICES="" nowcasting-evaluate-optical-flow \
  --dataset-root "$dataset" \
  --mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --mapping-height-orig 654 \
  --mapping-width-orig 656 \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --step 5 \
  --stride 5 \
  --neighborhood-radius 2 \
  --ridge-alphas 0.1,1,10 \
  --run-name "$run"
```

Os artefatos são salvos em `outputs/experiments/$run/`: configuração,
`summary.json` com a seleção de `alpha` e métricas, e
`ridge_readouts.joblib` com as cinco Ridge ajustadas.

## B2b: fluxo óptico + persistência

`nowcasting-evaluate-optical-flow-blend` combina B2a com B1. Para cada
estação, B1 repete a última chuva observada nos cinco instantes de entrada;
B2a fornece a leitura calibrada do campo advectado. A combinação é feita em
mm/15 min, não no espaço `log1p`:

```text
previsão = peso_fluxo * B2a + (1 - peso_fluxo) * B1
```

O peso e `alpha` são selecionados conjuntamente pelo MAE de 2022. Portanto,
a avaliação de 2023--2024 não participa da escolha de hiperparâmetros. Peso
zero significa persistência pura; peso um significa B2a pura.

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
run=B2b-optical-flow-persistence-audited-v3-$(date +%Y%m%d)

CUDA_VISIBLE_DEVICES="" nowcasting-evaluate-optical-flow-blend \
  --dataset-root "$dataset" \
  --mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --mapping-height-orig 654 \
  --mapping-width-orig 656 \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --step 5 \
  --stride 5 \
  --neighborhood-radius 2 \
  --ridge-alphas 0.1,1,10 \
  --blend-weights 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --run-name "$run"
```

## Critérios de comparação

Compare B2a e B2b com B1, C1 e C3 somente quando os arquivos de resumo declararem
os mesmos splits, continuidade temporal e número de pares válidos de teste.
Avalie especialmente os horizontes T+45, T+60 e T+75 e as faixas moderada,
forte e extrema: métricas globais são dominadas por períodos secos.
