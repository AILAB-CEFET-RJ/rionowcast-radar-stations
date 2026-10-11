# Ablação Controlada C4R com GOES-16 C13

## Pergunta experimental

O canal infravermelho C13 do GOES-16 acrescenta informação preditiva ao
modelo residual que já recebe radar RGB e persistência causal das estações?

O experimento não compara C13 a um C3R treinado em mais anos. Ele compara:

- **C3R-C13-control**: radar RGB e persistência, filtrado pela mesma máscara
  de disponibilidade causal de C13, mas sem C13 na entrada;
- **C4R**: a mesma configuração, com C13 concatenado aos três canais RGB do
  radar antes da STConvS2S.

Em ambos, a previsão final é persistência causal por estação mais a correção
da rede. Portanto, a única diferença entre os braços é a presença de C13.

## Dados e auditoria

O recorte usa treino em 2018--2021, validação em 2022 e teste em 2023--2024.
Os anos anteriores a 2018 não entram porque ainda não há C13 processado para
eles. A auditoria estrita em `outputs/analysis/goes16/c13_2018_2024_v2`
confirmou cobertura de 99,46% e zero violações da regra de causalidade por
fim de cena.

A máquina de treinamento deve conter o diretório `goes16` de **todos** os
anos do recorte, de 2018 a 2024. C13 é consultado no treino, na validação e
no teste; copiar apenas os anos de treino faz a construção do split de
validação falhar antes de iniciar as épocas.

Para cada sequência são exigidos cinco frames C13 causais na entrada. Isso
produz os seguintes tamanhos, iguais nos dois braços:

| Split | Sequências antes do filtro C13 | Sequências pareadas |
|---|---:|---:|
| Treino, 2018--2021 | 18.949 | 18.689 |
| Validação, 2022 | 5.031 | 4.982 |
| Teste, 2023--2024 | 9.261 | 9.147 |

O argumento `--require-goes-availability` implementa o filtro no controle sem
concatenar C13 à entrada. Ele é obrigatório para que qualquer diferença não
seja atribuível a mudanças nas sequências avaliadas.

## Execução da primeira seed

Execute primeiro os dois braços com a seed 1000. Os logs devem reportar os
mesmos totais de amostras acima. C4R deve reportar quatro canais para a
STConvS2S (RGB + C13), enquanto o controle deve reportar três.

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

common=(
  --dataset-root "$dataset"
  --train-years 2018-2021
  --val-years 2022
  --test-years 2023-2024
  --target-source alertario
  --crop-stations
  --crop-margin-pixels 20
  --station-mapping "$mapping"
  --mapping-height-orig 654
  --mapping-width-orig 656
  --input-stations
  --forecast-formulation residual-persistence
  --model stconvs2s-c
  --batch-size 2
  --epochs 30
  --patience 10
  --step 5
  --stride 5
  --workers 2
  --persistent-workers
  --log-interval 250
  --loss masked-huber
  --cuda 0
  --checkpoint-every 1
  --seed 1000
)

PYTHONPATH=src python -u -m nowcasting.cli.train \
  "${common[@]}" \
  --require-goes-availability \
  --run-name C3R-C13-control-seed-1000

PYTHONPATH=src python -u -m nowcasting.cli.train \
  "${common[@]}" \
  --input-goes \
  --run-name C4R-radar-goes-c13-seed-1000
```

Após confirmar que ambos terminam e exportam previsões canônicas de validação
e teste, repetir para as seeds 1001 e 1002. Checkpoints, pesos de ensemble e
limiares de alerta devem ser selecionados somente com 2022; 2023--2024 fica
reservado para a comparação final.
