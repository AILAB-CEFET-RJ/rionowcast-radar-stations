# Avaliação Auditável de Previsões

Esta infraestrutura compara B1, B2a, B2b e checkpoints STConvS2S usando
registros canônicos por estação, instante-alvo e horizonte. Ela não assume que
2024 seja o último ano: os anos são sempre informados nos comandos e o
timestamp armazenado identifica cada previsão. Quando 2025--2026 estiverem
disponíveis no dataset auditável, poderão compor splits posteriores sem mudar o
formato dos artefatos.

## Contrato de registros

Cada Parquet contém uma linha por chave
`(year, target_timestamp, horizon, station_id)` e as colunas
`predicted_mm_15min`, `observed_mm_15min` e `is_observed`. O avaliador rejeita
comparações com chaves, máscara ou observações diferentes. Logo, igualdade de
`n` não substitui alinhamento efetivo dos pares.

## Exportar previsões

Exemplo para o teste cronológico atual:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
records=outputs/evaluation_records
mkdir -p "$records"

nowcasting-export-station-persistence-records \
  --dataset-root "$dataset" \
  --years 2023-2024 \
  --mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --mapping-height-orig 654 --mapping-width-orig 656 \
  --step 5 --stride 5 \
  --experiment-id B1 \
  --output "$records/B1.parquet"
```

B2a e B2b exportam no próprio comando por `--forecast-records`:

```bash
nowcasting-evaluate-optical-flow \
  --dataset-root "$dataset" \
  --train-years 2012-2021 --val-years 2022 --test-years 2023-2024 \
  --mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --mapping-height-orig 654 --mapping-width-orig 656 \
  --step 5 --stride 5 \
  --run-name B2a-optical-flow-audited-v3 \
  --forecast-records "$records/B2a.parquet"
```

Para C1, C2 ou C3 já treinado, use o checkpoint da melhor época:

```bash
nowcasting-export-radar-checkpoint-records \
  --experiment-dir outputs/experiments/C3-exemplo \
  --checkpoint outputs/experiments/C3-exemplo/iteration_1_best.pt \
  --experiment-id C3 \
  --output "$records/C3.parquet" \
  --batch-size 8 --cuda 0
```

O exportador reconstrói o dataset, crop, entradas de estações e entradas GOES
a partir de `configuration.json`; não retreina o modelo.

## Métricas contínuas, categóricas e incerteza

```bash
nowcasting-evaluate-forecast-records \
  --forecast B1="$records/B1.parquet" \
  --forecast B2a="$records/B2a.parquet" \
  --forecast B2b="$records/B2b.parquet" \
  --forecast C1="$records/C1.parquet" \
  --forecast C3="$records/C3.parquet" \
  --baseline B1 \
  --thresholds 1.25,6.25,12.5 \
  --bootstrap-replicates 2000 \
  --output-dir outputs/analysis/evaluation/audited_v1
```

O comando produz MAE, RMSE e viés globais, por horizonte e por intensidade;
POD, FAR, razão de sucesso, CSI, viés de frequência, ETS e HSS por limiar;
intervalos de confiança de bootstrap pareado por dia local; e um diagrama de
desempenho por limiar. O bootstrap reamostra dias completos em
`America/Sao_Paulo`, preservando a dependência entre estações, horizontes e
janelas de um mesmo dia.

## Eventos e limiares operacionais

```bash
nowcasting-evaluate-forecast-events \
  --forecast B1="$records/B1.parquet" \
  --forecast C3="$records/C3.parquet" \
  --thresholds 6.25,12.5 \
  --event-gap-minutes 15 \
  --output outputs/analysis/evaluation/audited_v1/events.json
```

Os eventos são definidos por estação como excedências observadas contíguas. A
saída informa fração de eventos detectados, antecedência disponível e falsos
alertas por registro. Ela não é uma métrica espacial de campo; FSS permanece
fora do escopo enquanto não houver precipitação observada em grade densa.

Para calibrar um limiar de alerta, forneça Parquets de validação e de teste
separados. O comando seleciona o limiar apenas na validação e o congela no
teste:

```bash
nowcasting-calibrate-alert-thresholds \
  --validation-forecast B1="$records/val_B1.parquet" \
  --validation-forecast C3="$records/val_C3.parquet" \
  --test-forecast B1="$records/B1.parquet" \
  --test-forecast C3="$records/C3.parquet" \
  --output outputs/analysis/evaluation/audited_v1/alert_calibration.json
```

Essa calibração afeta somente a decisão categórica de alerta. MAE, RMSE e viés
devem continuar sendo apresentados sem ajuste posterior.
