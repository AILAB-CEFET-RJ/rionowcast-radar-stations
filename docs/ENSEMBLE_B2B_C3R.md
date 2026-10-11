# Ensemble Causal B2b + C3R

## Objetivo

O B2b tem menor RMSE e melhor MAE em T+45--T+75, enquanto C3R é superior em
T+15 e em CSI bruto. Este ensemble combina os dois somente depois de ambos
terem produzido previsões canônicas por estação, sem retreinar nenhum modelo.

Para cada horizonte `h`, a previsão é:

```text
previsão[h] = (1 - w[h]) * B2b[h] + w[h] * C3R[h]
```

Os cinco pesos `w[h]` são selecionados exclusivamente na validação de 2022,
por MAE ou RMSE. A mesma regra é aplicada sem ajuste no teste 2023--2024.
Combinar em mm/15 min, e não em `log1p`, preserva a interpretação física da
mistura.

## Execução por seed

O B2b é determinístico; cada seed C3R gera um ensemble independente.

```bash
b2=outputs/evaluation_records/B2-audited-v3
c3r=outputs/evaluation_records/C3R-audited-v3-seeds
records=outputs/evaluation_records/B3-horizon-ensemble
report=outputs/analysis/evaluation/B3-horizon-ensemble
mkdir -p "$records/validation" "$records/test" "$report"

for seed in 1000 1001 1002; do
  run="B3-B2b-C3R-seed-$seed"
  PYTHONPATH=src python -u -m nowcasting.cli.blend_forecast_records \
    --validation-baseline "$b2/validation/B2b.parquet" \
    --validation-candidate "$c3r/validation/C3R-seed-$seed.parquet" \
    --test-baseline "$b2/test/B2b.parquet" \
    --test-candidate "$c3r/test/C3R-seed-$seed.parquet" \
    --candidate-weights 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
    --selection-metric mae \
    --experiment-id "$run" \
    --validation-output "$records/validation/$run.parquet" \
    --test-output "$records/test/$run.parquet" \
    --summary "$report/$run.json"
done
```

Depois, calibre limiares somente em 2022 e avalie os eventos em 2023--2024,
como nos demais modelos. A seleção de pesos e a seleção de limiares são etapas
distintas: a primeira minimiza erro contínuo; a segunda transforma a previsão
fixada em alerta.

## Variante orientada a alerta municipal (B3A)

Para uma comparação operacional distinta do B3, o objetivo
`municipal-csi` seleciona em 2022 um **único** peso global de C3R. Para
cada peso candidato, calibra um limiar de decisão por intensidade e mede o CSI
depois de agregar a previsão máxima entre as estações para cada
timestamp--horizonte. O peso que maximiza a média de CSI nos três limiares é
selecionado; `--minimum-pod` pode impor uma sensibilidade mínima em cada
limiar. Esse modo não altera o B3 por MAE e deve ser tratado como uma ablação
orientada a alerta.

```bash
alert_records=outputs/evaluation_records/B3A-municipal-csi
alert_report=outputs/analysis/evaluation/B3A-municipal-csi
mkdir -p "$alert_records/validation" "$alert_records/test" "$alert_report"

for seed in 1000 1001 1002; do
  run="B3A-B2b-C3R-seed-$seed"
  PYTHONPATH=src python -u -m nowcasting.cli.blend_forecast_records \
    --validation-baseline "$b2/validation/B2b.parquet" \
    --validation-candidate "$c3r/validation/C3R-seed-$seed.parquet" \
    --test-baseline "$b2/test/B2b.parquet" \
    --test-candidate "$c3r/test/C3R-seed-$seed.parquet" \
    --selection-objective municipal-csi \
    --thresholds 1.25,6.25,12.5 \
    --minimum-pod 0.20 \
    --experiment-id "$run" \
    --validation-output "$alert_records/validation/$run.parquet" \
    --test-output "$alert_records/test/$run.parquet" \
    --summary "$alert_report/$run.json"
done
```

Ao calibrar os limiares para B3A, usar também `--scope municipal` em
`nowcasting.cli.calibrate_alert_thresholds`; isso mantém a unidade de seleção
do limiar consistente com o objetivo da variante.
