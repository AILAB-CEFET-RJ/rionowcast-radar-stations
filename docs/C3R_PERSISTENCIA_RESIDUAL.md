# C3R: Persistência Residual com Radar

## Objetivo

O C3R preserva a persistência causal por estação como previsão base e treina a
STConvS2S para prever somente uma correção espacial guiada pelo radar:

```text
previsão = persistência B1 + correção STConvS2S(radar[, GOES])
```

O último valor válido em cada histórico de cinco janelas é repetido nos cinco
horizontes. Uma estação sem histórico válido recebe base zero. Os valores ficam
na mesma escala interna `log1p(mm/15 min)` dos targets; não há conversão ou
uso de observações futuras.

## Diferença para C3

O C3 concatena chuva lagada e máscara como dois mapas esparsos à entrada da
STConvS2S. No C3R, esses dois canais não entram na rede convolucional: servem
somente para construir o atalho determinístico de persistência. A STConvS2S
recebe radar RGB e, quando configurado, GOES. Isso evita exigir que a rede
reaprenda a copiar valores pontuais de estações pela arquitetura espacial.

## Comando de referência

```bash
python -u -m nowcasting.cli.train \
  --dataset-root data/datasets/radar_sumare_2012_2024_15min_128_audited_v3 \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --target-source alertario \
  --crop-stations --crop-margin-pixels 20 \
  --station-mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --mapping-height-orig 654 --mapping-width-orig 656 \
  --input-stations \
  --forecast-formulation residual-persistence \
  --model stconvs2s-c \
  --loss masked-huber \
  --batch-size 2 --workers 2 \
  --step 5 --stride 5 \
  --epochs 80 --patience 10 \
  --cuda 0 \
  --run-name C3R-residual-persistence-audited-v3-seed-1000
```

O checkpoint deve ser exportado pelo mesmo
`nowcasting-export-radar-checkpoint-records` usado em C1/C2/C3. O exportador
reconstrói a formulação por `forecast_formulation` gravada em
`configuration.json`.

## Avaliação

Compare C3R com B1, C1 e C3 sobre registros canônicos alinhados. O critério não
é apenas MAE global: reportar métricas por horizonte e intensidade, skill contra
B1, métricas categóricas, bootstrap diário e eventos. O C3R é promissor se
preservar B1 em T+15 e acrescentar skill favorável nos horizontes posteriores.
