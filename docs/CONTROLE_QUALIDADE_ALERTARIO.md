# Controle De Qualidade AlertaRio

## Fonte Original

Os Parquets originais ficam em `data/pluviometricos_alertario/`. Eles podem
conter registros de várias estações e anos no mesmo arquivo. As colunas usadas
no pipeline são `estacao_id`, `dia_utc` e `m15`.

O valor `-99.99` em `m15` é uma sentinela de ausência. Ele, valores `NaN` e
valores negativos não são observações de chuva e devem permanecer fora da
máscara de targets, da loss e das métricas.

## Auditoria Da Fonte

```bash
nowcasting-audit-alertario \
  --input-root data/pluviometricos_alertario \
  --output-dir outputs/analysis/alertario/raw_audit_v1
```

O comando gera resumos por estação/ano e por estação, com cobertura temporal,
ausências, sentinela, negativos, valores acima de 50 e 175 mm/15 min, além de
timestamps exatamente duplicados. O relatório também registra múltiplas
leituras no mesmo intervalo de 15 minutos; estas são esperadas para séries em
maior frequência e são agregadas por máximo. A geração de targets consolida
registros entre todos os arquivos por estação e intervalo de 15 minutos.

## Targets Com Proveniência

Os targets novos devem ser construídos numa raiz de dataset separada, já
contendo os frames e timestamps do radar. Use hard links para não duplicar os
18 GiB de frames quando origem e destino estiverem no mesmo sistema de
arquivos:

```bash
source=data/datasets/radar_sumare_2012_2024_15min_128_sparse_rebuilt
target=data/datasets/radar_sumare_2012_2024_15min_128_alertario_v2

cp -al "$source" "$target"
find "$target" -type f \( -name 'targets_alertario_sparse.npz' -o \
  -name 'targets_alertario_metadata.json' \) -delete
```

Em seguida, gere os targets a partir dos Parquets originais:

```bash
nowcasting-build-alertario-sparse-targets \
  --alertario-root data/pluviometricos_alertario \
  --mapping data/mapeamento_pixel_estacao_alertario.csv \
  --radar-root "$source" \
  --output-root "$target" \
  --year-start 2012 \
  --year-end 2024 \
  --height 128 \
  --width 128
```

O novo `targets_alertario_sparse.npz` inclui os campos usuais `frame`, `row`,
`column` e `value`, mais `station_id`. O `StationSequenceDataset` usa esse
campo quando disponível; portanto, o baseline por estações não depende mais
de reconstruir a identidade a partir da coordenada espacial.

Não sobrescreva `radar_sumare_2012_2024_15min_128_sparse_rebuilt`: ele é a
referência dos experimentos C1 e C2 já concluídos.
