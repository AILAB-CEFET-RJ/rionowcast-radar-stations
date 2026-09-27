# Pipeline Auditavel do Radar Sumare

Este pipeline reconstrui memmaps de radar a partir dos PNGs brutos sem alterar
datasets historicos existentes. O contrato inicial esta em
`configs/radar_sumare_historical_audited_v1.json`.

## Etapas

1. Auditar os PNGs antes da geracao:

```bash
nowcasting-audit-radar-pngs \
  --data-root data/raw/radar_sumare \
  --output-dir outputs/analysis/radar_sumare/png_audit_v1 \
  --year-start 2023 --year-end 2024 \
  --capture-config configs/radar_sumare_historical_audited_v1.json \
  --verify-images
```

O comando produz cobertura de cada janela de 15 minutos, lacunas entre janelas
elegiveis e PNGs cropados para inspecao visual. A configuracao e provisoria:
qualquer geometria inesperada encerra a verificacao ou a geracao.

2. Gerar um piloto isolado:

```bash
nowcasting-build-radar \
  --data-root data/raw/radar_sumare \
  --output-root data/datasets/radar_sumare_2023_2024_15min_128_audited_v3 \
  --year-start 2023 --year-end 2024 \
  --capture-config configs/radar_sumare_historical_audited_v1.json \
  --height 128 --width 128 --aggregate-minutes 15 \
  --min-frames-per-window 6 --aggregation rgb-max
```

Cada ano e construido em `year=AAAA.partial` e publicado apenas ao final. O
memmap usa somente janelas elegiveis, enquanto `window_coverage.csv` preserva
todas as janelas previstas no calendario. `metadata.json` inclui hash da
configuracao, estatisticas das fontes e `enforce_timestamp_continuity=true`.

## Continuidade temporal

`RadarStationMemmapDataset` aplica continuidade apenas quando este marcador
esta presente. Uma amostra com cinco passos de entrada e cinco de previsao e
descartada se qualquer intervalo entre frames diferir de 15 minutos. Datasets
legados permanecem inalterados.

## Convenção temporal confirmada

Os nomes dos PNGs históricos representam hora local de `America/Sao_Paulo`.
O gerador converte-os para UTC, inclusive nos períodos históricos de horário de
verão. Os registros `m15` do AlertaRio são acumulados cujo timestamp marca o
fim do intervalo; portanto, para associá-los ao timestamp inicial da janela de
radar, gere targets com `--observation-to-radar-offset-minutes -15`.

Não use os datasets legados que tratam o nome local do PNG como UTC para
avaliar nowcasting baseado em radar.

## Sazonalidade de treinamento

Mantenha todos os meses no dataset canônico. Para decidir se a amostragem de
treinamento deve dar mais peso aos meses chuvosos, use as janelas efetivas do
loader, e não somente registros brutos de estação:

```bash
nowcasting-audit-training-seasonality \
  --dataset-root data/datasets/radar_sumare_2012_2024_15min_128_audited_v3 \
  --years 2012-2021 \
  --target-source alertario \
  --t-in 5 --t-out 5 --stride 5 \
  --output-dir outputs/analysis/alertario/training_seasonality_v1
```

Os CSVs resultantes preservam contagens por mês, ano e classe do balanced
sampler. A aplicação de pesos sazonais é um experimento de treinamento; ela não
deve eliminar meses da geração canônica ou do conjunto de teste.

## Alinhamento das estações

Os PNGs históricos auditados medem 654 linhas por 656 colunas, enquanto o
mapeamento legado declara 656 por 654. Antes de gerar targets, compare as
transformações candidatas com eventos chuvosos do AlertaRio:

```bash
nowcasting-audit-historical-station-mapping \
  --data-root data/raw/radar_sumare \
  --alertario-root data/pluviometricos_alertario \
  --mapping data/mapeamento_pixel_estacao_alertario.csv \
  --capture-config configs/radar_sumare_historical_audited_v1.json \
  --output-dir outputs/analysis/radar_sumare/historical_mapping_audit_2023 \
  --year 2023 --max-events 100 --min-rain-mm 1.25
```

O ranking usa a presença de eco em uma vizinhança de cada estação e deve ser
revisado junto com os overlays. Ele é evidência para selecionar uma
transformação, não uma autorização automática para gerar targets.

Repita a auditoria com offsets de radar de `-30`, `-15`, `0`, `+15` e `+30`
minutos, e confirme a hipótese selecionada em um ano diferente. O `m15` pode
representar o fim do intervalo acumulado, e essa convenção altera a associação
entre chuva de estação e eco de radar.

Quando essa validação não for estável entre anos, compare primeiro a presença
e a extensão dos ecos em eventos selecionados pelas estações, sem assumir que
RGB seja uma medida física de refletividade:

```bash
nowcasting-audit-historical-radar-events \
  --data-root data/raw/radar_sumare \
  --alertario-root data/pluviometricos_alertario \
  --mapping data/mapeamento_pixel_estacao_alertario.csv \
  --capture-config configs/radar_sumare_historical_audited_v1.json \
  --output-dir outputs/analysis/radar_sumare/event_echo_comparison_2023_2024_v1 \
  --year-start 2023 --year-end 2024 \
  --radar-offset-minutes -15 \
  --max-events 100 --preview-count 12
```

O comando grava métricas por evento em `event_echo_metrics.csv` e previews dos
episódios de maior chuva por ano. Ele detecta, por exemplo, mudanças de paleta,
ausência de eco, ou diferença de cobertura antes de atribuir a divergência a um
mapeamento espacial.

### Candidato geográfico físico

O contrato histórico registra o centro oficial do radar no Sumaré e o raio de
`138,9 km` informado pela operação para o produto de imagem. A hipótese
geográfica usa uma projeção azimutal equidistante centrada no radar, norte no
topo e leste à direita. Ela é deliberadamente um candidato, e não substitui a
validação chuva-sinal:

```bash
nowcasting-build-historical-station-mapping \
  --input-mapping data/mapeamento_pixel_estacao_alertario.csv \
  --capture-config configs/radar_sumare_historical_audited_v1.json \
  --output-mapping outputs/analysis/radar_sumare/geographic_mapping_candidate_v1.csv
```

Os oito candidatos geográficos, identificados por `geographic_aeqd_*`, aparecem
automaticamente em `candidate_scores.csv`. Eles preservam centro e escala
físicos, mas testam as permutações e espelhamentos de eixos possíveis em um PNG
sem metadados GIS. As auditorias independentes de 2023 e 2024 selecionaram
`geographic_aeqd_north_up` com offset de `-195` minutos. O mapeamento validado
está em `configs/mapeamento_pixel_estacao_alertario_historical_v1.csv`.

## Limite conhecido

`rgb-max` replica a agregacao visual adotada no pipeline legado, mas nao e uma
variavel fisica de refletividade. Ele deve ser tratado como baseline de
compatibilidade. Uma calibracao da paleta RGB para refletividade ou taxa de
chuva e uma etapa posterior, com validacao meteorologica.
