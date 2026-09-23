# Geracao Do Dataset

Este guia descreve o fluxo para gerar o dataset anual usado pelo treinamento:
frames de radar agregados em 15 minutos e targets de estacoes alinhados aos
timestamps do radar.

## Estrutura De Dados

Use caminhos relativos ao checkout:

```text
data/
  raw/
    radar_sumare/                         PNGs por ano/mes/dia
    alertario/pluviometricos_parquet/     Parquets AlertaRio preparados
    websirene/                            Parquets WebSirene por estacao/ano
  mapeamento_pixel_estacao_alertario.csv
  datasets/
    radar_sumare_2012_2024_15min_128_sparse/
```

Os dados brutos e datasets grandes sao ignorados pelo Git. Consulte
[`docs/ORGANIZACAO_DADOS_E_RESULTADOS.md`](docs/ORGANIZACAO_DADOS_E_RESULTADOS.md)
para a classificacao completa.

## Preparacao Do Ambiente

```bash
conda activate ailab
python -m pip install -e .
```

Os Parquets AlertaRio devem conter pelo menos `estacao_id`, `estacao`,
`dia_utc` e `m15`. Alguns arquivos recebidos com extensao `.parquet` possuem
conteudo CSV; converta-os para Parquet verdadeiro antes desta etapa. A EDA de
fonte esta em `notebooks/01_eda/01_alertario_eda.ipynb`, mas conversoes devem
ser implementadas como scripts reproduziveis, nunca em notebook.

## 1. Gerar Frames De Radar

O comando le os PNGs, agrega janelas de 15 minutos e grava memmaps por ano:

```bash
nowcasting-build-radar \
  --data-root data/raw/radar_sumare \
  --output-root data/datasets/radar_sumare_2012_2024_15min_128_sparse \
  --year-start 2012 \
  --year-end 2024 \
  --aggregate-minutes 15 \
  --height 128 \
  --width 128 \
  --min-frames-per-window 3
```

Para cada ano, sao produzidos `radar_frames.dat`, `radar_timestamps.npy` e
`metadata.json`.

## 2. Gerar Targets AlertaRio

O mapeamento de estacoes para pixels fica em
`data/mapeamento_pixel_estacao_alertario.csv`. Gere os targets alinhados aos
timestamps ja produzidos:

```bash
nowcasting-build-alertario-targets \
  --alertario-root data/raw/alertario/pluviometricos_parquet \
  --mapping data/mapeamento_pixel_estacao_alertario.csv \
  --radar-root data/datasets/radar_sumare_2012_2024_15min_128_sparse \
  --year-start 2012 \
  --year-end 2024 \
  --height 128 \
  --width 128
```

Esse comando produz temporariamente `Y_alertario.dat`, `M_alertario.dat` e
`targets_alertario_metadata.json` por ano. Os valores de precipitacao usam
`log1p(mm/15min)` e a mascara indica onde existe uma observacao de estacao.

## 3. Converter Targets Para O Formato Esparso

O formato final evita manter campos densos para pixels sem estacao. A CLI
opera sobre um ano por vez, para permitir conversoes interrompidas e retomadas
sem reprocessar os demais anos:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_sparse
for year in $(seq 2012 2024); do
  year_dir="$dataset/year=$year"
  nowcasting-convert-sparse-targets \
    --source-year-dir "$year_dir" \
    --output-year-dir "$year_dir" \
    --output-height 128 \
    --output-width 128 \
    --chunk-size 64
done
```

O comando grava `targets_alertario_sparse.npz` e atualiza
`targets_alertario_metadata.json` para declarar o formato esparso. Valide todo
o dataset antes de remover manualmente os temporarios densos
`Y_alertario.dat` e `M_alertario.dat`. A estrutura final de cada ano e:

```text
year=AAAA/
  radar_frames.dat
  radar_timestamps.npy
  metadata.json
  targets_alertario_sparse.npz
  targets_alertario_metadata.json
```

O arquivo esparso armazena os campos `frame`, `row`, `column` e `value`; o
loader reconstrui somente as janelas requisitadas pelo batch.

## 4. Reduzir Um Dataset Existente

Quando ja houver memmaps em `256 x 256`, crie uma versao `128 x 128` sem
reprocessar os PNGs:

```bash
nowcasting-downsample \
  --source-root data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --output-root data/datasets/radar_sumare_2012_2024_15min_128_por_ano \
  --year-start 2012 \
  --year-end 2024 \
  --height 128 \
  --width 128 \
  --target-source alertario \
  --chunk-size 64
```

Depois da reducao, converta os targets densos gerados para o formato esparso.
O processo nao sobrescreve arquivos existentes; use um diretorio de saida
novo ou valide e remova resultados incompletos antes de reiniciar.

## 5. Validacao Minima

Para cada ano, verifique a presenca dos arquivos finais:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_sparse
for year in $(seq 2012 2024); do
  dir="$dataset/year=$year"
  [ -s "$dir/radar_frames.dat" ] &&
  [ -s "$dir/targets_alertario_sparse.npz" ] &&
  [ -f "$dir/metadata.json" ] &&
  [ -f "$dir/targets_alertario_metadata.json" ] &&
    echo "$year: OK" || echo "$year: AUSENTE"
done
```

Use um smoke test de treinamento, com poucos exemplos e uma epoca, antes de
iniciar um experimento multianual completo.
