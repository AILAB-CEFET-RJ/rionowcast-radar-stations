# RioNowCast Radar Stations

Pipeline de nowcasting de precipitacao que combina imagens do Radar do Sumare
com observacoes pluviometricas de estacoes AlertaRio e WebSirene. A arquitetura
STConvS2S e uma dependencia externa fixada como submodulo; dados, splits,
losses, metricas e experimentos pertencem a este repositorio.

## Estrutura

```text
src/nowcasting/     codigo importavel e comandos de linha de comando
data/               dados brutos, temporarios e datasets processados
outputs/            resultados de experimentos e analises geradas
notebooks/          EDA e visualizacao
external/stconvs2s/ submodulo com a arquitetura neural
```

As regras para dados, memmaps, CSVs, checkpoints e figuras estao em
[`docs/ORGANIZACAO_DADOS_E_RESULTADOS.md`](docs/ORGANIZACAO_DADOS_E_RESULTADOS.md).
O uso futuro da rede WebSirene depende da auditoria documentada em
[`docs/CONTROLE_QUALIDADE_WEBSIRENE.md`](docs/CONTROLE_QUALIDADE_WEBSIRENE.md).
O procedimento de auditoria e regeneracao rastreavel dos targets AlertaRio esta
em [`docs/CONTROLE_QUALIDADE_ALERTARIO.md`](docs/CONTROLE_QUALIDADE_ALERTARIO.md).

## Instalacao

```bash
git clone --recurse-submodules https://github.com/AILAB-CEFET-RJ/rionowcast-radar-stations.git
cd rionowcast-radar-stations
conda activate ailab
python -m pip install -r requirements.txt
python -m pip install -e .
```

Para um clone ja existente, inicialize o submodulo antes de executar o treino:

```bash
git submodule update --init --recursive
```

O ultimo comando de instalacao disponibiliza os entrypoints `nowcasting-*`.

## Dataset final

O treinamento multianual usa um dataset anual de radar em `128 x 128`, com
targets de estacoes no formato esparso:

```text
data/datasets/radar_sumare_2012_2024_15min_128_sparse/
  year=2012/
    radar_frames.dat
    radar_timestamps.npy
    metadata.json
    targets_alertario_sparse.npz
    targets_alertario_metadata.json
  ...
  year=2024/
```

Os dados grandes nao fazem parte do Git. Consulte
[`README_gerar_dataset.md`](README_gerar_dataset.md) para construir ou migrar
um dataset.

## Treinamento

O protocolo temporal base usa 2012-2021 para treino, 2022 para validacao e
2023-2024 para teste. Um exemplo com o modelo de radar e loss Huber mascarada:

```bash
nowcasting-train \
  --dataset-root data/datasets/radar_sumare_2012_2024_15min_128_sparse \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --target-source alertario \
  --model stconvs2s-c \
  --loss masked-huber \
  --batch-size 2 \
  --epochs 30 \
  --patience 10 \
  --step 5 \
  --stride 5 \
  --cuda 0 \
  --run-name example-masked-huber
```

Para executar em segundo plano, anteponha `nohup setsid` e redirecione a
saida para um arquivo de log. O runner grava `configuration.json`,
`summary.json`, metricas por horizonte e intensidade, e checkpoints retomaveis
em `outputs/experiments/<run-name>/`.

Os comandos principais sao:

| Comando | Finalidade |
|---|---|
| `nowcasting-train` | Treina STConvS2S com radar e targets de estacoes |
| `nowcasting-train-stations` | Baselines somente com historico das estacoes |
| `nowcasting-compare` | Compara `summary.json` de experimentos |
| `nowcasting-build-radar` | Gera memmaps de radar a partir de PNGs |
| `nowcasting-build-alertario-targets` | Gera targets densos AlertaRio alinhados ao radar |
| `nowcasting-convert-sparse-targets` | Converte targets densos para o formato esparso final |
| `nowcasting-downsample` | Reduz espacialmente um dataset memmap existente |

## Documentacao

- [`README_treinamento.md`](README_treinamento.md): guia de treinamento, retomada e DDP.
- [`README_gerar_dataset.md`](README_gerar_dataset.md): geracao e validacao do dataset.
- [`docs/ROADMAP_EXPERIMENTOS_MULTIANUAIS.md`](docs/ROADMAP_EXPERIMENTOS_MULTIANUAIS.md): protocolo e andamento experimental.
- [`docs/ARQUITETURA_DO_PROJETO.md`](docs/ARQUITETURA_DO_PROJETO.md): limites entre o projeto e o submodulo STConvS2S.

## Escopo dos dados

Os dados de radar e estacoes possuem restricoes de acesso e nao sao
distribuidos neste repositorio. O projeto deve ser executado somente em
ambientes autorizados que tenham esses dados disponiveis.
