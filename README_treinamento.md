# Guia De Treinamento

Este documento descreve a execucao reproduzivel dos experimentos de
nowcasting. A arquitetura STConvS2S esta em `external/stconvs2s`; o runner do
projeto e `nowcasting-train`.

## Preparacao

No ambiente Conda escolhido, instale o pacote local e inicialize o submodulo:

```bash
conda activate ailab
git submodule update --init --recursive
python -m pip install -e .
```

Verifique a interface instalada:

```bash
nowcasting-train --help
```

O dataset final esperado deve conter `radar_frames.dat`,
`radar_timestamps.npy`, `metadata.json`, `targets_alertario_sparse.npz` e
`targets_alertario_metadata.json` em cada diretorio `year=AAAA`.

## Split Temporal

O protocolo multianual atual e:

| Split | Anos | Uso |
|---|---|---|
| Treino | 2012-2021 | Otimizacao |
| Validacao | 2022 | Early stopping e escolha de hiperparametros |
| Teste | 2023-2024 | Avaliacao final |

Os splits nao podem compartilhar anos. O runner rejeita configuracoes com
vazamento temporal.

## Treino De Radar

```bash
nohup setsid nice -n 10 nowcasting-train \
  --dataset-root data/datasets/radar_sumare_2012_2024_15min_128_sparse \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --target-source alertario \
  --model stconvs2s-c \
  --batch-size 2 \
  --epochs 30 \
  --patience 10 \
  --step 5 \
  --stride 5 \
  --workers 2 \
  --pin-memory \
  --persistent-workers \
  --loss masked-huber \
  --cuda 0 \
  --run-name m2-masked-huber \
  > m2-masked-huber.log 2>&1 < /dev/null &
```

Use `--loss weighted-huber --loss-weights 1,5,10,20` para ponderar faixas de
intensidade e `--balanced-sampler` para sorteio balanceado no treino. Esses
dois mecanismos devem ser avaliados como hipoteses experimentais distintas.

Para a regiao recortada em torno das estacoes, acrescente:

```bash
--crop-stations --crop-margin-pixels 20
```

## Retomada

Ao fim de cada epoca, o runner grava
`outputs/experiments/<run-name>/iteration_1_last.pt`. O checkpoint preserva
pesos, otimizador, estado do early stopping, historico e estados aleatorios.

Para retomar, mantenha dataset, splits, modelo, loss, batch efetivo e seed:

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
  --epochs 60 \
  --patience 10 \
  --step 5 \
  --stride 5 \
  --resume outputs/experiments/m2-masked-huber/iteration_1_last.pt
```

E permitido aumentar `--epochs`, ajustar `--patience` e alterar parametros
operacionais, como `--workers` e `--log-interval`.

## DDP E Acumulacao De Gradientes

Em uma maquina com duas GPUs, use DDP com `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m nowcasting.cli.train \
  --distributed \
  --dataset-root data/datasets/radar_sumare_2012_2024_15min_128_sparse \
  --train-years 2012-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --target-source alertario \
  --model stconvs2s-c \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --loss masked-huber \
  --run-name ddp-example
```

O batch efetivo e `batch-size x numero_de_ranks x gradient-accumulation-steps`.
O sampler balanceado e aplicado globalmente e depois particionado entre ranks.

## Monitoramento E Resultados

```bash
tail -f m2-masked-huber.log
pgrep -af 'nowcasting-train|torchrun|nowcasting.cli.train'
nvidia-smi
```

Cada experimento grava em `outputs/experiments/<run-name>/`:

- `configuration.json`: parametros, splits e metadados do core;
- `iteration_1.json`: historico e metricas de teste;
- `summary.json`: consolidacao do experimento;
- `iteration_1_last.pt`: checkpoint para retomada.

Use `nowcasting-compare` para consolidar resultados de varios experimentos.
