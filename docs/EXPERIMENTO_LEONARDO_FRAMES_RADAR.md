# Experimento de Leonardo: previsão de frames RGB do radar

## Objetivo

Demonstrar a STConvS2S em uma tarefa visual e espaço-temporal simples: usar os
cinco compósitos RGB anteriores do radar do Sumaré, com cadência de 15 minutos,
para prever os cinco compósitos RGB seguintes. Cada exemplo, portanto, contém
75 minutos de histórico e 75 minutos de previsão.

Este não é um experimento de precipitação em superfície. O acervo disponível
é o bitmap RGB do produto visual do radar, sem campo numérico calibrado de
refletividade. No relatório e nos slides, descreva a saída como **previsão de
frames RGB do radar**; não como refletividade quantitativa, chuva em mm/h ou
alerta pluviométrico.

## Protocolo mínimo

| Papel | Anos |
|---|---|
| Treino | 2018--2021 |
| Validação e escolha de época | 2022 |
| Teste para apresentação | 2023 |
| Holdout futuro, não usar na seleção | 2024 |

As janelas que cruzam lacunas de 15 minutos do acervo são descartadas. O
split é cronológico: 2023 não participa do treino, da escolha de época ou do
ajuste de hiperparâmetros.

São comparados dois métodos no mesmo teste:

1. **Persistência visual:** repete o último frame de entrada nos cinco
   horizontes futuros.
2. **STConvS2S-C:** aprende diretamente a sequência RGB futura.

As métricas são MAE e RMSE normalizados em `[0, 1]`, para todos os canais RGB,
e também somente nos pixels cujo target futuro possui algum canal não preto
(`active_echo`). Esta última evita que o fundo preto domine artificialmente a
métrica. Nenhuma dessas métricas tem unidade de precipitação.

## Execução

Na máquina com GPU e o dataset de radar instalado:

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate ailab
git switch feature/leonardo-radar-frame-nowcast
git pull --ff-only origin feature/leonardo-radar-frame-nowcast
git submodule update --init --recursive

dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3

PYTHONPATH=src python -u -m nowcasting.cli.train_radar_frames \
  --dataset-root "$dataset" \
  --train-years 2018-2021 \
  --val-years 2022 \
  --test-years 2023 \
  --model stconvs2s-c \
  --t-in 5 --t-out 5 --stride 5 \
  --batch-size 2 \
  --workers 2 \
  --epochs 30 \
  --patience 8 \
  --loss huber \
  --seed 1000 \
  --cuda 0 \
  --run-name Leonardo-radar-rgb-stconvs2s-seed-1000
```

## Artefatos para os slides

O diretório `outputs/experiments/Leonardo-radar-rgb-stconvs2s-seed-1000/`
conterá:

- `summary.json`: métricas da STConvS2S e da persistência no teste;
- `best.pt`: checkpoint escolhido exclusivamente pela validação de 2022;
- `qualitative_test_example.png`: comparação visual, em cinco horizontes,
  entre último frame de entrada, persistência, STConvS2S e frame observado;
- `configuration.json`: parâmetros e splits usados.

Para uma apresentação curta, reporte a tabela de MAE/RMSE global e de eco
ativo, a métrica por horizonte da `summary.json` e a figura qualitativa.
