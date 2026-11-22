# A0-cap: Controle de Capacidade da STConvS2S

## Objetivo

`A0-cap` controla se um eventual ganho de uma futura STConvS2S com atenção
decorre do operador de atenção ou apenas do aumento de parâmetros e
profundidade. Ele não usa atenção nem acrescenta uma nova fonte de dados.

As variantes comparáveis são:

- `A0`: `--model stconvs2s-c`;
- `A0-cap`: `--model stconvs2s-c-cap`.

## Arquitetura

O controle é inserido depois do bloco espacial e antes do gerador temporal,
quando este existe:

```text
TemporalBlock -> SpatialBlock -> ResidualCapacityBlock -> Generator -> ConvFinal
```

O bloco aplica normalização por canal em cada token espaço-temporal, projeção
pontual `C -> 2C -> C`, GELU, dropout e conexão residual:

```text
z_cap = z + gamma * Conv1x1(GELU(Conv1x1(LayerNorm(z))))
```

`gamma` é treinável e começa em zero. Com a mesma seed, os parâmetros comuns
de `A0` e `A0-cap` têm a mesma inicialização e as duas variantes produzem a
mesma saída antes da primeira atualização. A inicialização do modelo é isolada
do gerador aleatório usado pelo treino, de modo que o bloco adicional não
altera a ordem dos batches no experimento pareado.

Para `C=32`, o controle acrescenta 4.257 parâmetros: aproximadamente `4*C^2`
das projeções de uma futura atenção multi-head, mais normalização, vieses e
escala residual. No protocolo C3R atual, a contagem do núcleo passa de 425.697
para 429.954 parâmetros, aumento de aproximadamente 1,0%.

`configuration.json` e `summary.json` registram as contagens total e treinável
em `model_parameters`.

## Smoke test

Execute na workstation com o dataset V3 e o submódulo da branch atual:

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate ailab
git submodule update --init --recursive

dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

PYTHONPATH=src python -u -m nowcasting.cli.train \
  --dataset-root "$dataset" \
  --train-years 2021 --val-years 2022 --test-years 2023 \
  --target-source alertario \
  --crop-stations --crop-margin-pixels 20 \
  --station-mapping "$mapping" \
  --mapping-height-orig 654 --mapping-width-orig 656 \
  --input-stations \
  --forecast-formulation residual-persistence \
  --model stconvs2s-c-cap \
  --batch-size 2 --epochs 1 --patience 1 \
  --step 5 --stride 5 --workers 2 --persistent-workers \
  --loss masked-huber --cuda 0 --seed 1000 \
  --skip-test \
  --run-name smoke-A0-cap-seed-1000
```

## Experimento pareado

Fixe todos os argumentos, alterando somente `--model` e o identificador do
run. Execute seeds independentes `1000`, `1001` e `1002` para ambos:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

common=(
  --dataset-root "$dataset"
  --train-years 2012-2021 --val-years 2022 --test-years 2023-2024
  --target-source alertario
  --crop-stations --crop-margin-pixels 20
  --station-mapping "$mapping"
  --mapping-height-orig 654 --mapping-width-orig 656
  --input-stations --forecast-formulation residual-persistence
  --batch-size 2 --epochs 30 --patience 10
  --step 5 --stride 5 --workers 2 --persistent-workers
  --log-interval 250 --loss masked-huber
  --cuda 0 --checkpoint-every 1 --skip-test
)

for seed in 1000 1001 1002; do
  PYTHONPATH=src python -u -m nowcasting.cli.train \
    "${common[@]}" --model stconvs2s-c --seed "$seed" \
    --run-name "A0-stconvs2s-c-selection-seed-$seed"

  PYTHONPATH=src python -u -m nowcasting.cli.train \
    "${common[@]}" --model stconvs2s-c-cap --seed "$seed" \
    --run-name "A0-cap-selection-seed-$seed"
done
```

Primeiro compare `best_val_loss` em 2022. Depois de fixar a arquitetura e o
protocolo de avaliação, exporte os checkpoints para os registros canônicos de
2023--2024 e compare métricas contínuas, por horizonte, intensidade e evento.

## Resultado da seleção em 2022

| Seed | A0 | A0-cap | Melhor variante |
|---:|---:|---:|---|
| 1000 | 0,002935 | 0,002942 | A0 |
| 1001 | 0,002946 | 0,002957 | A0 |
| 1002 | 0,002959 | 0,002911 | A0-cap |
| Média | 0,002947 | 0,002937 | A0-cap |

A0-cap venceu somente uma das três seeds. A pequena melhora média, cerca de
0,34%, foi conduzida pela seed 1002 e não demonstra ganho robusto causado pelo
aumento de capacidade. O teste de 2023--2024 não foi consultado.

## Critério de interpretação

- atenção futura melhor que `A0-cap`: evidência de agregação adaptativa útil;
- atenção futura equivalente a `A0-cap`: ganho explicado por capacidade;
- `A0-cap` melhor que atenção: o viés convolucional é mais adequado;
- `A0-cap` e atenção piores que `A0`: complexidade adicional não se justifica.

Quando o desenho da atenção for congelado, a diferença entre os parâmetros
adicionais de `A0-cap` e da atenção deve permanecer abaixo de aproximadamente
5%. Se o orçamento mudar, uma nova versão do controle deve ser treinada.
