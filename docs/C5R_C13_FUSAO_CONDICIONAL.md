# C5R: Fusão Condicional e Tardia com GOES-16 C13

## Motivação

Na ablação C4R, a concatenação precoce de C13 ao RGB do radar não produziu
ganho reproduzível entre três seeds. Isso não invalida a hipótese física do
piloto condicional: C13 pode ser útil principalmente quando o eco RGB local é
fraco ou ambíguo. C5R testa essa hipótese sem assumir que uma única rede deve
aprender, desde a primeira camada, a compatibilidade entre as duas fontes.

## Formulação

A previsão mantém a formulação residual causal:

\[
\widehat{y} = y_{\mathrm{persistência}} + r_{\mathrm{RGB}} +
g(\mathrm{RGB}_{T-15})\,r_{\mathrm{C13}}.
\]

Há dois ramos STConvS2S independentes. O ramo RGB recebe somente os três
canais do radar; o ramo C13 recebe somente C13. A porta espacial
$g\in(0,1)$ é aprendida, mas é monotonicamente menor em pixels com eco RGB
não preto no último frame de entrada. Assim, C13 pode corrigir a previsão
sobretudo onde a evidência visual local do radar é ausente.

O experimento possui dois braços com a mesma disponibilidade causal, o mesmo
número de parâmetros e os mesmos dados:

- **C5R-C13-capacity-control:** usa os dois ramos, mas zera C13 antes do ramo
  satelital. Ele controla a capacidade adicional da arquitetura;
- **C5R-C13-conditional:** usa C13 normalmente.

O argumento `--conditional-goes-control` só é permitido na formulação
`conditional-late-fusion`. Ele não elimina o ramo C13, apenas remove seus
valores. Portanto, qualquer diferença entre os braços não pode ser atribuída
à disponibilidade de C13 nem ao número de parâmetros.

## Protocolo

Usar treino em 2018--2021, validação em 2022 e teste final em 2023--2024.
Todos os splits exigem cinco cenas C13 causais na entrada. Checkpoint, escolha
de seed, limiares de alerta e qualquer peso de ensemble são definidos somente
em 2022.

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
mapping=configs/mapeamento_pixel_estacao_alertario_historical_v1.csv

common=(
  --dataset-root "$dataset"
  --train-years 2018-2021 --val-years 2022 --test-years 2023-2024
  --target-source alertario
  --crop-stations --crop-margin-pixels 20
  --station-mapping "$mapping" --mapping-height-orig 654 --mapping-width-orig 656
  --input-stations --input-goes
  --forecast-formulation conditional-late-fusion
  --model stconvs2s-c
  --batch-size 2 --epochs 30 --patience 10
  --step 5 --stride 5
  --workers 2 --persistent-workers
  --log-interval 250 --loss masked-huber
  --cuda 0 --checkpoint-every 1
  --seed 1000
)

PYTHONPATH=src python -u -m nowcasting.cli.train \
  "${common[@]}" \
  --conditional-goes-control \
  --run-name C5R-C13-capacity-control-seed-1000

PYTHONPATH=src python -u -m nowcasting.cli.train \
  "${common[@]}" \
  --run-name C5R-C13-conditional-seed-1000
```

Repetir em `1001` e `1002` somente se a primeira seed produzir checkpoints e
registros exportáveis nos dois braços.

## Avaliação Estratificada

Ao exportar checkpoints, os registros Parquet passam a incluir
`radar_echo_fraction`: fração de pixels RGB não pretos na vizinhança de raio
dois de cada estação, no último frame de entrada. Esse contexto é calculado
somente a partir da entrada causal e se repete nos cinco horizontes do mesmo
exemplo.

Além da avaliação global, comparar os dois braços nos estratos:

- `low_echo`: `radar_echo_fraction <= 0,05`;
- `echo_present`: `radar_echo_fraction > 0,05`.

O avaliador `nowcasting.cli.evaluate_forecast_echo_strata` aplica as mesmas
métricas contínuas, categóricas e bootstrap pareado por dia do avaliador
canônico. A alegação de utilidade de C13 requer, no mínimo, ganho consistente
no estrato `low_echo`, com intervalo de confiança favorável e sem piora
operacionalmente inaceitável nos alertas moderados e fortes.

Registros antigos não têm a coluna de eco e devem ser reexportados antes da
estratificação. A métrica global continua obrigatória: estratificação é uma
análise de hipótese, não uma substituição seletiva do teste final.

## Resultado e Decisão

Foram executadas três seeds independentes. Na seed 1000, C13 reduziu MAE de
0,068983 para 0,065535 e RMSE de 0,560617 para 0,536640. O resultado não se
reproduziu: nas seeds 1001 e 1002, C13 elevou RMSE de 0,548119 para 0,556948
e de 0,543638 para 0,552025, respectivamente. As diferenças de MAE também
foram pequenas e inconsistentes.

A estratificação não confirmou de forma útil a hipótese condicional. O
estrato de eco fraco concentrou a maior parte dos pares e foi dominado por
chuva fraca ou seca; nele, os ganhos de MAE foram muito pequenos e não houve
padrão estável nas faixas moderada, forte e extrema. No estrato com eco
presente, a seed 1000 melhorou substancialmente, mas as outras duas pioraram.

Os alertas calibrados reforçaram essa conclusão. Para 6,25~mm/15~min no proxy
municipal, a fração de eventos detectados variou entre $-1,5$, $+0,5$ e
0,0 pontos percentuais entre as seeds, sem mudança na antecedência mediana.
Para 12,5~mm/15~min, a seed 1000 reduziu falsos alertas ao custo de perder
eventos, enquanto as demais seeds não apresentaram ganho consistente.

Portanto, C5R é uma segunda ablação negativa: a fusão tardia condicionada não
justifica o uso de C13 no modelo atual. Novas variações arquiteturais de C13
não devem ser priorizadas antes de ampliar e auditar a supervisão espacial.
