# Pipeline GOES-16 Auditavel

Esta etapa adiciona produtos ABI do GOES-16 como entrada opcional do modelo,
sem alterar `radar_frames.dat`, os targets AlertaRio ou o dataset de radar ja
publicado. O contrato estritamente causal está em `configs/goes16_abi_audited_v2.json`.

## Contrato V1

- produto: `ABI-L2-CMIPF` no bucket publico `noaa-goes16`;
- canais: `C08`, `C09`, `C13`, `C14` e `C15`;
- grade: a mesma grade radar-centrica AEQD de `128 x 128` do dataset auditado;
- timestamps: UTC;
- selecao temporal: a cena mais recente cujo fim seja menor ou igual ao
  inicio da janela de radar;
- idade maxima da cena: 20 minutos;
- ausencia: `NaN` no memmap e `0` em `goes_available.npy`;
- normalizacao V1: valor CMI dividido por 400. Essa escala fisica fixa evita
  estimar estatisticas com validacao ou teste.

O timestamp de início é preservado no manifesto para rastreabilidade, mas não
é usado para aprovar causalidade: a varredura ABI leva alguns minutos e uma
cena que termina após o instante do radar é rejeitada.

Nao use a feature `FA` do pipeline AtmoSeer como entrada: sua definicao usa
`C13(t+10 min) - C13(t)` e inclui uma observacao futura. Features derivadas
temporais devem ser calculadas causalmente em versao posterior.

## Streaming anual: C13 ou pilha multicanal

O construtor anual aceita uma cena causal por canal. Para C13 isolado, use
`configs/goes16_abi_c13_audited_v2.json`. Para a pilha ABI infravermelha
`C08`, `C09`, `C13`, `C14` e `C15`, use
`configs/goes16_abi_audited_v2.json`. A disponibilidade é conjunta: uma janela
só é publicada como disponível quando todos os canais solicitados possuem uma
cena causal e valores finitos na grade do radar. Isso impede que o treinamento
receba exemplos com modalidades incompletas.

## Smoke test streaming anual

O construtor `nowcasting-stream-goes16-memmaps` grava um tensor anual no
dataset canônico, mas permite limitar a aquisição a um intervalo de datas.
Isso possibilita validar o contrato, a reprojeção, a normalização e o loader
multimodal antes do download histórico completo:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3

nowcasting-stream-goes16-memmaps \
  --dataset-root "$dataset" \
  --goes-config configs/goes16_abi_c13_audited_v2.json \
  --radar-config configs/radar_sumare_historical_audited_v1.json \
  --year-start 2022 --year-end 2022 \
  --start-date 2022-01-01 --end-date 2022-01-01 \
  --download-workers 4 \
  --checkpoint-interval 25 \
  --retries 3

nowcasting-audit-goes16 \
  --dataset-root "$dataset" \
  --output-dir outputs/analysis/goes16/c13_streaming_2022-01-01_v1 \
  --year-start 2022 --year-end 2022
```

Nesse modo, a cobertura global do eixo anual será baixa por construção. A
auditoria deve ser interpretada pela cobertura dentro de `coverage_scope`:
para o smoke test de 1º de janeiro de 2022, `scoped_available_frames` deve ser
igual a `scoped_radar_frames`, `outside_scope_available_frames` deve ser zero e
`causality_errors` deve ser zero.

O construtor V2 reutiliza os índices de reprojeção da primeira cena, limita a
pré-busca concorrente ao valor de `--download-workers` e grava checkpoint a
cada `--checkpoint-interval` frames. Se houver falha, não remova
`year=<ano>/goes16.partial`; execute novamente o mesmo comando com `--resume`.
O contrato, o eixo temporal e o escopo devem ser idênticos, ou a retomada será
recusada. A publicação de `goes16/` só ocorre após todos os frames do escopo
estarem concluídos.

No benchmark de 1º de janeiro de 2022, quatro workers processaram 96 frames
C13 em 133,5 segundos, com 100\% de cobertura no escopo e zero violações
causais. Esse valor é uma referência local, não uma garantia de desempenho em
todos os anos ou ambientes.

## Piloto C13 por eventos

Primeiro instale as dependencias e publique os entry points no ambiente:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

O primeiro piloto não usa o downloader completo nem grava produtos no dataset
canônico. Baixar cinco canais Full Disk para todos os timestamps de 2023--2024
exigiria muitos terabytes. O piloto seleciona sequências representativas e
baixa somente uma cena C13 por frame de entrada, apagando o NetCDF logo após a
reprojeção.

Selecione até 24 eventos úmidos e controles secos do mesmo mês e hora. A
seleção usa o target observado deliberadamente e, portanto, é uma auditoria de
sinal, não uma avaliação preditiva:

```bash
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
pilot=artifacts/goes16/c13_event_pilot_v2

nowcasting-select-goes16-pilot-events \
  --dataset-root "$dataset" \
  --year-start 2023 --year-end 2024 \
  --max-wet-events 24 \
  --min-rain-mm 6.25 \
  --min-separation-hours 24 \
  --output-dir "$pilot/selection"
```

Construa os tensores C13 por streaming. Cada artefato contém cinco frames de
entrada, disponibilidade, timestamps de origem e chaves S3. O comando é
retomável por evento já publicado:

```bash
nowcasting-build-goes16-event-pilot \
  --events "$pilot/selection/events.json" \
  --output-dir "$pilot" \
  --goes-config configs/goes16_abi_c13_event_pilot_v2.json \
  --radar-config configs/radar_sumare_historical_audited_v1.json \
  --resume
```

Audite cobertura, causalidade e o contraste entre C13 e observações das
estações:

```bash
nowcasting-audit-goes16-event-pilot \
  --pilot-root "$pilot" \
  --dataset-root "$dataset" \
  --output-dir outputs/analysis/goes16/c13_event_pilot_v2
```

O arquivo `summary.json` deve reportar `causality_errors: 0`. Revise também os
painéis em `figures/`, a cobertura por evento e a correlação de Spearman entre
chuva e `-C13` (`wet_coldness_m15_spearman`).

O downloader e o construtor anual de cinco canais continuam disponíveis para
uma etapa posterior, após a decisão de expandir o processamento.

## Treinamento experimental

Depois de processar 2018--2024 e aprovar a auditoria, execute uma comparacao
justa contra C3 com a mesma intersecao de janelas validas:

```bash
nowcasting-train \
  --dataset-root "$dataset" \
  --train-years 2018-2021 \
  --val-years 2022 \
  --test-years 2023-2024 \
  --target-source alertario \
  --crop-stations --crop-margin-pixels 20 \
  --input-stations --input-goes \
  --model stconvs2s-c \
  --loss masked-huber \
  --batch-size 2 --epochs 30 --patience 10 \
  --step 5 --stride 5 --cuda 0 \
  --run-name G2-c3-abi-audited-v1
```

O loader rejeita sequencias cuja entrada de cinco janelas atravesse uma lacuna
de radar ou possua uma cena GOES indisponivel. A configuracao do experimento
grava o hash do contrato e a ordem de canais GOES.

## Inferencia operacional

Checkpoints com `--input-goes` ainda nao sao aceitos pelo CLI operacional. A
proxima extensao deve fornecer as cinco cenas ABI causais correspondentes aos
cinco frames de radar e validar o mesmo contrato salvo no checkpoint. Portanto,
nunca use um checkpoint GOES no modo operacional atual como se ele fosse um
checkpoint somente de radar.
