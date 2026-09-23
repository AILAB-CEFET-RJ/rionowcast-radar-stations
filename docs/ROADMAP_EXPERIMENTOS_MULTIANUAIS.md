# Roadmap de Experimentos Multianuais

## Objetivo

Treinar e avaliar o nowcasting com anos inteiros, separados temporalmente,
para ampliar a diversidade de eventos de chuva e evitar vazamento entre
treino, validação e teste.

## Estado Atual

- [x] Integração de targets AlertaRio em memmap.
- [x] Dataset 2024 reduzido para `128 x 128`.
- [x] Smoke test em GPU com entrada `[batch, 3, 5, 128, 128]` e saída
  `[batch, 1, 5, 128, 128]`.
- [x] Losses mascaradas e ponderadas, sampler balanceado e métricas por faixa
  implementados no pacote do projeto.
- [x] Runner independente do projeto criado em
  `scripts/train_nowcasting.py`.
- [x] Dataset de radar `128 x 128` disponível e validado para 2012-2024.
- [x] Targets AlertaRio esparsos `128 x 128` disponíveis e validados para
  2012-2024.
- [x] Loader compatível com `targets_alertario_sparse.npz` e com o formato
  denso legado.
- [x] Primeiro experimento com split temporal multianual concluído (M1).

## Protocolo Inicial

| Split | Anos | Uso |
|---|---|---|
| Treino | 2012-2021 | Otimização dos parâmetros |
| Validação | 2022 | Early stopping e escolha de hiperparâmetros |
| Teste | 2023-2024 | Avaliação final |

Os anos devem ser revisados após uma auditoria de cobertura de radar e de
estações. O runner rejeita anos repetidos entre splits.

## Etapas

### Dataset

- [x] Gerar `radar_frames.dat`, `radar_timestamps.npy` e `metadata.json` em
  `128 x 128` para 2012-2024, diretamente dos PNGs de radar.
- [x] Converter os targets AlertaRio densos de cada ano para
  `targets_alertario_sparse.npz`, com campos `frame`, `row`, `column` e
  `value`.
- [x] Remover os temporários densos `256 x 256` após a conversão anual.
- [x] Validar a estrutura esparsa em 2012: 1.093.997 observações, índices
  temporais e espaciais dentro dos limites da grade.
- [x] Auditar a distribuição por faixa no conjunto de treino.
- [ ] Registrar a distribuição por faixa também em validação e teste.

O formato esparso elimina os memmaps densos `Y_alertario.dat` e
`M_alertario.dat` do dataset final. O loader reconstrói somente a janela alvo
do batch, preservando a interface das losses mascaradas e reduzindo o uso de
disco sem alterar a arquitetura STConvS2S.

### Código

- [x] Remover o split interno 60/20/20 do caminho novo de treinamento.
- [x] Implementar `--train-years`, `--val-years` e `--test-years`.
- [x] Restringir o sampler balanceado ao treino.
- [x] Registrar configuração, commit do core, checkpoints, histórico e métricas.
- [x] Smoke test do loader esparso para 2012.
- [x] Smoke test ponta a ponta com os três splits temporais, incluindo treino,
  validação, checkpoint, recarga e teste.
- [x] Adicionar `--gradient-accumulation-steps` para simular batch efetivo
  maior em GPUs com VRAM limitada.
- [x] Implementar retomada V1 entre épocas, com checkpoint atômico do último
  estado, otimizador, early stopping, histórico e estados aleatórios.
- [x] Implementar crop dinâmico da região das estações AlertaRio, sem duplicar
  os memmaps, com margem configurável e metadados no experimento.
- [ ] Registrar no log a distribuição efetivamente sorteada pelo sampler em
  cada época.

### Região Das Estações E Baselines

O crop é derivado de `data/mapeamento_pixel_estacao_alertario.csv`, convertido
para a resolução do dataset e expandido por uma margem em pixels. O runner usa
`--crop-stations --crop-margin-pixels 20` para ativá-lo; a configuração salva
os limites efetivos, dimensão e quantidade de estações. Os targets esparsos
são filtrados e reindexados em memória, mantendo o dataset original intacto.

- [x] Crop dinâmico de radar, targets e máscara para a região das estações.
- [x] Garantir compatibilidade do crop com o sampler balanceado e com
  checkpoints retomáveis.
- [ ] Executar M2 e M4 equivalentes usando o crop, com o mesmo split temporal.
- [x] Implementar persistência por estação como baseline sem radar.
- [x] Implementar modelo temporal multivariado somente com estações.
- [ ] Comparar os três modelos nos mesmos pares timestamp/estação/horizonte,
  incluindo métricas por intensidade e por estação.

O STConvS2S atual recebe somente os campos de radar; as estações fornecem os
targets e a máscara da loss. Portanto, a primeira comparação será
**radar supervisionado por estações** versus **somente histórico das
estações**. Um modelo de fusão que receba radar e histórico de estações como
entrada é uma extensão posterior e não deve ser confundido com o STConvS2S
atual.

`scripts/train_station_baseline.py` oferece os modelos `persistence` e `mlp`.
O MLP recebe, para cada uma das 33 estações, os cinco valores passados em
`log1p(mm/15min)` e suas máscaras de disponibilidade; ele prevê cinco passos
futuros nas mesmas estações. Seus resultados usam o mesmo esquema de métricas
globais, por horizonte e por intensidade dos experimentos de radar.

### Matriz Experimental

| ID | Loss | Sampler | Repetições iniciais | Estado |
|---|---|---|---:|
| M1 | `masked-mae` | não | 1 | Concluído |
| M2 | `masked-huber` | não | 1 | Interrompido por congelamento da Skat; reiniciar na Arietis |
| M3 | `weighted-huber` | não | 1 | Pendente |
| M4 | `weighted-huber` | balanceado | 1 | Pendente |

As configurações finalistas devem ser repetidas com 2 ou 3 seeds. Métricas
globais devem ser complementadas por resultados por horizonte e intensidade.

### Auditoria do Sampler

No split de treino (2012-2021, `stride=5`), há 59.897 janelas:

| Faixa | Janelas | Proporção |
|---|---:|---:|
| Fraca/sem chuva | 54.075 | 90,28% |
| Moderada | 4.375 | 7,30% |
| Forte | 934 | 1,56% |
| Extrema | 513 | 0,86% |

Uma janela extrema é uma sequência distinta `(ano, índice inicial)` cujo
máximo observado nos cinco horizontes futuros é pelo menos `12,5 mm/15 min`.
O sampler balanceado atual equaliza as quatro classes em expectativa e, por
isso, repetiria cada uma das 513 janelas extremas cerca de 29 vezes por época.
M4 deve ser interpretado como experimento exploratório; após sua avaliação,
testar um sampler de probabilidades moderadas para reduzir essa repetição.

## Registro

| Data | Item | Estado | Evidência |
|---|---|---|---|
| 2026-09-10 | Dataset AlertaRio 2024 em `128 x 128` | Concluído | 25.505 frames e 3,2 GB. |
| 2026-09-10 | Smoke test 2024 em GPU | Concluído | Treinamento, validação e teste concluídos. |
| 2026-09-10 | `weighted-huber` + sampler em 2024 | Concluído | Duas repetições exploratórias. |
| 2026-09-10 | Migração da lógica de domínio | Em andamento | Código movido para `src/nowcasting`. |
| 2026-09-13 | Radar 128x128 para 2012-2024 | Concluído | Frames anuais gerados diretamente dos PNGs. |
| 2026-09-14 | Targets AlertaRio esparsos 128x128 | Concluído | `targets_alertario_sparse.npz` gerado para 2012-2024; dataset final ocupa 18 GiB. |
| 2026-09-14 | Smoke do loader esparso em 2012 | Concluído | 6.837 janelas; sampler: 6.314/398/90/35 por faixa. |
| 2026-09-19 | M1 multianual na Skat | Concluído | Early stopping na época 28; melhor época 18; teste: RMSE 0,5396, MAE 0,06382, Bias -0,06099. |
| 2026-09-19 | M2 multianual na Skat | Interrompido | Congelamento da máquina durante o experimento. |
| 2026-09-20 | Dataset esparso na Arietis | Concluído | Dataset 128x128 reconstruído e validado para 2012-2024, com 18 GiB. |
| 2026-09-20 | Auditoria do treino | Concluído | 513 janelas extremas, equivalentes a 0,8565% das 59.897 janelas. |
| 2026-09-21 | Retomada V1 | Concluído | `iteration_1_last.pt` permite retomar na próxima época após interrupção; validada por testes unitários. |
| 2026-09-22 | Crop da região das estações | Concluído | Crop dinâmico por CSV de mapeamento, com margem configurável e testes de preservação das observações. |
| 2026-09-22 | Baselines somente com estações | Concluído | Dataset temporal para 33 estações, persistência e MLP multivariado implementados e testados. |
