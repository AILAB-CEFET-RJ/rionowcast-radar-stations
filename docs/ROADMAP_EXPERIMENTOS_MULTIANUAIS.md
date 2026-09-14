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
- [ ] Primeiro experimento com split temporal multianual.

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
- [ ] Registrar, para todos os splits, contagem de observações e distribuição
  por faixa de intensidade.

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
- [ ] Executar o smoke test ponta a ponta no ambiente `ailab` da
  `workstation02`, usando os três splits temporais.

### Matriz Experimental

| ID | Loss | Sampler | Repetições iniciais |
|---|---|---|---:|
| M1 | `masked-mae` | não | 1 |
| M2 | `masked-huber` | não | 1 |
| M3 | `weighted-huber` | não | 1 |
| M4 | `weighted-huber` | sim | 1 |

As configurações finalistas devem ser repetidas com 2 ou 3 seeds. Métricas
globais devem ser complementadas por resultados por horizonte e intensidade.

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
