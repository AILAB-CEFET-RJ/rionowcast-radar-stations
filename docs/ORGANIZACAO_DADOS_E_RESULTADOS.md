# Organizacao De Dados E Resultados

Este projeto separa codigo, dados e resultados gerados para manter o pipeline
reproduzivel e impedir que artefatos locais sejam confundidos com fonte.

| Tipo | Local | Exemplos | Controle de versao |
|---|---|---|---|
| Codigo | `src/nowcasting/` | datasets, losses, treino e CLIs | Versionado |
| Dados brutos | `data/raw/` | PNGs, ZIPs e Parquets recebidos | Ignorado |
| Dados temporarios | `data/interim/` | conversoes e etapas intermediarias | Ignorado |
| Datasets processados | `data/processed/` ou `data/datasets/` | memmaps, `npy` e `npz` consumidos no treino | Ignorado |
| Experimentos | `outputs/experiments/<run-name>/` | configuracao, metricas, checkpoints e logs | Ignorado |
| Analises | `outputs/analysis/<fonte>/` | CSVs, tabelas e graficos derivados | Ignorado |
| Figuras para documentos | `figures/` | figuras selecionadas para relatorios | Versionado quando selecionadas |

## Regras

- Um `npy` ou `npz` que faz parte de um dataset de treino pertence a `data/`.
- Um CSV, tabela ou grafico produzido por analise pertence a `outputs/analysis/`.
- Checkpoints e resultados de execucoes pertencem a `outputs/experiments/`.
- `src/` contem apenas codigo fonte; nao deve conter resultados gerados.
- `data/datasets/` e `outputs/` sao ignorados pelo Git. A rastreabilidade de
  resultados deve ser mantida por configuracoes, seeds, logs e resumos
  estruturados arquivados fora do repositorio, nao pelo versionamento de
  artefatos gerados.

## Historico WebSirene

Os CSVs de maximos anuais e os graficos e tabelas de elevacao historicos foram
migrados para `outputs/analysis/websirene/`. O mapa de distribuicao das
estacoes AlertaRio e salvo em `outputs/analysis/geospatial/`. Novas execucoes
devem escrever nesses locais ou em um subdiretorio equivalente da fonte
analisada.

A auditoria reproduzivel WebSirene grava Parquets em
`data/processed/websirene_qc/<versao>/` e relatórios em
`outputs/analysis/websirene/qc/<versao>/`. Consulte
[`CONTROLE_QUALIDADE_WEBSIRENE.md`](CONTROLE_QUALIDADE_WEBSIRENE.md) antes de
usar esta fonte como supervisao.
