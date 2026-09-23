# Arquivo Historico de Notebooks

Os notebooks desta pasta foram preservados como evidencia de exploracoes
anteriores. Eles nao fazem parte do pipeline reproduzivel atual e nao devem
ser usados para gerar datasets, treinar os modelos multianuais ou produzir
resultados oficiais sem uma refatoracao explicita.

## Conteudo

- `RGB_Rainfall_Calibration.ipynb`: calibracao exploratoria RGB/rainfall com
  Random Forest e dados INMET.
- `websirenes_teste.ipynb`: variante quase duplicada de
  `RGB_Rainfall_Calibration.ipynb`.
- `websirenes_train_model.ipynb`: treinamento exploratorio WebSirene com
  dependencias e saidas historicas.
- `load_radar_dataset.ipynb`: inspecao do formato NPZ legado, substituido pelos
  memmaps anuais atuais.
- `overlay_radar.ipynb`: overlay geoespacial com caminhos absolutos do Atmoseer
  e mapeamento WebSirene antigo.
- `analise_exploratoria_alertario_legacy.ipynb`: versao anterior da EDA do
  AlertaRio, que tambem convertia arquivos e continha caminhos absolutos.
- `eda_websirene_legacy.ipynb`: exploracao antiga de calibracao WebSirene com
  caminhos relativos a artefatos que nao pertencem ao pipeline atual.
- `01_gera_websirenes_mapeamento_legacy.ipynb` e
  `02_gera_mascara_precipitacao_estacao_radar_legacy.ipynb`: notebooks de
  geracao antigos, preservados apos serem removidos de `src/datasets/`.

O material potencialmente reutilizavel deve ser extraido para `src/nowcasting`,
com caminhos relativos ao projeto, parametros explicitos e testes.
Os notebooks ativos estao organizados por tema em `notebooks/01_eda/` e
`notebooks/02_geospatial/`.
