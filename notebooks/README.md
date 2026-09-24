# Notebooks

Esta pasta contem notebooks ativos de exploracao e validacao. Cada notebook
deve usar caminhos relativos ao repositorio, declarar os dados de entrada e
evitar gravar resultados grandes no proprio arquivo.

## Organizacao

- `01_eda/`: analise exploratoria de fontes observacionais. Inclui AlertaRio
  e WebSirene, que permanecem relevantes para experimentos futuros.
- `02_geospatial/`: mapas, cobertura espacial e verificacoes geograficas.
- `03_modeling/`: materiais didaticos sobre o dataset, os tensores e os
  modelos de nowcasting.
- `archive/`: notebooks historicos mantidos apenas para rastreabilidade. Eles
  podem referenciar datasets, dependencias ou caminhos que nao fazem parte do
  pipeline atual.

## Comparacao de experimentos

A comparacao reprodutivel dos resultados de treinamento e feita por
`nowcasting-compare` (modulo `nowcasting.cli.compare_experiments`). Ela deve permanecer como script para que a
mesma analise possa ser executada em ambientes sem Jupyter e automatizada em
novos resultados. Um notebook de visualizacao pode ser criado futuramente
apenas quando houver uma necessidade especifica de narrativa ou figuras.

## Convencoes

- Nao instale dependencias em celulas de notebook.
- Nao use caminhos absolutos de uma maquina pessoal.
- Nao use o notebook para gerar datasets de treinamento; mantenha essa etapa
  em scripts versionados.
- Limpe saidas pesadas antes de versionar um notebook.

## Mapas Das Redes

- `02_geospatial/01_mapa_estacoes_alertario.ipynb`: distribuicao e pixels da
  rede AlertaRio.
- `02_geospatial/02_mapa_redes_pluviometricas.ipynb`: camadas Folium para
  AlertaRio e WebSirene, com nomes selecionaveis e verificacao de pixels
  compartilhados na grade `128 x 128`.

## Modelagem

- `03_modeling/01_tensores_entrada_saida.ipynb`: introducao aos formatos dos
  tensores de radar, targets e mascaras; inclui a modalidade radar mais
  estacoes e requisitos para futuras fontes, como GOES.
- `03_modeling/02_loss_metricas_mascaras.ipynb`: losses mascaradas e
  ponderadas, escala `log1p` e metricas em `mm/15 min`.
- `03_modeling/03_splits_temporais_e_vazamento.ipynb`: janelas temporais,
  split por anos e regras para evitar vazamento.
- `03_modeling/04_balanceamento_eventos_extremos.ipynb`: distribuicao de
  intensidades, sampler balanceado e repeticao esperada de eventos raros.
