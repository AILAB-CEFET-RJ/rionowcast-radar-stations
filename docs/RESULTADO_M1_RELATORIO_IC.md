# Resultado M1 para o Relatório de IC

## Identificação

O experimento M1 estabelece o primeiro baseline multianual do projeto. Ele
treina a STConvS2S-C com targets de precipitação do AlertaRio e avalia em anos
posteriores aos usados na otimização, evitando vazamento temporal entre os
conjuntos.

## Configuração experimental

| Item | Configuração |
|---|---|
| Imagens de radar | Radar Sumaré, agregação de 15 min |
| Resolução espacial | 128 x 128 pixels |
| Targets | Observações de estações AlertaRio em formato esparso |
| Entrada do modelo | 5 imagens de radar consecutivas |
| Horizonte previsto | 5 passos de 15 min (T+1 a T+5) |
| Arquitetura | STConvS2S-C |
| Loss | MAE mascarada (`masked-mae`) |
| Sampler balanceado | Não utilizado |
| Batch size | 2 |
| Stride entre janelas | 5 frames |
| Máximo de épocas | 30 |
| Early stopping | Paciência de 10 épocas, monitorando MAE mascarada na validação |

O split temporal adotado foi:

| Conjunto | Anos |
|---|---|
| Treino | 2012-2021 |
| Validação | 2022 |
| Teste | 2023-2024 |

O conjunto de treino continha 59.897 janelas, o de validação 5.677 janelas e
o de teste 10.537 janelas. As métricas de teste foram calculadas apenas nos
pixels e instantes que possuem observação de estação, totalizando 1.736.907
observações nos cinco horizontes.

## Resultado

O treinamento foi encerrado por early stopping na época 28. A melhor época,
selecionada pela perda de validação, foi a época 18.

| Métrica no teste (2023-2024) | Valor |
|---|---:|
| RMSE (mm/15 min) | 0,5396 |
| MAE (mm/15 min) | 0,06382 |
| Bias (mm/15 min) | -0,06099 |

O viés negativo indica subestimação média da precipitação pelo modelo. Como o
conjunto é fortemente desbalanceado, com predominância de janelas sem chuva ou
de chuva fraca, essas métricas globais devem ser interpretadas junto com a
avaliação por intensidade e por horizonte de previsão.

## Texto sugerido para o relatório

> Foi conduzido um experimento baseline multianual (M1) com a arquitetura
> STConvS2S-C, utilizando imagens do Radar do Sumaré em resolução de 128 x
> 128 pixels e observações esparsas das estações AlertaRio. Os dados foram
> separados temporalmente em treino (2012-2021), validação (2022) e teste
> (2023-2024). O modelo recebeu cinco imagens consecutivas de radar e previu
> os cinco instantes seguintes, com resolução temporal de 15 minutos. Foi
> empregada a função de perda MAE mascarada, calculada somente nos pixels que
> possuem observações de precipitação. O treinamento atingiu early stopping na
> época 28, sendo a época 18 selecionada como a melhor pelo desempenho em
> validação. No conjunto de teste, o modelo obteve RMSE de 0,5396 mm/15 min,
> MAE de 0,06382 mm/15 min e viés de -0,06099 mm/15 min. O viés negativo
> evidencia uma tendência de subestimação, que será investigada nos
> experimentos seguintes com Huber mascarada, loss ponderada e amostragem
> balanceada.

## Limitações e próximos passos

- M1 é o baseline de comparação; ele não usa loss ponderada nem sampler
  balanceado.
- O resultado consolidado ainda deve incluir métricas separadas por horizonte
  e por faixa de intensidade de precipitação.
- Os experimentos M2, M3 e M4 avaliarão, respectivamente, Huber mascarada,
  Huber ponderada e Huber ponderada com sampler balanceado.
