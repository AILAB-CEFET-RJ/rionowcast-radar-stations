# Resultado M2 para o Relatório de IC

## Identificação

O experimento M2 avalia a substituição da MAE mascarada pela perda Huber
mascarada. A configuração foi mantida compatível com o baseline M1 para que a
comparação isole o efeito da função de perda.

## Configuração experimental

| Item | Configuração |
|---|---|
| Imagens de radar | Radar Sumaré, agregação de 15 min |
| Resolução espacial | 128 x 128 pixels |
| Targets | Observações de estações AlertaRio em formato esparso |
| Entrada do modelo | 5 imagens de radar consecutivas |
| Horizonte previsto | 5 passos de 15 min (T+1 a T+5) |
| Arquitetura | STConvS2S-C |
| Loss | Huber mascarada (`masked-huber`, delta = 0,1) |
| Sampler balanceado | Não utilizado |
| Batch size | 2 |
| Stride entre janelas | 5 frames |
| Máximo de épocas | 30 |
| Early stopping | Paciência de 10 épocas, monitorando a perda de validação |

O split temporal foi o mesmo do M1:

| Conjunto | Anos |
|---|---|
| Treino | 2012-2021 |
| Validação | 2022 |
| Teste | 2023-2024 |

Foram usadas 59.897 janelas para treino, 5.677 para validação e 10.537 para
teste. A avaliação considerou 1.736.907 observações de estações distribuídas
nos cinco horizontes de previsão.

## Resultado

O treinamento executou as 30 épocas previstas. Não houve acionamento do early
stopping: ao fim da época 30, o contador estava em 9/10. A melhor perda de
validação foi `0,002943`, obtida na época 21. O job foi executado no ambiente
CENAPAD e consumiu 5 h 31 min 34 s de walltime.

| Métrica no teste (2023-2024) | Valor |
|---|---:|
| RMSE (mm/15 min) | 0,5303 |
| MAE (mm/15 min) | 0,06968 |
| Bias (mm/15 min) | -0,04505 |

Em relação ao M1, o M2 reduziu o RMSE de 0,5396 para 0,5303 e tornou o viés
menos negativo, de -0,06099 para -0,04505 mm/15 min. Por outro lado, o MAE
aumentou de 0,06382 para 0,06968 mm/15 min. Assim, a Huber mascarada não
apresentou ganho inequívoco nas métricas globais; a interpretação final depende
da análise por horizonte e, principalmente, por intensidade de precipitação.

## Texto sugerido para o relatório

> No experimento M2, a função de perda MAE mascarada foi substituída pela
> perda Huber mascarada, mantendo-se a arquitetura STConvS2S-C, o split
> temporal e os demais hiperparâmetros do experimento baseline. O treinamento
> utilizou os anos de 2012 a 2021, a validação foi realizada com 2022 e o
> teste com 2023 e 2024. O treinamento completou as 30 épocas programadas,
> com a melhor perda de validação obtida na época 21. No conjunto de teste, o
> modelo apresentou RMSE de 0,5303 mm/15 min, MAE de 0,06968 mm/15 min e viés
> de -0,04505 mm/15 min. Em comparação ao baseline M1, houve redução do RMSE e
> da subestimação média, embora o MAE tenha aumentado. Esses resultados
> indicam que a perda Huber altera o compromisso entre as métricas globais,
> devendo ser complementada por análises estratificadas por intensidade de
> chuva e horizonte de previsão.

## Limitações e próximos passos

- M2 não utiliza ponderação por intensidade nem sampler balanceado; eventos
  fortes e extremos continuam raros no treinamento.
- A comparação entre M1 e M2 é uma repetição inicial com uma única seed; os
  modelos finalistas deverão ser repetidos com duas ou três seeds.
- M3 avaliará a Huber ponderada, e M4 acrescentará o sampler balanceado.
