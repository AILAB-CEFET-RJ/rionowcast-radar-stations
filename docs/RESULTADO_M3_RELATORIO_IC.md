# Resultado M3 para o Relatório de IC

## Identificação

O experimento M3 avalia a perda Huber mascarada ponderada por faixa de
intensidade de precipitação. O objetivo é reduzir a predominância de janelas
sem chuva ou com chuva fraca na otimização, atribuindo maior peso aos eventos
moderados, fortes e extremos.

## Configuração experimental

| Item | Configuração |
|---|---|
| Imagens de radar | Radar Sumaré, agregação de 15 min |
| Resolução espacial | 128 x 128 pixels |
| Targets | Observações de estações AlertaRio em formato esparso |
| Entrada do modelo | 5 imagens de radar consecutivas |
| Horizonte previsto | 5 passos de 15 min (T+1 a T+5) |
| Arquitetura | STConvS2S-C |
| Loss | Huber mascarada ponderada (`weighted-huber`) |
| Delta da Huber | 0,1 |
| Pesos por faixa | fraca: 1; moderada: 5; forte: 10; extrema: 20 |
| Sampler balanceado | Não utilizado |
| Batch size | 2 |
| Stride entre janelas | 5 frames |
| Máximo de épocas | 30 |
| Early stopping | Paciência de 10 épocas, monitorando a perda de validação |

O split temporal foi mantido igual aos experimentos M1 e M2:

| Conjunto | Anos |
|---|---|
| Treino | 2012-2021 |
| Validação | 2022 |
| Teste | 2023-2024 |

Foram utilizadas 59.897 janelas para treino, 5.677 para validação e 10.537
para teste. A avaliação considerou 1.736.907 observações de estações nos cinco
horizontes de previsão.

## Resultado

O early stopping foi acionado na época 18. A melhor época foi a 8, com perda
de validação de `0,005930`.

| Métrica no teste (2023-2024) | Valor |
|---|---:|
| RMSE (mm/15 min) | 1,2471 |
| MAE (mm/15 min) | 0,08002 |
| Bias (mm/15 min) | -0,02832 |

Embora o viés global tenha ficado menos negativo, o M3 apresentou RMSE e MAE
globais piores que M1 e M2. O efeito é particularmente evidente nos
horizontes longos: o RMSE foi 0,6030 em T+3, 1,0741 em T+4 e 2,3839 mm/15 min
em T+5. Em contraste, os erros nas faixas de precipitação moderada, forte e
extrema foram menores que os do M2.

| Faixa | MAE M2 | MAE M3 | Variação M3 versus M2 |
|---|---:|---:|---:|
| Fraca/sem chuva | 0,03295 | 0,04450 | +35,1% |
| Moderada | 2,33745 | 2,24154 | -4,1% |
| Forte | 8,44474 | 8,32098 | -1,5% |
| Extrema | 17,05862 | 16,88188 | -1,0% |

O RMSE da faixa fraca/sem chuva aumentou de 0,1334 no M2 para 1,1461 no M3.
Como o MAE dessa faixa é muito menor que seu RMSE, o resultado sugere falsos
positivos raros, mas de intensidade elevada. Esses erros passam a dominar o
RMSE global e se tornam mais acentuados em T+4 e T+5.

## Texto sugerido para o relatório

> No experimento M3, foi empregada a perda Huber mascarada ponderada por
> intensidade de precipitação, com pesos 1, 5, 10 e 20 para as faixas fraca,
> moderada, forte e extrema, respectivamente. A configuração de dados, o
> split temporal e a arquitetura STConvS2S-C foram mantidos iguais aos
> experimentos anteriores. O treinamento foi interrompido por early stopping
> na época 18, tendo a época 8 apresentado a menor perda de validação. No
> conjunto de teste, o modelo obteve RMSE de 1,2471 mm/15 min, MAE de 0,08002
> mm/15 min e viés de -0,02832 mm/15 min. Apesar de reduzir os erros nas
> faixas moderada, forte e extrema em relação ao M2, a ponderação elevou
> fortemente o erro quadrático em observações fracas ou sem chuva,
> principalmente nos horizontes de previsão mais longos. Os resultados
> indicam que os pesos adotados melhoram parcialmente a sensibilidade aos
> eventos intensos, mas prejudicam a calibração global do modelo.

## Registro de execução

O job no CENAPAD registrou `Exit_status = 127` após 3 h 19 min 33 s. Esse
status foi causado por `--checkpoint-every` em uma linha isolada do script
PBS, que o shell tentou executar como comando após o término do Python. O log
registra explicitamente o early stopping e as métricas de teste antes desse
erro; portanto, os resultados deste experimento são válidos. O script PBS foi
corrigido para os experimentos seguintes.

## Limitações e próximos passos

- M3 é uma única repetição e não utiliza sampler balanceado.
- A melhoria nas faixas intensas foi pequena diante da degradação global.
- M4 combina `weighted-huber` e sampler balanceado para testar se a maior
  frequência de janelas intensas altera esse compromisso.
- Caso M4 não resolva os falsos positivos, os próximos testes devem usar
  pesos de loss mais moderados do que `1,5,10,20`.
