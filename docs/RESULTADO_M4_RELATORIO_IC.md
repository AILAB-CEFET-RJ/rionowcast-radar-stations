# Resultado M4 para o Relatório de IC

## Identificação

O experimento M4 combina a perda Huber mascarada ponderada por intensidade,
adotada em M3, com o sampler balanceado de janelas de treinamento. O objetivo
é aumentar a frequência de janelas moderadas, fortes e extremas apresentadas
ao otimizador, sem alterar o split temporal ou a arquitetura.

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
| Sampler balanceado | Sim, com limiares 1,25; 6,25; 12,5 mm/15 min |
| Batch size | 2 |
| Stride entre janelas | 5 frames |
| Máximo de épocas | 30 |
| Early stopping | Paciência de 10 épocas, monitorando a perda de validação |

O split temporal foi mantido igual aos experimentos anteriores:

| Conjunto | Anos |
|---|---|
| Treino | 2012-2021 |
| Validação | 2022 |
| Teste | 2023-2024 |

Foram utilizadas 59.897 janelas para treino, 5.677 para validação e 10.537
para teste. A avaliação considerou 1.736.907 observações de estações nos cinco
horizontes de previsão.

## Resultado

O early stopping foi acionado na época 14. A melhor época foi a 4, com perda
de validação de `0,007170`. A execução na Skat durou aproximadamente 2 h 07
min.

| Métrica no teste (2023-2024) | Valor |
|---|---:|
| RMSE (mm/15 min) | 0,53957 |
| MAE (mm/15 min) | 0,10985 |
| Bias (mm/15 min) | +0,01868 |

| Horizonte | RMSE | MAE | Bias |
|---|---:|---:|---:|
| T+1 | 0,54999 | 0,11112 | +0,01562 |
| T+2 | 0,54219 | 0,12019 | +0,02894 |
| T+3 | 0,52365 | 0,10780 | +0,01632 |
| T+4 | 0,54475 | 0,11190 | +0,02005 |
| T+5 | 0,53688 | 0,09823 | +0,01245 |

O M4 reduziu drasticamente o RMSE global do M3 (de 1,2471 para 0,5396), que
havia sido comprometido por falsos positivos intensos em observações fracas.
Porém, em relação ao M2, o RMSE global foi ligeiramente maior (0,5396 versus
0,5303) e o MAE cresceu de 0,06968 para 0,10985. O viés também mudou de
subestimação para superestimação média.

| Faixa | MAE M3 | MAE M4 | Variação M4 versus M3 |
|---|---:|---:|---:|
| Fraca/sem chuva | 0,04450 | 0,07810 | +75,5% |
| Moderada | 2,24154 | 1,96194 | -12,5% |
| Forte | 8,32098 | 7,74246 | -7,0% |
| Extrema | 16,88188 | 16,29724 | -3,5% |

O sampler balanceado melhorou todas as três faixas de maior intensidade em
relação ao M3. A melhora foi mais pronunciada em chuva moderada. Para chuva
fraca, o RMSE caiu de 1,1461 em M3 para 0,2487 em M4, mas o MAE aumentou de
0,04450 para 0,07810 e o bias foi positivo (`+0,05046`). Isso indica que M4
reduziu falsos positivos muito intensos, mas passou a prever precipitação leve
em excesso.

Os cinco horizontes tiveram RMSE próximo de 0,52--0,55 mm/15 min; não houve a
degradação acentuada em T+4 e T+5 observada em M3. Ainda assim, o aumento de
MAE em todos os horizontes mostra que a calibração do modelo continua
inadequada para uso como configuração final.

## Texto sugerido para o relatório

> No experimento M4, a função de perda Huber ponderada por intensidade foi
> combinada a um sampler balanceado de janelas de treinamento. A arquitetura
> STConvS2S-C, os dados, o split temporal e os pesos da loss foram mantidos
> iguais aos de M3. O treinamento foi interrompido por early stopping na época
> 14, tendo a época 4 apresentado a menor perda de validação. No conjunto de
> teste, o modelo obteve RMSE de 0,53957 mm/15 min, MAE de 0,10985 mm/15 min e
> viés de +0,01868 mm/15 min. Em relação a M3, o balanceamento reduziu os erros
> nas faixas moderada, forte e extrema e eliminou a degradação severa de RMSE
> nos horizontes mais longos. Entretanto, o aumento do MAE e a mudança para
> viés positivo revelam superestimação de chuva fraca. Portanto, o sampler
> balanceado com os pesos atuais melhora a sensibilidade aos eventos intensos,
> mas não produz o melhor compromisso global de calibração.

## Limitações e próximos passos

- M4 é uma única repetição e não possui métricas por estação, pois foi iniciado
  antes da implementação dessa coleta no runner.
- O sampler balanceado atual equaliza quatro classes e repete muitas janelas
  raras; ele deve ser tratado como experimento exploratório.
- O próximo teste de loss/sampler deve reduzir a agressividade da ponderação ou
  usar probabilidades de amostragem moderadas, em vez de equalização completa.
- M2 permanece como a melhor configuração global entre M1--M4 pelo menor RMSE
  e menor MAE entre as configurações Huber avaliadas, embora ainda subestime
  eventos intensos.
