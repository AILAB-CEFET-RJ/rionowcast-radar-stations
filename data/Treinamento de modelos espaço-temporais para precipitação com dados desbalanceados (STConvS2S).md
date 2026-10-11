Eu perguntando: Vou fazer uma busca na literatura em vários eixos: funções de custo, treinamento e hiperparâmetros, e avaliação.Fiz a busca em quatro eixos: dados, função de custo, hiperparâmetros e avaliação. A parte de batch size e learning rate é a mais fraca da literatura (detalhes na seção 3).

## 1. Dados e pré-processamento

- **Reamostragem de sequências.** O DGMR (DeepMind) rebalanceou o dataset para incluir mais observações de chuva forte, e limitou a taxa máxima a 128 mm/h. Segundo o artigo, o rebalanceamento permite que os modelos aprendam previsões úteis. A survey de Nowcasting com DL também lista oversampling de casos chuvosos, amostragem hierárquica e amostragem por bins da distribuição ou ajuste da proporção de pixels chuvosos.
- **Subamostragem de "sem chuva".** Um trabalho recente com radar e satélite manteve apenas 20% das imagens sem chuva, removeu outliers acima de 200 mm/h e normalizou com transformação logarítmica.
- **Transformação e máscara.** Para radar, uma prática comum é converter refletividade em taxa de chuva (Marshall-Palmer), aplicar escala logarítmica com piso fixo, usar crops aleatórios e flips, e mascarar regiões fora da cobertura do radar no cálculo da loss. A máscara é relevante se você usa pluviômetros ou radar com cobertura parcial.

## 2. Função de custo (o ponto mais bem estudado)

- **MSE/MAE puros são problemáticos.** Perdas pontuais tendem a gerar previsões excessivamente suavizadas e a representar mal a chuva forte.
- **Loss ponderada por intensidade (B-MSE/B-MAE).** É a referência clássica, do benchmark HKO-7 (Shi et al., NeurIPS 2017). Os autores observaram que as variantes balanceadas são mais consistentes com o desempenho em múltiplos limiares e essenciais para bom desempenho em taxas altas. O DGMR também usa ponderação, com clipping do campo de chuva em [0, 24].
- **Hybrid Weighting.** Combina reponderação e não-ponderação, melhorando intensidades altas sem degradar as baixas, ao contrário dos métodos de reponderação puros. Isso é útil porque pesos agressivos costumam prejudicar a chuva fraca.
- **Nowcasting Loss (Ko et al., 2022).** É uma loss baseada em CSI diferenciável, com termos para chuva em geral e chuva forte. A chuva forte entra nos dois termos, o que mitiga o desbalanceamento. Os autores reportam ganho de até 43,6% no CSI para chuva forte, além de um esquema de pré-treino em refletividade.
- **PP-Loss (Xu et al., 2024).** Atribui a cada amostra um peso inversamente proporcional à probabilidade de ocorrência do seu valor, via plotting position. Em ConvLSTM com ERA5-Land, o desempenho foi comparável a MSE, Focal-R, Gumbel Loss e LDS. O ganho aparece, portanto, mais como alternativa baseada em regressão desbalanceada do que como superioridade clara.
- **Termos contra o borramento.** O GDL (gradient difference loss) foi usado junto com a loss ponderada no FDNet, com melhora média de 0,007 em CSI e HSS no SRAD2020. Já a **FACL** (Fourier Amplitude and Correlation Loss, NeurIPS 2024) regulariza a amplitude no domínio de Fourier e complementa com a correlação de fase, substituindo MSE e MSE ponderado. A FACL é parameter-free, e vale testá-la como complemento.
- **Abordagem generativa.** DGMR e NowcastNet atacam o borramento com GAN/adversarial. O discriminador espacial garante consistência espacial e desencoraja previsões borradas. Isso custa mais complexidade de treino.

## 3. Hiperparâmetros: o que a literatura reporta

Poucos trabalhos fazem ablação sistemática de batch size e learning rate. Em geral só se relata a configuração final, então os valores abaixo são pontos de partida, não evidência causal:

- **Shi et al. 2017:** Adam com lr 1e-4; batch 4 para modelos recorrentes e 8 para CNNs.
- **Ko et al. 2022:** Adam com lr 2e-5; 50 mil passos com batch 20 no pré-treino e 35 mil com batch 24 no fine-tuning.
- **Único estudo com busca explícita que encontrei** (U-Net com múltiplos fluxos ópticos, Weather and Forecasting): busca em lr entre 1e-5 e 1e-4 e batch entre 8 e 64; o melhor foi lr 1e-4, batch 8, StepLR com passo de 15 épocas e gamma 0,5. Batch pequeno funcionando melhor sugere que o ruído do gradiente ajuda a não ignorar os eventos raros, mas é um resultado isolado.
- **Modelos mais recentes** usam AdamW com warmup e cosine: CasCast, com AdamW e cosine schedule, batch global 32; outro trabalho, com AdamW, lr base 1e-3, warm-up cosine e batch 16, treinado com FACL.
- **Early stopping e overfitting:** no Weather4Cast 2025, o ConvLSTM tendeu a sobreajustar com poucos dados e convergiu mais lentamente que o ConvGRU.

## 4. Estratégias de treino

- **Pré-treino em tarefa correlata** (por exemplo, refletividade antes de taxa de chuva), como em Ko et al.
- **Fine-tuning online/incremental.** No benchmark HKO-7, os modelos com fine-tuning online superaram consistentemente os offline.
- **Transfer learning com encoder congelado** para horizontes longos, como no trabalho de LPPP.

## 5. Avaliação

Por causa do desbalanceamento, accuracy e RMSE globais são enganosos. A prática predominante é reportar **CSI, POD, FAR, HSS por limiar** de intensidade, mais **FSS** para tolerância espacial. A FACL propõe ainda a RHD (Regional Histogram Divergence) para medir similaridade de padrões com tolerância a deslocamentos locais. Escolha o *checkpoint* por métrica nos limiares altos, não pela loss de validação, que é dominada pelos pixels sem chuva.

## Síntese e lacunas

Minha leitura: o mínimo defensável é **dados com subamostragem de "sem chuva" + transformação log + loss ponderada (ou híbrida) + termo anti-borramento (GDL/FACL) + seleção de modelo por CSI/FSS em limiares altos**. Para batch e LR, faça seu próprio *sweep* (por exemplo, batch de 4 a 32 e lr de 1e-5 a 1e-3 com scheduler), porque a literatura quase não tem ablações. Isso, aliás, é uma lacuna publicável.

Posso aprofundar em algum eixo: detalhar as fórmulas (B-MSE, Nowcasting Loss, FACL), buscar ablações de batch/LR em satélite (GOES) ou IMERG, ou montar uma tabela comparativa para uma seção de related work.


Eu perguntando: pretendo usar a arquitetura de deep learning denominadaSTConvS2S.

# Treinamento de modelos espaço-temporais para precipitação com dados desbalanceados (STConvS2S)

Revisão da literatura e recomendações práticas · Outubro de 2026

## 1. Contexto

Dados de chuva são extremamente desbalanceados: a maior parte dos pixels e das sequências tem pouca ou nenhuma precipitação, e os eventos fortes, que são os de maior impacto, são raros. Este documento resume o que a literatura de nowcasting com deep learning recomenda para dados, função de custo, hiperparâmetros e avaliação, e como isso se aplica à arquitetura STConvS2S.

## 2. Dados e pré-processamento

- **Reamostragem de sequências.** O DGMR (DeepMind) rebalanceou o dataset para incluir mais observações de chuva forte e limitou a taxa máxima a 128 mm/h; os autores relatam que isso permitiu aprender previsões úteis. A survey de nowcasting com deep learning (Ko et al. e outros) lista ainda oversampling de casos chuvosos, amostragem hierárquica e amostragem por bins da distribuição.
- **Subamostragem de "sem chuva".** Um trabalho recente com radar e satélite manteve apenas 20% das imagens sem chuva, removeu outliers acima de 200 mm/h e normalizou com transformação logarítmica.
- **Transformação e máscara.** Prática comum em radar: converter refletividade em taxa de chuva (Marshall-Palmer), aplicar escala logarítmica com piso fixo, usar crops aleatórios e flips, e mascarar regiões fora da cobertura no cálculo da loss.

## 3. Função de custo

Este é o eixo mais bem estudado.

- **MSE e MAE puros são problemáticos.** Perdas pontuais tendem a gerar previsões excessivamente suavizadas e a representar mal a chuva forte.
- **Loss ponderada por intensidade (B-MSE/B-MAE).** Referência clássica do benchmark HKO-7 (Shi et al., NeurIPS 2017). As variantes balanceadas mostraram-se mais consistentes com o desempenho em múltiplos limiares e essenciais para bom desempenho em taxas altas. O DGMR também usa ponderação, com clipping do campo de chuva em \[0, 24\].
- **Hybrid Weighting.** Combina reponderação e não-ponderação, melhorando intensidades altas sem degradar as baixas, ao contrário dos métodos de reponderação puros.
- **Nowcasting Loss (Ko et al., 2022).** Loss baseada em CSI diferenciável, com termos para chuva em geral e chuva forte; a chuva forte entra nos dois termos, o que mitiga o desbalanceamento. Os autores reportam ganho de até 43,6% no CSI para chuva forte e propõem também um esquema de pré-treino em refletividade.
- **PP-Loss (Xu et al., 2024).** Atribui a cada amostra um peso inversamente proporcional à probabilidade de ocorrência do seu valor (plotting position). Em ConvLSTM com ERA5-Land, o desempenho foi comparável a MSE, Focal-R, Gumbel Loss e LDS; o ganho é mais uma alternativa baseada em regressão desbalanceada do que superioridade clara.
- **Termos contra o borramento.** O GDL (gradient difference loss) foi usado junto com a loss ponderada no FDNet, com melhora média de 0,007 em CSI e HSS no SRAD2020. A FACL (Fourier Amplitude and Correlation Loss, NeurIPS 2024) regulariza a amplitude no domínio de Fourier e complementa com a correlação de fase, substituindo MSE e MSE ponderado; é parameter-free.
- **Abordagem generativa.** DGMR e NowcastNet atacam o borramento com treino adversarial: o discriminador espacial garante consistência espacial e desencoraja previsões borradas, ao custo de maior complexidade de treino.

## 4. Hiperparâmetros reportados na literatura

Poucos trabalhos fazem ablação sistemática de batch size e learning rate; em geral só se relata a configuração final. Os valores abaixo são pontos de partida, não evidência causal.

| Trabalho | Otimizador e lr | Batch | Observações |
| --- | --- | --- | --- |
| Shi et al. 2017 (HKO-7) | Adam, 1e-4 | 4 (recorrentes), 8 (CNNs) | Loss balanceada |
| Ko et al. 2022 | Adam, 2e-5 | 20 (pré-treino), 24 (fine-tuning) | 50 mil e 35 mil passos |
| U-Net com fluxos ópticos (Weather and Forecasting) | Adam, 1e-4 | 8 | Único com busca explícita (lr 1e-5 a 1e-4, batch 8 a 64); StepLR, passo 15, gamma 0,5 |
| CasCast | AdamW, 1e-4 a 5e-4, cosine | 32 (global) | Autoencoder e difusão |
| Extreme Weather Nowcasting (LPPP) | AdamW, 1e-3, warm-up cosine | 16 | Treinado com FACL |
| Weather4Cast 2025 | Adam, 1e-3 | 25 | ConvLSTM tendeu a sobreajustar com poucos dados |

O resultado de batch pequeno (8) ter sido o melhor na única busca explícita sugere que o ruído do gradiente pode ajudar a não ignorar eventos raros, mas é um resultado isolado.

## 5. Estratégias de treino

- Pré-treino em tarefa correlata (por exemplo, refletividade antes de taxa de chuva), como em Ko et al.
- Fine-tuning online ou incremental: no benchmark HKO-7, os modelos com fine-tuning online superaram consistentemente os offline.
- Transfer learning com encoder congelado para horizontes mais longos.

## 6. Avaliação

Accuracy e RMSE globais são enganosos com dados desbalanceados. A prática predominante é reportar CSI, POD, FAR e HSS por limiar de intensidade, mais FSS para tolerância espacial. A FACL propõe ainda a RHD (Regional Histogram Divergence). A seleção de checkpoint deve usar a métrica nos limiares altos, não a loss de validação, que é dominada pelos pixels sem chuva.

## 7. STConvS2S: o que já foi reportado

- **Configuração original:** mini-batch, 50 épocas e RMSprop com learning rate 1e-3; loss de erro quadrático (RMSE/MSE), sem tratamento de desbalanceamento. A arquitetura é totalmente convolucional, e uma das variantes foi reportada como 23% melhor e cinco vezes mais rápida no treino que o baseline recorrente.
- **Extensões com sensoriamento remoto:** um trabalho posterior usou a STConvS2S como baseline com batch 25, MSE e early stopping, e mostrou que trocar a batch normalization padrão por EvoNormB0-3D trouxe melhoras significativas.
- **Outra aplicação:** uma arquitetura híbrida comparada à STConvS2S usou SGD com momentum 0,9, lr inicial 1e-3, redução de lr pela metade em platô, early stopping com paciência 10 e RMSE como loss.
- **Lacuna:** não foi encontrado estudo que aplique losses para dados desbalanceados (seção 3) à STConvS2S. Se isso se confirmar, é uma contribuição direta.

## 8. Recomendações para a STConvS2S

Síntese que combina a literatura com a estrutura da arquitetura; não são resultados validados na STConvS2S.

1. **Troque a loss, não a arquitetura.** A rede é agnóstica à loss, então B-MSE, Hybrid Weighting e PP-Loss entram sem mudança estrutural. Calcule os pesos na escala original (mm/h), antes da normalização ou do log. Como a rede prevê a sequência inteira em uma passada, também é possível ponderar por horizonte de previsão.
2. **Cuidado com batch normalization e desbalanceamento.** Batches pequenos podem ter estatísticas dominadas por pixels secos. Sugestões: amostragem estratificada para garantir sequências chuvosas em todo batch, e teste de EvoNorm ou GroupNorm, que não dependem do batch.
3. **Batch size e learning rate.** Varrer batch 8 a 32 e lr entre 1e-4 e 1e-3, com scheduler (ReduceLROnPlateau ou cosine com warmup) em vez de lr fixo.
4. **Seleção de checkpoint e early stopping** pela métrica nos limiares altos (CSI e FSS), não pela loss de validação.
5. **Máscara na loss** se o alvo tiver cobertura parcial (por exemplo, pluviômetros esparsos), para que posições sem observação não contribuam para o gradiente.
6. **Split cronológico.** A frequência de chuva forte varia entre anos; verifique se a validação tem eventos suficientes nos limiares altos, senão o early stopping fica ruidoso.

## 9. Experimento mínimo sugerido

Tabela de ablação com a STConvS2S fixa: (i) MSE, (ii) B-MSE, (iii) Hybrid Weighting, (iv) B-MSE com GDL ou FACL. Cada configuração com e sem subamostragem de "sem chuva", avaliando CSI, POD, FAR e FSS por limiar. Isso isola o efeito de loss e dados.

## 10. Referências

- Shi et al. Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model. NeurIPS 2017. https://arxiv.org/abs/1706.03458
- Ko et al. Effective Training Strategies for Deep-learning-based Precipitation Nowcasting and Estimation. Computers & Geosciences, 2022. https://arxiv.org/abs/2202.10555
- Xu et al. PP-Loss: An imbalanced regression loss based on plotting position for improved precipitation nowcasting. Theoretical and Applied Climatology, 2024. https://link.springer.com/article/10.1007/s00704-024-04984-w
- Yan et al. Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting. NeurIPS 2024. https://arxiv.org/abs/2410.23159
- Ravuri et al. Skilful precipitation nowcasting using deep generative models of radar. Nature, 2021. https://www.nature.com/articles/s41586-021-03854-z
- Deep learning for precipitation nowcasting: A survey from the perspective of time series forecasting. https://arxiv.org/abs/2406.04867
- Hybrid Weighting Loss for Precipitation Nowcasting from Radar Images. IEEE. https://ieeexplore.ieee.org/document/9746758/
- FDNet: A Deep Learning Approach with Two Parallel Cross Encoding Pathways for Precipitation Nowcasting. https://arxiv.org/abs/2105.02585
- CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling. https://arxiv.org/abs/2402.04290
- A Deep Learning Model for Precipitation Nowcasting Using Multiple Optical Flow Algorithms. Weather and Forecasting, 2023. https://journals.ametsoc.org/view/journals/wefo/39/1/WAF-D-23-0104.1.xml
- Computationally-efficient deep learning models for nowcasting of precipitation: Weather4cast 2025. https://arxiv.org/abs/2511.11197
- Enhancing Heavy Rain Nowcasting with Multimodal Data: Integrating Radar and Satellite Observations. https://arxiv.org/abs/2511.00716
- Nascimento et al. STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for weather forecasting. Neurocomputing. https://arxiv.org/abs/1912.00134
- Spatio-Temporal Forecasting With Gridded Remote Sensing Data. ACM SIGSPATIAL. https://dl.acm.org/doi/pdf/10.1145/3397536.3422247
- Sequence to sequence architecture based on hybrid LSTM global and local encoders for meteorological factors forecasting. PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12219302/
