# Experimento de Leonardo: previsão de frames RGB do radar

## Objetivo

Demonstrar a STConvS2S em uma tarefa visual e espaço-temporal simples: usar os
cinco compósitos RGB anteriores do radar do Sumaré, com cadência de 15 minutos,
para prever os cinco compósitos RGB seguintes. Cada exemplo, portanto, contém
75 minutos de histórico e 75 minutos de previsão.

Este não é um experimento de precipitação em superfície. O acervo disponível
é o bitmap RGB do produto visual do radar, sem campo numérico calibrado de
refletividade. No relatório e nos slides, descreva a saída como **previsão de
frames RGB do radar**; não como refletividade quantitativa, chuva em mm/h ou
alerta pluviométrico.

## Protocolo mínimo

| Papel | Anos |
|---|---|
| Treino | 2018--2021 |
| Validação e escolha de época | 2022 |
| Teste para apresentação | 2023 |
| Holdout futuro, não usar na seleção | 2024 |

As janelas que cruzam lacunas de 15 minutos do acervo são descartadas. O
split é cronológico: 2023 não participa do treino, da escolha de época ou do
ajuste de hiperparâmetros.

São comparados dois métodos no mesmo teste:

1. **Persistência visual:** repete o último frame de entrada nos cinco
   horizontes futuros.
2. **STConvS2S-C:** aprende diretamente a sequência RGB futura.

As métricas são MAE e RMSE normalizados em `[0, 1]`, para todos os canais RGB,
e também somente nos pixels cujo target futuro possui algum canal não preto
(`active_echo`). Esta última evita que o fundo preto domine artificialmente a
métrica. Nenhuma dessas métricas tem unidade de precipitação.

## Execução

Na máquina com GPU e o dataset de radar instalado:

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate ailab
git switch feature/leonardo-radar-frame-nowcast
git pull --ff-only origin feature/leonardo-radar-frame-nowcast
git submodule update --init --recursive

dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3

PYTHONPATH=src python -u -m nowcasting.cli.train_radar_frames \
  --dataset-root "$dataset" \
  --train-years 2018-2021 \
  --val-years 2022 \
  --test-years 2023 \
  --model stconvs2s-c \
  --t-in 5 --t-out 5 --stride 5 \
  --batch-size 2 \
  --workers 2 \
  --epochs 30 \
  --patience 8 \
  --loss huber \
  --seed 1000 \
  --cuda 0 \
  --run-name radar-rgb-stconvs2s-seed-1000
```

## Artefatos para os slides

O diretório `outputs/experiments/radar-rgb-stconvs2s-seed-1000/`
conterá:

- `summary.json`: métricas da STConvS2S e da persistência no teste;
- `best.pt`: checkpoint escolhido exclusivamente pela validação de 2022;
- `qualitative_test_T+15.png` até `qualitative_test_T+75.png`: uma figura por
  horizonte, com último frame de entrada, persistência, STConvS2S e frame
  observado (gold);
- `configuration.json`: parâmetros e splits usados.

Para uma apresentação curta, reporte a tabela de MAE/RMSE global e de eco
ativo, a métrica por horizonte da `summary.json` e a figura qualitativa.

Para regenerar as figuras de um experimento já concluído, sem retreinar:

```bash
PYTHONPATH=src python -u -m nowcasting.cli.render_radar_frame_example \
  --run-dir outputs/experiments/radar-rgb-stconvs2s-seed-1000 \
  --dataset-root "$dataset" \
  --cuda 0
```

## Casos secos e chuvosos para discussão qualitativa

As estações do AlertaRio podem ser usadas somente para selecionar, de forma
retrospectiva, casos ilustrativos no teste. Isso não altera a função de perda,
o checkpoint, as métricas RGB nem simula uma decisão em tempo real. O caso
seco exige 0,0 mm/15 min em todas as estações e nos cinco horizontes; o caso
chuvoso exige ao menos 6,25 mm/15 min em alguma estação futura, com todas as
observações válidas.

Primeiro, gere candidatos de 2023:

```bash
run=outputs/experiments/radar-rgb-stconvs2s-seed-1000
showcases="$run/showcases"
mkdir -p "$showcases"

PYTHONPATH=src python -u -m nowcasting.cli.select_radar_frame_showcases \
  --dataset-root "$dataset" \
  --years 2023 \
  --mapping configs/mapeamento_pixel_estacao_alertario_historical_v1.csv \
  --t-in 5 --t-out 5 --stride 5 \
  --wet-threshold-mm15 6.25 \
  --candidates-per-class 5 \
  --output "$showcases/candidates_2023.json"
```

Inspecione `candidates_2023.json` e escolha um `example_index` seco e outro
chuvoso. Em seguida, gere as cinco figuras de cada caso. O título explicita a
classificação retrospectiva pelas estações:

```bash
PYTHONPATH=src python -u -m nowcasting.cli.render_radar_frame_example \
  --run-dir "$run" \
  --dataset-root "$dataset" \
  --example-index INDICE_SECO \
  --output-dir "$showcases/dry" \
  --output-prefix dry \
  --title "Caso seco retrospectivo: AlertaRio = 0,0 mm/15 min" \
  --cuda 0

PYTHONPATH=src python -u -m nowcasting.cli.render_radar_frame_example \
  --run-dir "$run" \
  --dataset-root "$dataset" \
  --example-index INDICE_CHUVOSO \
  --output-dir "$showcases/wet" \
  --output-prefix wet \
  --title "Caso chuvoso retrospectivo: maximo AlertaRio >= 6,25 mm/15 min" \
  --cuda 0
```

## Resultado da execução inicial

A primeira execução, `radar-rgb-stconvs2s-seed-1000`, usou a seed
1000 e selecionou a época 13 exclusivamente pela menor MAE de eco ativo na
validação de 2022 (0,2021). O treinamento durou aproximadamente 68 minutos
na GPU utilizada. O teste de 2023 contém 4.817 sequências cronologicamente
posteriores ao treino; cada uma contribui com cinco frames, três canais RGB e
uma grade de 128 por 128 pixels.

A Tabela abaixo resume a comparação no teste. Os valores estão na escala RGB
normalizada em `[0,1]`, e não têm unidade de chuva.

| Métrica no teste de 2023 | Persistência visual | STConvS2S-C | Variação da STConvS2S-C |
|---|---:|---:|---:|
| MAE global | 0,01004 | 0,00863 | -14,0% |
| RMSE global | 0,07730 | 0,06616 | -14,4% |
| MAE em eco ativo | 0,18178 | 0,22354 | +23,0% |
| RMSE em eco ativo | 0,32688 | 0,35809 | +9,5% |

A métrica global favorece a STConvS2S-C a partir de T+30: os MAEs da rede nos
horizontes T+15, T+30, T+45, T+60 e T+75 foram, respectivamente, 0,00829,
0,00835, 0,00868, 0,00889 e 0,00895. Para persistência, os valores foram
0,00719, 0,00947, 0,01059, 0,01123 e 0,01171. Assim, persistência é melhor
no primeiro passo, mas a rede reduz o erro global nos quatro horizontes mais
distantes.

Entretanto, a rede foi pior nos pixels onde o target continha eco visual. O
resultado correto não é que a rede previu melhor chuva: ela reduziu o erro
médio do bitmap RGB, mas ainda não representou adequadamente as regiões mais
informativas do radar. Isso é esperado em uma primeira configuração, pois o
fundo preto é dominante e o produto RGB também pode conter artefatos visuais.

## Uso na apresentação

O Leonardo pode apresentar este experimento como uma demonstração auditável
da tarefa espaço-temporal, não como um sistema operacional de alerta. Uma
sequência curta de slides pode seguir este roteiro:

1. **Pergunta:** cinco imagens RGB passadas conseguem antecipar os cinco
   frames futuros do radar?
2. **Dados e split:** mostrar a cadência de 15 minutos, as janelas de 75
   minutos e a separação cronológica 2018--2021 / 2022 / 2023.
3. **Métodos:** explicar a persistência visual e a STConvS2S-C. Persistência
   apenas repete o último frame; a rede aprende padrões de evolução no treino.
4. **Resultados:** usar a tabela acima e enfatizar que a rede melhora o erro
   RGB global depois de T+15, mas não melhora a reconstrução de eco ativo.
5. **Figuras qualitativas:** usar uma figura por horizonte, de
   `qualitative_test_T+15.png` a `qualitative_test_T+75.png`. Cada uma coloca
   previsão e gold lado a lado, preservando também o último frame de entrada
   e a persistência como referência. A seleção atual contém padrões radiais
   verdes no gold de T+45 e T+60; apresente-os como exemplo de artefato do
   produto visual e de uma dificuldade que a rede não resolveu, não como uma
   tempestade prevista pela rede.
6. **Conclusão e continuidade:** o pipeline, o split temporal e a comparação
   com baseline funcionam. Os próximos passos são selecionar casos
   qualitativos meteorologicamente interpretáveis, filtrar artefatos e, se
   possível, usar refletividade numérica calibrada em vez do bitmap RGB.

Uma formulação segura para a conclusão oral é: ``A STConvS2S aprendeu a
reduzir o erro médio dos frames RGB em horizontes maiores, mas a análise dos
pixels com eco e da figura qualitativa mostra que ainda não podemos interpretar
o resultado como melhoria na previsão de precipitação.''

## Figuras recomendadas para Leonardo

Para a apresentação, foi selecionado um caso chuvoso do **teste independente
de 2023**, e não do conjunto de validação: a sequência cujo primeiro horizonte
futuro começa em 7 de fevereiro de 2023 às 20:15 UTC. O máximo observado nas
33 estações do AlertaRio ao longo dos cinco horizontes foi 33,8 mm/15 min.

As figuras aprovadas para uso estão em:

```text
outputs/experiments/radar-rgb-stconvs2s-seed-1000/
showcases/wet-2023-0207/
```

| Arquivo | Horizonte |
|---|---:|
| `wet-2023-0207_T+15.png` | 15 min |
| `wet-2023-0207_T+30.png` | 30 min |
| `wet-2023-0207_T+45.png` | 45 min |
| `wet-2023-0207_T+60.png` | 60 min |
| `wet-2023-0207_T+75.png` | 75 min |

Cada figura deve ser lida da seguinte forma: o painel superior esquerdo é o
último frame de radar disponível na entrada; o superior direito é a
persistência visual; o inferior esquerdo é a previsão da STConvS2S-C; e o
inferior direito é o frame RGB observado, denominado “gold”. O gold é o
produto visual posterior do radar, não uma medida independente de chuva.

Para os slides principais, recomenda-se mostrar T+15, T+45 e T+75 em três
slides consecutivos ou em uma sequência animada. Ao apresentá-los, Leonardo
deve comparar principalmente a STConvS2S-C com o gold e usar persistência
como referência: em T+15, repetir o último frame é naturalmente competitivo;
nos horizontes maiores, a rede pode reduzir o erro global, mas as imagens e a
métrica de eco ativo mostram que estruturas coloridas ainda são difíceis de
reconstruir. As duas figuras intermediárias podem ficar como material de apoio.

Não usar essas imagens para afirmar que a rede previu chuva em milímetros,
refletividade calibrada ou um alerta operacional. A formulação apropriada é
que elas ilustram a previsão de imagens RGB do radar em um evento que,
retrospectivamente, teve chuva forte registrada pelas estações.
