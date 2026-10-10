# Resultados Auditados V3: Previsão e Alertas

## Protocolo

O dataset é `radar_sumare_2012_2024_15min_128_audited_v3`. O split
cronológico usa 2012--2021 para treinamento, 2022 para validação e 2023--2024
para teste. Cada exemplo contém cinco janelas de entrada e cinco horizontes de
15 minutos. As métricas usam somente pares estação--tempo observados e excluem
sequências que atravessam lacunas do radar.

Foram comparados B1 (persistência causal), C1 (radar RGB e previsão direta),
C3 (C1 mais mapas esparsos de chuva lagada e máscara na entrada convolucional)
e C3R (persistência causal mais correção aprendida a partir do radar RGB).
As redes foram executadas com as seeds 1000, 1001 e 1002. `média ± DP` indica
média e desvio-padrão amostral entre seeds, não intervalo de confiança por dia.

## Erros Contínuos

Teste 2023--2024, com 1.526.562 pares válidos. Unidades: mm/15 min.

| Modelo | RMSE | MAE | Viés |
|---|---:|---:|---:|
| B1 -- persistência | 0,6300 | 0,0751 | +0,0025 |
| C1 -- radar RGB | **0,5434 ± 0,0007** | **0,0672 ± 0,0022** | -0,0516 ± 0,0037 |
| C3 -- radar RGB + mapas lagados | 0,5427 ± 0,0070 | 0,0776 ± 0,0029 | -0,0388 ± 0,0093 |
| C3R -- persistência residual + radar RGB | 0,5718 ± 0,0099 | 0,0699 ± 0,0012 | **-0,0068 ± 0,0017** |

C1 tem o menor erro contínuo global, mas subestima sistematicamente a
precipitação. C3 reduz RMSE, mas aumenta MAE frente a B1. C3R não supera C1
em RMSE ou MAE, porém reduz substancialmente o viés por preservar B1 como
previsão base.

## Alertas Calibrados

Os limiares de decisão foram selecionados somente na validação de 2022 e
congelados antes do teste. O limiar observado define o evento; o limiar
calibrado somente define a emissão de alerta. Portanto, não há ajuste no teste.

### Métricas Categóricas por Estação

| Limiar observado | Modelo | Decisão | CSI | POD | FAR | ETS | Viés frequência |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,25 | B1 | 1,00 | 0,233 | **0,482** | 0,688 | 0,227 | 1,547 |
| 1,25 | C1 | 0,25 | 0,088 ± 0,004 | 0,164 ± 0,021 | 0,833 ± 0,037 | 0,083 ± 0,005 | 1,023 ± 0,308 |
| 1,25 | C3 | 0,25 | 0,075 ± 0,058 | 0,164 ± 0,184 | 0,850 ± 0,049 | 0,070 ± 0,056 | 0,970 ± 0,957 |
| 1,25 | C3R | 1,00 | **0,240 ± 0,001** | 0,414 ± 0,006 | **0,637 ± 0,007** | **0,234 ± 0,001** | 1,143 ± 0,038 |
| 6,25 | B1 | 6,25 | 0,112 | 0,210 | **0,807** | 0,111 | **1,088** |
| 6,25 | C1 | 0,33 ± 0,14 | 0,013 ± 0,004 | 0,083 ± 0,041 | 0,984 ± 0,004 | 0,012 ± 0,004 | 5,329 ± 3,110 |
| 6,25 | C3 | 0,33 ± 0,14 | 0,009 ± 0,009 | 0,044 ± 0,045 | 0,988 ± 0,010 | 0,008 ± 0,009 | 3,466 ± 0,724 |
| 6,25 | C3R | 4,00 | **0,115 ± 0,001** | **0,259 ± 0,001** | 0,828 ± 0,004 | **0,114 ± 0,001** | 1,508 ± 0,039 |
| 12,5 | B1 | 8,00 | 0,068 | **0,246** | 0,914 | 0,068 | 2,842 |
| 12,5 | C1 | 0,42 ± 0,29 | 0,003 ± 0,002 | 0,064 ± 0,054 | 0,997 ± 0,002 | 0,002 ± 0,002 | 17,078 ± 14,227 |
| 12,5 | C3 | 0,42 ± 0,29 | 0,003 ± 0,003 | 0,016 ± 0,012 | 0,997 ± 0,004 | 0,002 ± 0,003 | 8,655 ± 4,994 |
| 12,5 | C3R | 9,33 ± 1,15 | **0,088 ± 0,004** | 0,205 ± 0,023 | **0,864 ± 0,018** | **0,088 ± 0,004** | **1,545 ± 0,381** |

C1 e C3 requerem limiares de decisão muito inferiores ao limiar observado e,
mesmo assim, apresentam FAR próximo de 1 nos limiares forte e extremo. As
formulações diretas subestimam a magnitude e não servem como alerta operacional
nesta versão.

### Eventos Municipais

O evento municipal é um proxy: uma excedência em qualquer uma das 33 estações.
Excedências separadas por até 15 minutos são agrupadas. A tabela mostra a
fração detectada, a antecedência mediana e falsos eventos em todo o teste.

| Limiar observado | Modelo | Detecção | Antecedência mediana (min) | Falsos eventos |
|---|---|---:|---:|---:|
| 1,25 | B1 | **0,684** | 75,0 | 551 |
| 1,25 | C1 | 0,239 ± 0,059 | 70,0 ± 8,7 | **286,7 ± 113,3** |
| 1,25 | C3 | 0,393 ± 0,081 | 45,0 ± 15,0 | 486,3 ± 142,7 |
| 1,25 | C3R | 0,638 ± 0,018 | **75,0 ± 0,0** | 519,7 ± 24,2 |
| 6,25 | B1 | 0,508 | 45,0 | **129** |
| 6,25 | C1 | 0,276 ± 0,065 | **47,5 ± 15,6** | 293,7 ± 34,0 |
| 6,25 | C3 | 0,303 ± 0,041 | 40,0 ± 8,7 | 611,3 ± 91,3 |
| 6,25 | C3R | **0,601 ± 0,003** | 45,0 ± 0,0 | 205,7 ± 1,5 |
| 12,5 | B1 | **0,590** | **45,0** | 104 |
| 12,5 | C1 | 0,310 ± 0,173 | 40,0 ± 8,7 | 250,3 ± 92,4 |
| 12,5 | C3 | 0,280 ± 0,062 | 37,5 ± 7,5 | 536,0 ± 277,2 |
| 12,5 | C3R | 0,530 ± 0,044 | 32,5 ± 4,3 | **89,3 ± 15,5** |

## Conclusões

1. **C1 é a referência de regressão contínua**, mas não de alerta: sua
   subestimação impede decisões úteis nos limiares avaliados.
2. **C3 deve ser descartado como formulação principal.** Mapas esparsos de
   estação como canais convolucionais não melhoraram MAE nem alertas.
3. **C3R é o modelo neural mais promissor para alerta.** Supera B1 em CSI nos
   três limiares, reduz FAR em chuva fraca e extrema e detecta mais eventos
   municipais fortes: 0,601 contra 0,508.
4. **B1 continua difícil de superar.** Para o limiar extremo, detecta mais
   eventos municipais e com maior antecedência; C3R, porém, tem menos falsos
   eventos e CSI superior.

Uma conclusão operacional requer bootstrap pareado por dia, escolha explícita
do compromisso entre detecção e falsos alertas e teste em anos futuros
independentes, quando 2025--2026 estiverem auditados.

## Rastreabilidade

Fontes dos resultados categóricos e de evento:

- `outputs/analysis/evaluation/audited-v3-multiseed-alerts/alert_calibration.json`;
- `outputs/analysis/evaluation/audited-v3-multiseed-alerts/events_raw_vs_calibrated.json`.

Os resultados contínuos foram extraídos dos `summary.json` das três seeds de
C1, C3 e C3R. O protocolo de avaliação está em
[AVALIACAO_PREVISOES_AUDITAVEL.md](AVALIACAO_PREVISOES_AUDITAVEL.md).
