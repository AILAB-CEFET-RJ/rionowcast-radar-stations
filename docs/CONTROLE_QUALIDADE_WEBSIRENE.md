# Controle De Qualidade WebSirene

## Objetivo

O WebSirene amplia de forma relevante a cobertura espacial da rede de pluviometros,
mas nao deve entrar como supervisao de treinamento sem um controle de qualidade
reproduzivel. Esta etapa preserva os Parquets brutos e cria uma versao auditada,
com flags por observacao e uma whitelist preliminar por estacao.

## Politica V1

A configuracao versionada esta em
[`configs/websirene_qc_v1.json`](../configs/websirene_qc_v1.json). Ela marca:

| Categoria | Regra |
|---|---|
| Rejeitada | data/hora invalida, `m15` invalido, negativo ou superior a 175 mm/15 min |
| Suspeita | `m15` superior a 50 mm/15 min, conflito no mesmo timestamp ou sequencia positiva constante |
| Informativa | duplicata identica e lacuna superior a 30 minutos |

Uma estacao e aprovada preliminarmente quando possui pelo menos 1.000 observacoes
aceitas em tres anos distintos e nao ultrapassa 1% de observacoes rejeitadas ou
suspeitas. Esses limiares sao operacionais, nao uma afirmacao climatologica: devem
ser revisados apos inspecao de eventos e comparacao externa.

## Executar A Auditoria

O diretorio de entrada deve seguir o contrato
`station_id=<id>/year=<ano>/data.parquet`. Os dados brutos nao sao alterados.

```bash
nowcasting-audit-websirene \
  --input-root data/raw/websirene \
  --processed-root data/processed/websirene_qc/v1 \
  --analysis-dir outputs/analysis/websirene/qc/v1 \
  --config configs/websirene_qc_v1.json \
  --year-start 2012 \
  --year-end 2024
```

Para repetir deliberadamente a mesma versao de saida, acrescente `--overwrite`.
As saidas principais sao:

- `data/processed/websirene_qc/v1/.../data.parquet`: observacoes originais com
  `timestamp_15min`, `qc_flags`, `qc_status` e `qc_version`.
- `outputs/analysis/websirene/qc/v1/station_summary.csv`: indicadores e status
  preliminar por estacao.
- `outputs/analysis/websirene/qc/v1/station_whitelist.csv`: apenas estacoes
  aprovadas preliminarmente.
- `outputs/analysis/websirene/qc/v1/flagged_observations.csv`: linhas suspeitas
  ou rejeitadas para revisao.

`data/processed/` e `outputs/` sao artefatos locais e permanecem fora do Git.

## Comparar Com AlertaRio

Antes de usar uma estacao aprovada como target, compare suas leituras aceitas com
a estacao AlertaRio mais proxima. O comando abaixo associa apenas pares separados
por no maximo 5 km e calcula `MAE`, viés e correlacao de Pearson em timestamps de
15 minutos coincidentes.

```bash
nowcasting-compare-websirene-alertario \
  --websirene-root data/processed/websirene_qc/v1 \
  --alertario-root data/raw/alertario/pluviometricos_parquet \
  --websirene-mapping data/mapeamento_pixel_estacao.csv \
  --alertario-mapping data/mapeamento_pixel_estacao_alertario.csv \
  --output-dir outputs/analysis/websirene/qc/v1/alertario_comparison \
  --year-start 2012 \
  --year-end 2024 \
  --max-distance-km 5 \
  --min-pairs 100
```

Esta comparacao e evidência adicional, nao substitui a revisao de eventos extremos:
pluviometros proximos podem divergir por variabilidade espacial real da chuva.

## Gerar Targets Auditados

O gerador legado `nowcasting-build-websirene-targets --ws-root ...` mantem o corte
historico de 50 mm/15 min. Para o novo experimento, use a fonte auditada. Apenas
linhas `qc_status=accepted` sao consumidas, e nao ha novo corte por intensidade:

```bash
nowcasting-build-websirene-targets \
  --qc-root data/processed/websirene_qc/v1 \
  --station-whitelist outputs/analysis/websirene/qc/v1/station_whitelist.csv \
  --mapping data/mapeamento_pixel_estacao.csv \
  --radar-root data/datasets/radar_sumare_2012_2024_15min_128_por_ano \
  --year-start 2012 \
  --year-end 2024 \
  --height 128 \
  --width 128 \
  --height-orig 656 \
  --width-orig 654
```

O primeiro experimento com WebSirene deve usar somente a whitelist e reportar
separadamente o desempenho por rede e por faixa de intensidade.
