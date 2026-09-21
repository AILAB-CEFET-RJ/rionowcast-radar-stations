# Geração do Dataset - Radar Sumaré + Alerta Rio

Este documento apresenta o processo de preparação dos dados pluviométricos do Alerta Rio e sua integração ao dataset anual do Radar Meteorológico do Sumaré.

O fluxo completo contém as seguintes etapas:

1. obtenção e extração dos arquivos ZIP do Alerta Rio;
2. conversão dos arquivos para o formato Parquet;
3. associação das estações aos pixels do radar;
4. alinhamento temporal com as imagens do Radar do Sumaré;
5. geração anual dos targets e máscaras em formato `memmap`.

## 1. Organização dos arquivos do Alerta Rio

Os dados pluviométricos do Alerta Rio estão compactados no formato ZIP.

```text
    └── alertario/
        └── pluviometricos_all/
            ├── arquivo_001.parquet
            ├── arquivo_002.parquet
            ├── arquivo_003.parquet
            └── ...
```

Nesta base foram identificados 471 arquivos pluviométricos.

- Apesar de os arquivos extraídos apresentarem a extensão `.parquet`, seu conteúdo original deve ser lido como CSV. Por isso, eles ainda precisam ser convertidos para arquivos Parquet verdadeiros antes da geração dos targets.

## 2. Análise exploratória e conversão para Parquet

A preparação inicial é realizada no código de análise exploratória do Alerta Rio.

Nessa etapa são executadas:

* leitura dos arquivos extraídos;
* verificação das colunas disponíveis;
* padronização dos tipos;
* tratamento dos timestamps;
* verificação dos identificadores das estações;
* análise da precipitação acumulada em 15 minutos;
* conversão dos arquivos para o formato Parquet verdadeiro.

Os arquivos de entrada são lidos a partir de:

```text
/home/noemi/atmoseer/data/alertario/pluviometricos_all
```

Exemplo da leitura necessária:

```python
df = pd.read_csv(caminho_arquivo)
```

Depois do tratamento, cada arquivo é salvo com:

```python
df.to_parquet(
    caminho_saida,
    index=False,
    compression="snappy",
)
```

Os arquivos convertidos são armazenados em:

```text
/home/noemi/atmoseer/data/alertario/pluviometricos_parquet
```

A estrutura resultante é:

```text
atmoseer/
└── data/
    └── alertario/
        ├── pluviometricos_all/
        │   ├── arquivo_001.parquet
        │   └── ...
        │
        └── pluviometricos_parquet/
            ├── arquivo_001.parquet
            ├── arquivo_002.parquet
            └── ...
```

Os arquivos da pasta `pluviometricos_all` representam os dados extraídos dos ZIPs. Os arquivos da pasta `pluviometricos_parquet` representam os dados já convertidos e preparados para as próximas etapas.

As principais informações utilizadas posteriormente são:

| Coluna       | Descrição                              |
| ------------ | -------------------------------------- |
| `estacao_id` | Identificador da estação pluviométrica |
| `dia_utc`    | Data e horário da observação em UTC    |
| `m15`        | Precipitação acumulada em 15 minutos   |

## 3. Dataset de radar utilizado

A integração utiliza um dataset do Radar do Sumaré previamente processado e separado por ano:

```text
/home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano
```

Estrutura inicial esperada:

```text
radar_sumare_2012_2024_15min_256_por_ano/
├── year=2012/
│   ├── radar_frames.dat
│   ├── radar_timestamps.npy
│   └── metadata.json
├── year=2013/
│   └── ...
└── year=2024/
    ├── radar_frames.dat
    ├── radar_timestamps.npy
    └── metadata.json
```

O arquivo `radar_frames.dat` contém as imagens do radar:

* agregadas temporalmente em intervalos de 15 minutos;
* redimensionadas para `256 × 256` pixels;
* separadas por ano;
* armazenadas no formato `memmap`.

O arquivo `radar_timestamps.npy` contém os timestamps correspondentes aos frames do radar.

O arquivo `metadata.json` contém o formato e o tipo dos dados necessários para abrir o `memmap`.

O script de geração dos targets do Alerta Rio não gera novamente os frames do radar. Ele utiliza esse dataset de radar já processado como referência espacial e temporal.

## 4. Mapeamento das estações para a grade do radar

Para colocar a precipitação das estações na mesma grade das imagens, utiliza-se o arquivo:

```text
mapeamento_pixel_estacao_alertario.csv
```

Local utilizado:

```text
/home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano/mapeamento_pixel_estacao_alertario.csv
```

Esse arquivo associa cada estação do Alerta Rio a um pixel da grade do Radar do Sumaré.

As principais informações são:

| Coluna                       | Descrição                               |
| ---------------------------- | --------------------------------------- |
| `station_id` ou `estacao_id` | Identificador da estação                |
| `pixel_i`                    | Linha correspondente na grade do radar  |
| `pixel_j`                    | Coluna correspondente na grade do radar |

As coordenadas originalmente calculadas na grade de aproximadamente `656 × 654` são convertidas para a resolução utilizada no treinamento:

```text
256 × 256
```

A conversão segue a relação:

```python
pixel_i_256 = round(pixel_i * 256 / 656)
pixel_j_256 = round(pixel_j * 256 / 654)
```

Depois da conversão, os valores são limitados ao intervalo válido da grade:

```text
0 a 255
```

## 5. Geração dos targets e máscaras

Depois da conversão dos arquivos pluviométricos e da criação do mapeamento espacial, os targets e máscaras são gerados pelo script:

```text
scripts/radar_sumare/gerar_targets_masks_alertario_memmap_por_ano.py
```

O script:

1. lê os Parquets da pasta `pluviometricos_parquet`;
2. lê o arquivo de mapeamento das estações;
3. abre os timestamps do radar de cada ano;
4. associa as observações pluviométricas aos respectivos timestamps;
5. posiciona cada observação no pixel correspondente;
6. converte a precipitação para `log1p(mm/15min)`;
7. gera uma máscara indicando as observações válidas;
8. salva os targets e máscaras anuais em formato `memmap`.

### Comando utilizado

```bash
python scripts/radar_sumare/gerar_targets_masks_alertario_memmap_por_ano.py \
  --alertario-root /home/noemi/atmoseer/data/alertario/pluviometricos_parquet \
  --mapping /home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano/mapeamento_pixel_estacao_alertario.csv \
  --radar-root /home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --year-start 2012 \
  --year-end 2024 \
  --height 256 \
  --width 256
```

## 6. Parâmetros do script

| Parâmetro          |          Valor utilizado | Descrição                                        |
| ------------------ | -----------------------: | ------------------------------------------------ |
| `--alertario-root` | `pluviometricos_parquet` | Diretório com os Parquets tratados do Alerta Rio |
| `--mapping`        |              arquivo CSV | Mapeamento entre estações e pixels do radar      |
| `--radar-root`     | diretório anual do radar | Diretório de entrada e saída do dataset          |
| `--year-start`     |                   `2012` | Primeiro ano processado                          |
| `--year-end`       |                   `2024` | Último ano processado                            |
| `--height`         |                    `256` | Altura da grade espacial de saída                |
| `--width`          |                    `256` | Largura da grade espacial de saída               |

Os argumentos disponíveis na versão atual do script podem ser consultados com:

```bash
python scripts/radar_sumare/gerar_targets_masks_alertario_memmap_por_ano.py --help
```

## 7. Resolução espacial

A resolução espacial dos targets e máscaras pode ser configurada por linha de comando:

```bash
--height 256 \
--width 256
```

Nesta execução, o formato gerado é:

```text
[tempo, 256, 256, 1]
```

Os valores de `--height` e `--width` precisam ser compatíveis com a resolução espacial de `radar_frames.dat`.

Não se deve alterar somente a resolução dos targets. Caso os targets sejam gerados em uma resolução diferente da usada nos frames do radar, haverá incompatibilidade entre os tensores de entrada e saída do modelo.

## 8. Resolução temporal

A resolução temporal não possui um argumento próprio nesse script.

O alinhamento é determinado pelos timestamps encontrados em:

```text
radar_timestamps.npy
```

Como o dataset de radar foi previamente agregado em intervalos de 15 minutos, os dados do Alerta Rio também são alinhados em intervalos de 15 minutos.

Portanto, a configuração atual utiliza:

```text
resolução temporal: 15 minutos
resolução espacial: 256 × 256 pixels
```

Para usar outra resolução temporal, seria necessário primeiro gerar um novo dataset de radar com a frequência desejada e garantir que os dados pluviométricos fossem alinhados aos novos timestamps.

## 9. Arquivos gerados

Em cada pasta anual, o script gera:

```text
year=AAAA/
├── Y_alertario.dat
├── M_alertario.dat
└── targets_alertario_metadata.json
```

### `Y_alertario.dat`

Contém os targets esparsos de precipitação das estações do Alerta Rio.

Características:

```text
dtype: float32
unidade original: mm/15min
transformação: log1p(mm/15min)
shape: [tempo, 256, 256, 1]
```

Para retornar os valores à unidade original:

```python
precipitacao_mm_15min = np.expm1(target_log)
```

### `M_alertario.dat`

Contém a máscara binária das observações:

```text
dtype: uint8
```

Os valores representam:

```text
1 = existe uma observação válida no pixel e horário
0 = não existe uma observação válida
```

A máscara é necessária porque os targets são esparsos: somente os pixels associados às estações possuem observações reais de precipitação.

### `targets_alertario_metadata.json`

Contém as informações necessárias para reabrir os arquivos `memmap`, incluindo:

* nomes dos arquivos;
* `dtype`;
* `shape`;
* unidade;
* transformação aplicada;
* fonte dos dados;
* resolução espacial e temporal.

## 10. Estrutura final do dataset

Após a execução, a estrutura esperada é:

```text
radar_sumare_2012_2024_15min_256_por_ano/
├── mapeamento_pixel_estacao_alertario.csv
├── year=2012/
│   ├── radar_frames.dat
│   ├── radar_timestamps.npy
│   ├── metadata.json
│   ├── Y_alertario.dat
│   ├── M_alertario.dat
│   └── targets_alertario_metadata.json
├── year=2013/
│   └── ...
└── year=2024/
    ├── radar_frames.dat
    ├── radar_timestamps.npy
    ├── metadata.json
    ├── Y_alertario.dat
    ├── M_alertario.dat
    └── targets_alertario_metadata.json
```

Caso os arquivos da WebSirene estejam presentes, eles podem permanecer nas mesmas pastas:

```text
Y_all.dat
M_all.dat
targets_metadata.json
```

A geração dos arquivos do Alerta Rio não deve sobrescrever os arquivos da WebSirene.

## 11. Redução espacial de um dataset existente

Quando os memmaps em `256 x 256` já existem, é possível produzir uma versão
menor sem reprocessar os PNGs brutos do radar. O script
`scripts/downsample_memmap_dataset.py` lê o dataset de origem em blocos,
preservando o uso de memória limitado.

O exemplo abaixo cria `128 x 128` para 2024. O diretório de saída deve ser
novo: o script se recusa a sobrescrever arquivos existentes.

```bash
python scripts/downsample_memmap_dataset.py \
  --source-root /home/ebezerra/ailab/stconvs2s/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --output-root /home/ebezerra/ailab/stconvs2s/data/datasets/radar_sumare_2012_2024_15min_128_por_ano \
  --year-start 2024 \
  --year-end 2024 \
  --height 128 \
  --width 128 \
  --target-source alertario \
  --chunk-size 64
```

Os frames RGB são reduzidos por vizinho mais próximo. Os pixels observados nos
targets e máscaras são remapeados para a nova grade; se duas estações caírem
no mesmo pixel, é mantido o maior valor de precipitação. Os timestamps e a
resolução temporal permanecem inalterados.

Para processar simultaneamente os targets WebSirene e AlertaRio, use
`--target-source both`, que é o valor padrão.

## 12. Resumo do fluxo

```text
Arquivos ZIP do Alerta Rio
        ↓
Extração para pluviometricos_all
        ↓
Análise exploratória e tratamento
        ↓
Conversão dos 471 arquivos
        ↓
Criação de pluviometricos_parquet
        ↓
Mapeamento estação → pixel do radar
        ↓
Alinhamento com radar_timestamps.npy
        ↓
Geração de Y_alertario.dat
        ↓
Geração de M_alertario.dat
        ↓
Geração de targets_alertario_metadata.json
```
