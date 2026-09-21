# Reprodução do Treinamento STConvS2S - Radar Sumaré + WebSirene

Este documento descreve os passos necessários para reproduzir o treinamento da arquitetura STConvS2S-C utilizando imagens do Radar Meteorológico do Sumaré e precipitação das estações WebSirene.

## 1. Dataset utilizado

O treinamento utiliza o dataset anual em formato `memmap`, com imagens de radar e alvos/máscaras por ano.

- Não é necessário clonar o repositório do atmoseer, já que iremos usar apenas os arquivos dos zips quando rodar o modelo.

Diretório do dataset:

`atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano`

- Adicionar os zips que enviei pelo Teams, nessa pasta, como no exemplo a seguir:

Estrutura esperada:

```bash
radar_sumare_2012_2024_15min_256_por_ano/
├── year=2012/
│   ├── radar_frames.dat
│   ├── radar_timestamps.npy
│   ├── metadata.json
│   ├── Y_all.dat
│   ├── M_all.dat
│   └── targets_metadata.json
├── year=2013/
│   └── ...
...
└── year=2024/
    └── ...
```

- Ao executar o 'main.py', pelo comando do Passo '3. Rodar o treinamento', o código carrega os arquivos `memmap` de cada ano, monta os conjuntos de treino, validação e teste e inicia o treinamento.

O arquivo `radar_frames.dat` contém as imagens de radar agregadas em intervalos de 15 minutos e redimensionadas para `256 x 256` pixels.
O arquivo `Y_all.dat` contém os alvos esparsos de precipitação das estações WebSirene, armazenados na escala `log1p(mm/15min)`.
O arquivo `M_all.dat` contém a máscara binária indicando onde existe observação pluviométrica disponível.


## 2. Clonar e instalar o repositório STConvs2s

Clonar a branch `noemi` do repositório:

```bash
git clone -b noemi https://github.com/noemicho/stconvs2s.git
cd stconvs2s
```

As dependências do projeto estão especificadas em `config/environment.yml`. 

```bash
conda env create -f config/environment.yml
conda activate pytorch
```

Alternativa com pip:
```bash
python -m pip install torch torchvision matplotlib ipykernel h5py pandas xarray dask bottleneck statsmodels scikit-learn cartopy
```

## 3. Rodar o treinamento

O treinamento é executado pelo arquivo `main.py`, utilizando o modelo `stconvs2s-c`.

Comando utilizado:

```bash
nohup python -u main.py \
  -m stconvs2s-c \
  -dsp /atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --years 2012-2024 \
  -b 4 \
  -e 2 \
  -i 1 \
  -w 0 \
  -s 5 \
  -c 0 \
  --verbose \
  > resultado_stconvs2s_2012_2024_b4_e2_log1p.log 2>&1 &
```

## 4. Parâmetros principais

| Parâmetro   |    Valor usado | Descrição                          |
| ----------- | -------------: | ---------------------------------- |
| `-m`        |  `stconvs2s-c` | Modelo utilizado                   |
| `-dsp`      | dataset memmap | Caminho do dataset                 |
| `--years`   |    `2012-2024` | Anos usados no treinamento         |
| `-b`        |            `4` | Tamanho do batch                   |
| `-e`        |            `2` | Número de épocas                   |
| `-i`        |            `1` | Número de iterações                |
| `-w`        |            `0` | Número de workers do DataLoader    |
| `-s`        |            `5` | Número de passos futuros previstos |
| `-c`        |            `0` | GPU utilizada                      |
| `--verbose` |        ativado | Exibe progresso no log             |

## 5. Acompanhar o treinamento

Para acompanhar a execução:

```bash
tail -f resultado_stconvs2s_2012_2024_b4_e2_log1p.log
```

Para verificar se o processo ainda está rodando:

```bash
pgrep -af "main.py"
```

## 6. Saídas geradas

Os checkpoints são salvos automaticamente no diretório de saída do projeto, em uma estrutura semelhante a:

```bash
/stconvs2s/output/full-dataset/checkpoints/stconvs2s-c/
```

O treinamento também gera arquivos de log com as métricas globais, métricas por horizonte e métricas por faixa de intensidade de precipitação, calculadas após o retorno dos valores para mm/15min.

## 7. Exemplo de log de saída do treinamento

- Nessa nova versão foram adicionados: Avaliação de Viés por Horizonte e treinamento com valores de chuva normalizados com log.
- Exemplo da versão antiga: 

```bash
RUN MODEL: STCONVS2S-C
Device: GPU
Settings: Namespace(version=0, iteration=1, epoch=2, batch=4, patience=16, workers=0, cuda='0', step='5', model='stconvs2s-c', num_layers=3, hidden_dim=32, kernel_size=5, pre_trained=None, email=False, plot=False, verbose=True, no_seed=False, no_stop=False, small_dataset=False, chirps=False, dataset_path='/home/ncho/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano', output_channels=None, years='2012-2024')
Usando RadarStationMemmapDataset a partir de: /home/ncho/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano
[train] Ano 2012 carregado | frames=34192 | shape=(34192, 256, 256, 3) | amostras possíveis=34183
[train] Ano 2013 carregado | frames=32256 | shape=(32256, 256, 256, 3) | amostras possíveis=32247
[train] Ano 2014 carregado | frames=32996 | shape=(32996, 256, 256, 3) | amostras possíveis=32987
[train] Ano 2015 carregado | frames=29979 | shape=(29979, 256, 256, 3) | amostras possíveis=29970
[train] Ano 2016 carregado | frames=31839 | shape=(31839, 256, 256, 3) | amostras possíveis=31830
[train] Ano 2017 carregado | frames=29164 | shape=(29164, 256, 256, 3) | amostras possíveis=29155
[train] Ano 2018 carregado | frames=29282 | shape=(29282, 256, 256, 3) | amostras possíveis=29273
[train] Ano 2019 carregado | frames=26580 | shape=(26580, 256, 256, 3) | amostras possíveis=26571
[train] Ano 2020 carregado | frames=23707 | shape=(23707, 256, 256, 3) | amostras possíveis=23698
[train] Ano 2021 carregado | frames=29560 | shape=(29560, 256, 256, 3) | amostras possíveis=29551
[train] Ano 2022 carregado | frames=28391 | shape=(28391, 256, 256, 3) | amostras possíveis=28382
[train] Ano 2023 carregado | frames=27190 | shape=(27190, 256, 256, 3) | amostras possíveis=27181
[train] Ano 2024 carregado | frames=25505 | shape=(25505, 256, 256, 3) | amostras possíveis=25496
[train] Total de amostras: 45666
[val] Ano 2012 carregado | frames=34192 | shape=(34192, 256, 256, 3) | amostras possíveis=34183
[val] Ano 2013 carregado | frames=32256 | shape=(32256, 256, 256, 3) | amostras possíveis=32247
[val] Ano 2014 carregado | frames=32996 | shape=(32996, 256, 256, 3) | amostras possíveis=32987
[val] Ano 2015 carregado | frames=29979 | shape=(29979, 256, 256, 3) | amostras possíveis=29970
[val] Ano 2016 carregado | frames=31839 | shape=(31839, 256, 256, 3) | amostras possíveis=31830
[val] Ano 2017 carregado | frames=29164 | shape=(29164, 256, 256, 3) | amostras possíveis=29155
[val] Ano 2018 carregado | frames=29282 | shape=(29282, 256, 256, 3) | amostras possíveis=29273
[val] Ano 2019 carregado | frames=26580 | shape=(26580, 256, 256, 3) | amostras possíveis=26571
[val] Ano 2020 carregado | frames=23707 | shape=(23707, 256, 256, 3) | amostras possíveis=23698
[val] Ano 2021 carregado | frames=29560 | shape=(29560, 256, 256, 3) | amostras possíveis=29551
[val] Ano 2022 carregado | frames=28391 | shape=(28391, 256, 256, 3) | amostras possíveis=28382
[val] Ano 2023 carregado | frames=27190 | shape=(27190, 256, 256, 3) | amostras possíveis=27181
[val] Ano 2024 carregado | frames=25505 | shape=(25505, 256, 256, 3) | amostras possíveis=25496
[val] Total de amostras: 15222
[test] Ano 2012 carregado | frames=34192 | shape=(34192, 256, 256, 3) | amostras possíveis=34183
[test] Ano 2013 carregado | frames=32256 | shape=(32256, 256, 256, 3) | amostras possíveis=32247
[test] Ano 2014 carregado | frames=32996 | shape=(32996, 256, 256, 3) | amostras possíveis=32987
[test] Ano 2015 carregado | frames=29979 | shape=(29979, 256, 256, 3) | amostras possíveis=29970
[test] Ano 2016 carregado | frames=31839 | shape=(31839, 256, 256, 3) | amostras possíveis=31830
[test] Ano 2017 carregado | frames=29164 | shape=(29164, 256, 256, 3) | amostras possíveis=29155
[test] Ano 2018 carregado | frames=29282 | shape=(29282, 256, 256, 3) | amostras possíveis=29273
[test] Ano 2019 carregado | frames=26580 | shape=(26580, 256, 256, 3) | amostras possíveis=26571
[test] Ano 2020 carregado | frames=23707 | shape=(23707, 256, 256, 3) | amostras possíveis=23698
[test] Ano 2021 carregado | frames=29560 | shape=(29560, 256, 256, 3) | amostras possíveis=29551
[test] Ano 2022 carregado | frames=28391 | shape=(28391, 256, 256, 3) | amostras possíveis=28382
[test] Ano 2023 carregado | frames=27190 | shape=(27190, 256, 256, 3) | amostras possíveis=27181
[test] Ano 2024 carregado | frames=25505 | shape=(25505, 256, 256, 3) | amostras possíveis=25496
[test] Total de amostras: 15223
Train samples: 45666
Val samples: 15222
Test samples: 15223
Sample X: torch.Size([3, 5, 256, 256])
Sample Y: torch.Size([1, 5, 256, 256])
Sample M: torch.Size([1, 5, 256, 256])
Train on 45666 samples, validate on 15222 samples
Shape enviado ao modelo: (1, 3, 5, 256, 256)
Output channels: 1
Epoch: 1/2 - loss: 0.0421 - val_loss: 0.0399
=> Saving a new best
Epoch: 2/2 - loss: 0.0399 - val_loss: 0.0401
=> Early stopping counter: 1 out of 16

Training time: 42:05:46.36 [151546.36048460007]
=> Loaded checkpoint radar_sumare_2012_2024_15min_256_por_ano_step5_0_20260622-015122.pth.tar (best epoch: 1, validation rmse: 0.0399)
Training time/epochs: 2:28:34.49 [8914.491793211768]
>>>>>>>>> Metric per observation (lat x lon) at each time step (t)
RMSE
0.04387046408532152,0.04343853327705372,0.04271225508517663,0.04260155831621496,0.042310842305087935
MAE
0.04387046409198955,0.04343853328494525,0.042712255091592315,0.04260155832019896,0.042310842313958257
>>>>>>>>

>>>>>>>>> Metric by precipitation intensity
Fraca (<5 mm/h equiv.): n=3445265, RMSE=0.1144, MAE=0.0219 Bias=-0.0213
Moderada (5-25 mm/h equiv.): n=26807, RMSE=2.7738, MAE=2.5003 Bias=-2.5003
Forte (25-50 mm/h equiv.): n=2739, RMSE=8.8158, MAE=8.6423 Bias=-8.6423
Extrema (>50 mm/h equiv.): n=1179, RMSE=18.7762, MAE=17.9354 Bias=-17.9354
>>>>>>>>
Test RMSE: 0.0430
Test MAE: 0.0430
Test Bias: -0.0425
```

## 8. Observação sobre unidades

A unidade nativa da precipitação é:

```bash
mm/15min
```

As faixas de intensidade também são apresentadas em taxa horária equivalente:

```bash
mm/h equivalente = 4 × mm/15min
```

Assim:

| Faixa          |     Unidade nativa | Unidade equivalente |
| -------------- | -----------------: | ------------------: |
| Chuva fraca    |    < 1,25 mm/15min |            < 5 mm/h |
| Chuva moderada | 1,25–6,25 mm/15min |           5–25 mm/h |
| Chuva forte    | 6,25–12,5 mm/15min |          25–50 mm/h |
| Chuva extrema  |    > 12,5 mm/15min |           > 50 mm/h |


