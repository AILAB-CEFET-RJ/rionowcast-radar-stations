# Guia: download auditável de canais ABI do GOES-16

Este guia prepara uma máquina Linux ou WSL para gerar os dados históricos do
canais ABI do GOES-16. O resultado é integrado ao dataset existente de radar do
Sumaré e das estações AlertaRio.

O pipeline baixa arquivos públicos da NOAA, reprojeta as cenas para a grade do
radar e mantém somente o tensor final. Os NetCDF Full Disk são temporários e
apagados automaticamente.

## Antes de começar

Você precisará de Linux ou WSL2, Internet, `git`, Conda ou Miniconda
(`conda --version`), acesso ao GitHub e uma cópia do dataset-base fornecida por
Eduardo. O Git **não** contém os dados grandes de radar ou estações.

O dataset-base ocupa aproximadamente 18 GB. Um ano do canal C13 ocupa cerca de
1,7 GB; a pilha de cinco canais disponíveis neste projeto ocupa cerca de 8,2
GB por ano. Mantenha ao menos 25 GB livres após copiar o dataset-base para C13,
ou 35 GB para a pilha de cinco canais:

```bash
df -h .
```

O guia usa a branch `feature/goes-integration-audit`. Execute-o somente após
Eduardo confirmar que essa branch está publicada no GitHub.

## 1. Clonar o repositório

```bash
mkdir -p ~/ailab
cd ~/ailab
git clone --recurse-submodules \
  --branch feature/goes-integration-audit \
  https://github.com/AILAB-CEFET-RJ/rionowcast-radar-stations.git
cd rionowcast-radar-stations
```

Para atualizar um clone já existente:

```bash
cd ~/ailab/rionowcast-radar-stations
git fetch origin
git switch feature/goes-integration-audit
git pull --ff-only origin feature/goes-integration-audit
git submodule update --init --recursive
```

## 2. Preparar o ambiente Python

Crie o ambiente uma única vez:

```bash
conda create -n rionowcast-goes python=3.10 -y
conda activate rionowcast-goes
```

Em cada terminal novo, execute:

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate rionowcast-goes
```

Instale as dependências e os comandos do projeto:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
nowcasting-stream-goes16-memmaps --help
```

Se aparecer a tela de ajuda, a instalação está correta. Se `conda` não existir
ou a instalação falhar, envie o erro a Eduardo e não instale pacotes do sistema
operacional sem orientação.

## 3. Copiar e verificar o dataset-base

Copie a pasta fornecida por Eduardo para este caminho:

```text
~/ailab/rionowcast-radar-stations/data/datasets/
  radar_sumare_2012_2024_15min_128_audited_v3/
```

Defina variáveis e confira os arquivos de um ano. No exemplo, `2022` pode ser
trocado pelo ano desejado:

```bash
cd ~/ailab/rionowcast-radar-stations
dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
year=2022
year_dir="$dataset/year=$year"

for file in radar_frames.dat radar_timestamps.npy metadata.json \
  targets_alertario_sparse.npz targets_alertario_metadata.json; do
  test -s "$year_dir/$file" && echo "OK: $file" || echo "AUSENTE: $file"
done
```

Todos devem aparecer como `OK`. Caso algum esteja ausente, não inicie o
download; solicite uma cópia completa do dataset-base.

## 4. Escolher canais e baixar um ano

Há duas configurações auditáveis. A primeira baixa somente C13. A segunda
baixa a pilha C08, C09, C13, C14 e C15. Nesta versão, somente essas duas opções
são suportadas; não altere manualmente a lista de canais nos arquivos JSON.

| Opção | Configuração | Canais | Espaço aproximado por ano |
| --- | --- | --- | --- |
| Econômica | `goes16_abi_c13_audited_v2.json` | C13 | 1,7 GB |
| Multicanal | `goes16_abi_audited_v2.json` | C08, C09, C13, C14, C15 | 8,2 GB |

O bloco abaixo processa **2022** com C13. Para outro ano, altere somente
`year`. Para a opção multicanal, altere somente a variável `goes_config` para
o segundo arquivo da tabela. O pipeline exige todos os canais selecionados em
cada frame; se um deles estiver indisponível, a janela é marcada como ausente.
Ele usa quatro downloads concorrentes e grava checkpoint a cada 25 frames.

```bash
cd ~/ailab/rionowcast-radar-stations
conda activate rionowcast-goes

dataset=data/datasets/radar_sumare_2012_2024_15min_128_audited_v3
year=2022
goes_config=configs/goes16_abi_c13_audited_v2.json
run="goes16-abi-${year}-v2"

nohup env PYTHONPATH=src nice -n 10 python -u \
  -m nowcasting.cli.stream_goes16_memmaps \
  --dataset-root "$dataset" \
  --goes-config "$goes_config" \
  --radar-config configs/radar_sumare_historical_audited_v1.json \
  --year-start "$year" --year-end "$year" \
  --download-workers 4 \
  --checkpoint-interval 25 \
  --retries 3 \
  --resume \
  > "${run}.log" 2>&1 < /dev/null &

echo $! > "${run}.pid"
echo "Processo iniciado: PID $(cat "${run}.pid")"
echo "Log: ${run}.log"
```

O processo continua após fechar o terminal. Não execute o mesmo comando duas
vezes para o mesmo ano: dois processos não podem escrever no mesmo ano.

## 5. Acompanhar a execução

Confira o processo:

```bash
ps -p "$(cat goes16-abi-2022-v2.pid)" -o pid,etime,stat,%cpu,%mem,cmd
```

Veja as últimas linhas do log ou acompanhe continuamente:

```bash
tail -n 30 goes16-abi-2022-v2.log
tail -f goes16-abi-2022-v2.log
```

`Ctrl+C` sai apenas do `tail`; não interrompe o downloader. A linha de sucesso
é semelhante a:

```text
[2022] GOES streaming concluído | disponíveis=.../... | destino=.../year=2022/goes16
```

## 6. Retomar após queda de energia, Internet ou reinicialização

Se a execução parar antes da mensagem de sucesso, **não apague**:

```text
data/datasets/.../year=2022/goes16.partial
```

Primeiro confirme que não há downloader ativo:

```bash
pgrep -af 'nowcasting.cli.stream_goes16_memmaps' || true
```

Se não houver processo, repita exatamente o bloco da Seção 4 com o mesmo ano
e `--resume`. O pipeline verifica o contrato e processa somente frames
pendentes. Não use `--overwrite` nem remova `goes16.partial` sem orientação de
Eduardo.

## 7. Auditar o ano concluído

Após a mensagem de sucesso, confirme a publicação:

```bash
test -f "$dataset/year=2022/goes16/metadata.json" \
  && echo "2022 FINALIZADO" \
  || echo "2022 NÃO FINALIZADO"
```

Execute a auditoria:

```bash
nowcasting-audit-goes16 \
  --dataset-root "$dataset" \
  --year-start 2022 --year-end 2022 \
  --output-dir outputs/analysis/goes16/c13_2022_v2
```

O resultado deve terminar com algo como:

```text
Auditoria GOES concluída | cobertura=...% | erros causais=0 | saída=...
```

Envie a Eduardo a última linha e o resumo:

```bash
cat outputs/analysis/goes16/c13_2022_v2/summary.json
```

Cobertura menor que 100% pode ocorrer por ausência de cenas, idade máxima
excedida ou valores inválidos sobre a área reprojetada. O requisito obrigatório
é `causality_errors: 0`: nenhuma cena pode terminar depois do início da janela
de radar correspondente.

## 8. Próximo ano e cuidados

Repita as Seções 3, 4 e 7, substituindo `2022` pelo próximo ano. Execute apenas
um ano por vez na máquina, para simplificar acompanhamento, retomada e auditoria.

- Não copie, mova ou apague arquivos dentro de `year=<ano>/goes16` depois da
  finalização.
- Não altere os arquivos de configuração citados no comando.
- Não preencha frames ausentes com zero: o pipeline registra indisponibilidade
  e o loader exclui sequências que cruzam lacunas.
- Não use o resultado sem executar a auditoria.
