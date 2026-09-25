# Manutencao Presencial da Arietis

## Objetivo

Diagnosticar dois problemas independentes antes de voltar a usar a Arietis
para experimentos longos:

1. `/dev/sdb2` (`/home`) foi montado com erros EXT4 e requer verificacao.
2. As duas RTX 2080 Ti permanecem em P0, com clocks altos e consumo de
   aproximadamente 60--70 W mesmo sem processos CUDA.

Nao iniciar treinamento, nao executar `fsck` com `/dev/sdb2` montado e nao
atualizar o driver antes de fazer backup e registrar o estado atual.

## Evidencias Ja Coletadas

- GPUs: duas NVIDIA GeForce RTX 2080 Ti, driver `580.178.04`.
- Sem processos CUDA registrados por `nvidia-smi`, `fuser` ou `nvidia-smi pmon`.
- Ambas em P0, memoria em 7000 MHz e clocks graficos em 1350 MHz quando ociosas.
- Ambas aparecem com `display_active=Disabled`; em 580, o campo
  `display_mode` esta obsoleto.
- Topologia entre as GPUs: `PHB`; NVLink inativo.
- `nv_open_q` apareceu com uso alto de CPU em uma amostra, mas as pilhas
  posteriores estavam bloqueadas em `down_interruptible`, sem indicio de loop
  ativo naquele instante. Depois da atualizacao do driver, cada thread usava
  aproximadamente 3,5% de CPU.
- O runtime PM do Linux esta configurado como `auto`, mas as duas GPUs seguem
  `active`, com `runtime_usage` 4 e 3 e sem tempo suspenso. O parametro do
  driver `DynamicPowerManagement` esta em 3; portanto, o gerenciamento nao
  esta simplesmente desativado por configuracao.
- Parar `nvidia-persistenced` nao alterou P0, clocks, consumo nem as referencias
  runtime; o daemon nao e a causa imediata.
- Cada GPU esta em PCIe x8, esperado para a divisao x8/x8 das 16 lanes do
  processador. Os links reduzem para 2,5 GT/s em idle, sem erros AER, mas ASPM
  esta desativado tanto nas GPUs quanto nos root ports PCIe.
- A politica do kernel e `[default]`, sem `pcie_aspm=off` na linha de comando;
  a desativacao de ASPM provavelmente vem da BIOS/firmware.
- Placa-mae: Gigabyte Z390 GAMING SLI-CF; BIOS American Megatrends `F10c`, de
  2019-12-18.
- O relatorio NVIDIA ja foi gerado em `~/nvidia-bug-report.log.gz` (proprietario
  `root`).
- Kernel reportou:

  ```text
  EXT4-fs (sdb2): warning: mounting fs with errors, running e2fsck is recommended
  EXT4-fs (sdb2): error count since last fsck: 862146444
  htree_dirblock_to_tree
  ```

## 1. Preservar Dados

Antes de desligar ou reparar, copie dados insubstituiveis de `/home`, em
prioridade: repositorios com alteracoes locais, `outputs/`, configuracoes,
logs de GPU e datasets que nao existam em outro local. Use um disco externo ou
um host remoto confiavel. Confirme a origem de `/home`:

```bash
findmnt -no SOURCE,TARGET /home
df -h /home
```

Registre tambem os erros recentes:

```bash
sudo dmesg -T | grep -Ei 'EXT4-fs|I/O error|blk_update|ata|reset|timeout' \
  | tee ~/arietis-storage-kernel.log
sudo smartctl -x /dev/sdb | tee ~/arietis-smartctl-sdb.log
```

Se o SMART indicar setores pendentes/realocados, erros de leitura ou falha de
autoteste, trate o disco como suspeito e conclua o backup antes de prosseguir.

## 2. Reparar o Filesystem com Seguranca

O `fsck` precisa ocorrer com `/dev/sdb2` desmontado. Como ele contem `/home`,
o caminho mais seguro e iniciar por uma midia de recuperacao/live USB.

1. Inicie a maquina por live USB.
2. Identifique a particao, sem assumir o nome:

   ```bash
   lsblk -f
   ```

3. Execute o reparo somente depois de confirmar que a particao de `/home` nao
   esta montada:

   ```bash
   sudo fsck -f /dev/sdb2
   ```

4. Reinicie normalmente e confirme que nao restaram erros:

   ```bash
   sudo dmesg -T | grep -Ei 'EXT4-fs|I/O error|blk_update|ata|reset|timeout'
   sudo tune2fs -l /dev/sdb2 | grep -E 'Filesystem state|Error count|Last checked'
   ```

## 3. Inspecao Fisica das GPUs

Com a maquina desligada, desconectada da energia e apos descarregar energia
residual:

- Verificar se ambas as ventoinhas giram livremente e sem ruidos.
- Remover poeira de dissipadores, ventoinhas e filtros do gabinete.
- Confirmar fluxo de ar frontal/entrada e traseiro/saida.
- Verificar espacamento entre as duas placas; a GPU superior costuma receber ar
  mais quente.
- Conferir encaixe das GPUs e conectores PCIe de alimentacao.
- Identificar qual monitor esta ligado a cada GPU; a GPU 0 estava em modo de
  display e a GPU 1 nao.
- Registrar marca/modelo das placas, fonte de alimentacao e resolucao/taxa de
  atualizacao de cada monitor.

Nao remova dissipadores nem troque pasta/pads termicos sem experiencia e
materiais adequados; isso e uma etapa posterior caso a limpeza nao resolva.

## 4. Teste Controlado de Display e Estado Ocioso

Depois de remontar a maquina, sem treinamento em execucao, capture a linha de
base:

```bash
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,pstate,\
clocks.current.graphics,clocks.current.memory,fan.speed,display_active,display_mode \
  --format=csv,noheader
ps -eLo pid,tid,stat,pcpu,comm | grep -E 'nv_open_q|nvidia' || true
```

Teste somente uma variavel por vez:

1. Mantenha os dois monitores desligados/desconectados por 30 segundos e repita
   a consulta.
2. Reconecte apenas o monitor principal a 60 Hz e repita.
3. Aumente gradualmente a taxa de atualizacao e repita.

Interprete assim:

- Se apenas a GPU 0 reduzir consumo/P-state ao desligar ou baixar a taxa do
  monitor, o display explica parte do comportamento dela.
- Se a GPU 1 continuar em P0 sem monitor e sem processos, o problema nao e
  somente display.
- Se ambas permanecerem perto de 60--70 W e P0, registre o resultado para
  avaliar driver, BIOS e configuracao do sistema.

Verifique a politica PCIe antes de modificar configuracoes:

```bash
cat /proc/cmdline
cat /sys/module/pcie_aspm/parameters/policy

for bridge in 00:01.0 00:01.1; do
  echo "=== bridge $bridge ==="
  sudo lspci -s "$bridge" -vvv | sed -n '/LnkCap:/,/LnkSta2:/p'
done
```

Nao force `pcie_aspm=force` nem altere `/sys/module/pcie_aspm/parameters/policy`
remotamente. ASPM pode reduzir uma parcela do consumo do link PCIe, mas nao
explica sozinho memoria a 7000 MHz, P0 e 60--70 W por placa.

## 5. BIOS e Firmware da Placa-Mae

A BIOS `F10c` e de 2019 e precede varias revisoes de firmware da placa. Quando
o filesystem estiver verificado e houver backup, entre na BIOS e registre, antes
de qualquer alteracao, opcoes com nomes como `ASPM`, `PEG ASPM`, `PCIe ASPM`,
`Native ASPM` ou `PCIe Link State Power Management`.

Teste uma unica mudanca por vez e repita a medicao de estado ocioso. Se houver
uma opcao para habilitar ASPM, ela e candidata a teste, pois o kernel esta em
politica `default` mas os root ports reportam `ASPM Disabled`.

Planeje tambem atualizar a BIOS usando Q-Flash, pendrive FAT32 e alimentacao
confiavel. Para estabilidade, prefira a versao estavel F11 antes da F12a beta,
salvo necessidade especifica de uma correcao mais recente. A pagina oficial
informa que a F11 altera a estrutura da BIOS e impede retorno a versoes
anteriores; trate a atualizacao como irreversivel e nao a faca remotamente:

<https://www.gigabyte.com/Motherboard/Z390-GAMING-SLI-rev-10/Support>

Procedimento presencial de Q-Flash:

1. Fazer backup e confirmar a revisao impressa fisicamente na placa-mae antes
   de baixar o arquivo; a pagina acima e para Z390 GAMING SLI rev. 1.0.
2. Baixar a F11 da pagina oficial em outro computador, extrair o ZIP e copiar
   o arquivo de BIOS para a raiz de um pendrive FAT32.
3. Desligar processos de GPU, reiniciar, entrar na BIOS com `Del` e abrir
   `Q-Flash` (normalmente `F8`).
4. Selecionar `Update BIOS` e o arquivo no pendrive. Nao desligar a maquina,
   reiniciar, retirar o pendrive ou executar outras tarefas durante o flash.
5. Apos o reinicio, carregar os padroes otimizados caso a BIOS solicite e
   reconfigurar ordem de boot, virtualizacao, perfis de memoria e demais
   ajustes necessarios.
6. Confirmar a versao instalada no Linux:

   ```bash
   sudo dmidecode -t bios | grep -E 'Version:|Release Date:'
   ```

Uma atualizacao de BIOS pode melhorar microcodigo, seguranca e comportamento
da plataforma PCIe, mas nao e uma correcao garantida para P0 ou para o consumo
ocioso das GPUs. Repita os testes de ASPM e consumo depois da atualizacao e
altere apenas uma variavel por vez.

## 6. Coletar Diagnostico do Driver

Evite alterar modulos do kernel antes de coletar estas informacoes:

```bash
cat /proc/driver/nvidia/version
nvidia-smi -q -d PERFORMANCE,POWER,TEMPERATURE
sudo cat /proc/$(pgrep -o nv_open_q)/stack
sudo journalctl -k -b | grep -Ei 'NVRM|Xid|GSP|nvidia|timeout|error'
sudo nvidia-bug-report.sh
sudo chown "$(id -un):$(id -gn)" nvidia-bug-report.log.gz
```

O ultimo comando devolve ao usuario atual a propriedade do relatorio gerado
com `sudo`.

## 7. Decisao Antes de Treinar

So autorize experimentos GPU longos quando:

- o filesystem estiver verificado/reparado e sem novos erros de I/O;
- as GPUs estiverem em temperaturas ociosas razoaveis apos limpeza;
- um teste curto de CUDA nao gerar erros no `dmesg`;
- um teste de carga de 10--15 minutos mantiver temperaturas sustentadas em
  faixa segura, preferencialmente abaixo de 80 C.

Como medida temporaria de protecao termica para testes, apos confirmar que as
GPUs estao saudaveis:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 180
sudo nvidia-smi -i 1 -pl 180
```

O limite de potencia reduz a carga termica, mas nao corrige P0 em ociosidade.

## 8. Medidas Posteriores, Caso P0 Persista

Se o problema persistir apos os testes fisicos e de display, planeje uma janela
de manutencao para testar uma versao alternativa do driver NVIDIA suportada pelo
Ubuntu 22.04. Nao faca downgrade/upgrade remoto. Preserve antes o
`nvidia-bug-report.log.gz`, a versao atual do kernel e as saidas deste guia.

Nao configure `CUDA_DISABLE_PERF_BOOST=1` como correcao global: ela afeta
processos CUDA e pode reduzir desempenho de treinamento, sem resolver a causa
do estado P0 do driver.
