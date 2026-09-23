# Arquitetura e Limites de Responsabilidade

## Componentes

| Componente | Responsabilidade | Local |
|---|---|---|
| Core STConvS2S | Arquiteturas neurais genéricas | `external/stconvs2s` (submódulo) |
| Projeto nowcasting | Dados, preprocessamento, treino, avaliacao e CLIs | `src/nowcasting` |
| Dados | Insumos, temporarios e datasets de treino | `data/` |
| Resultados | Experimentos, analises, tabelas e figuras geradas | `outputs/` |

## Regra de Dependência

O projeto pode importar modelos do submódulo, mas o submódulo não pode
importar módulos deste repositório. Isso permite atualizar ou substituir a
arquitetura sem alterar a lógica científica do experimento.

`nowcasting-train` (modulo `nowcasting.cli.train`) funciona como o ponto de entrada do treinamento. Ele recebe uma raiz de
dataset, três conjuntos explícitos de anos e opcionalmente um caminho
alternativo para o core. Por padrão, usa `external/stconvs2s`.

## Convenções

- Não adicionar memmaps, ZIPs, checkpoints ou logs ao Git.
- Registrar experimentos em `outputs/experiments/<run-name>/`.
- Cada execução grava `configuration.json`, incluindo o commit do core usado.
- Toda nova fonte de estações, loss com hipótese meteorológica ou métrica por
  intensidade deve ser implementada em `src/nowcasting`, não no submódulo.
- Entry points de linha de comando ficam em `src/nowcasting/cli`; cada um deve
  delegar a logica de dominio para modulos importaveis do pacote.
- Dados e resultados gerados nao pertencem a `src/`. Consulte
  `docs/ORGANIZACAO_DADOS_E_RESULTADOS.md` para a classificacao de caminhos.
- Mudanças no core só devem ser propostas quando forem independentes de Radar
  Sumaré, AlertaRio, WebSirene e de unidades de precipitação.
