# Arquitetura e Limites de Responsabilidade

## Componentes

| Componente | Responsabilidade | Local |
|---|---|---|
| Core STConvS2S | Arquiteturas neurais genéricas | `external/stconvs2s` (submódulo) |
| Projeto nowcasting | Dados, splits temporais, losses por chuva, sampler e métricas | `src/nowcasting` |
| Automação | Geração de datasets e execução de experimentos | `scripts/` |
| Artefatos | Memmaps, checkpoints, logs e gráficos | Ignorados pelo Git |

## Regra de Dependência

O projeto pode importar modelos do submódulo, mas o submódulo não pode
importar módulos deste repositório. Isso permite atualizar ou substituir a
arquitetura sem alterar a lógica científica do experimento.

`scripts/train_nowcasting.py` é o ponto de entrada. Ele recebe uma raiz de
dataset, três conjuntos explícitos de anos e opcionalmente um caminho
alternativo para o core. Por padrão, usa `external/stconvs2s`.

## Convenções

- Não adicionar memmaps, ZIPs, checkpoints ou logs ao Git.
- Registrar experimentos em `outputs/experiments/<run-name>/`.
- Cada execução grava `configuration.json`, incluindo o commit do core usado.
- Toda nova fonte de estações, loss com hipótese meteorológica ou métrica por
  intensidade deve ser implementada em `src/nowcasting`, não no submódulo.
- Mudanças no core só devem ser propostas quando forem independentes de Radar
  Sumaré, AlertaRio, WebSirene e de unidades de precipitação.
