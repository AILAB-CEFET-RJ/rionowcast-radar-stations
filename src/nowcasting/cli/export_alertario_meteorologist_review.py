#!/usr/bin/env python3
"""Export AlertaRio QC findings as a workbook for meteorological review."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


SPATIAL_STATUS_LABELS = {
    "below_spatial_screening_threshold": "Abaixo do limiar de triagem espacial",
    "spatially_supported_local": "Ocorrência apoiada por vizinho local",
    "spatially_supported_extended": "Ocorrência apoiada no raio ampliado",
    "gauge_isolated_radar_supported": "Isolado nos pluviômetros, com eco RGB",
    "websirene_only_support": "Apoio positivo somente do WebSirene",
    "multi_source_unsupported": "Sem apoio das fontes disponíveis",
    "insufficient_spatial_context": "Contexto espacial insuficiente",
}

DISPLAY_COLUMN_NAMES = {
    "Prioridade para revisao": "Prioridade para revisão",
    "Estacao": "Estação",
    "ID da estacao": "ID da estação",
    "Episodio": "Episódio",
    "Inicio - horario local": "Início - horário local",
    "Fim - horario local": "Fim - horário local",
    "Inicio - UTC": "Início - UTC",
    "Timestamps sinalizados": "Horários sinalizados",
    "Maximo m05 (mm/5 min)": "Máximo m05 (mm/5 min)",
    "Maximo m15 (mm/15 min)": "Máximo m15 (mm/15 min)",
    "Maior divergencia dos acumulados (mm)": "Maior divergência dos acumulados (mm)",
    "Timestamps com inconsistencia temporal": "Horários com inconsistência temporal",
    "Timestamps com eco RGB no radar": "Horários com eco RGB no radar",
    "Timestamps com vizinho AlertaRio molhado": "Horários com vizinho AlertaRio molhado",
    "Timestamps com consenso WebSirene": "Horários com consenso WebSirene",
    "Diagnostico automatico": "Diagnóstico automático",
    "Motivo da revisao": "Motivo da revisão",
    "Comentario da Juliana": "Comentário da Juliana",
    "Prioridade do episodio": "Prioridade do episódio",
    "Data e hora UTC": "Data e hora UTC",
    "Inconsistencia temporal": "Inconsistência temporal",
    "Submetido a triagem espacial": "Submetido à triagem espacial",
    "Distancia do vizinho molhado mais proximo (km)": "Distância do vizinho molhado mais próximo (km)",
    "Diagnostico espacial": "Diagnóstico espacial",
}

PRIORITY_ORDER = {"Alta": 0, "Média": 1, "Baixa": 2}
PRIORITY_FILLS = {
    "Alta": "F4CCCC",
    "Média": "FFF2CC",
    "Baixa": "D9EAD3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera planilha amigável para revisão meteorológica do QC AlertaRio."
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-workbook", type=Path, required=True)
    parser.add_argument("--output-questions", type=Path, required=True)
    parser.add_argument("--timezone", default="America/Sao_Paulo")
    parser.add_argument("--extreme-threshold-mm15", type=float, default=100.0)
    parser.add_argument("--spatial-screening-threshold-mm15", type=float, default=6.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _yes_no(value: object, *, unavailable: bool = False) -> str:
    if pd.isna(value):
        return "Não disponível" if unavailable else ""
    return "Sim" if bool(value) else "Não"


def _format_timestamp(value: object, timezone: str) -> tuple[str, str]:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "", ""
    utc_text = timestamp.strftime("%d/%m/%Y %H:%M UTC")
    local = timestamp.tz_convert(ZoneInfo(timezone))
    local_text = local.strftime("%d/%m/%Y %H:%M %Z")
    return local_text, utc_text


def _review_priority(
    row: pd.Series,
    *,
    extreme_threshold_mm15: float,
    spatial_screening_threshold_mm15: float,
) -> tuple[str, str]:
    max_m15 = float(row.get("max_m15", np.nan))
    max_error = float(row.get("max_consistency_abs_error_mm", np.nan))
    if bool(row.get("high_priority_episode", False)):
        return (
            "Alta",
            "Combina inconsistência temporal e pico sem apoio das fontes disponíveis.",
        )
    if np.isfinite(max_m15) and max_m15 >= extreme_threshold_mm15:
        return (
            "Alta",
            "Magnitude extrema; apoio de ocorrência não confirma o valor medido.",
        )
    if np.isfinite(max_m15) and max_m15 >= spatial_screening_threshold_mm15:
        return "Média", "Chuva relevante que requer confirmação meteorológica."
    if np.isfinite(max_error) and max_error >= 1.0:
        return "Média", "Divergência relevante entre acumulados temporais."
    return "Baixa", "Divergência pequena; revisar a semântica e o arredondamento."


def prepare_episode_sheet(
    episodes: pd.DataFrame,
    *,
    timezone: str,
    extreme_threshold_mm15: float,
    spatial_screening_threshold_mm15: float,
) -> pd.DataFrame:
    required = {
        "station_id",
        "station_name",
        "episode_number",
        "start_timestamp",
        "end_timestamp",
        "candidate_timestamps",
        "max_m05",
        "max_m15",
        "max_consistency_abs_error_mm",
        "temporal_inconsistent_count",
        "multi_source_unsupported_count",
        "radar_supported_count",
        "alertario_supported_count",
        "websirene_consensus_count",
        "high_priority_episode",
        "spatial_statuses",
    }
    if missing := required - set(episodes.columns):
        raise ValueError(f"Resumo de episodios sem colunas: {sorted(missing)}")

    rows = []
    for row in episodes.to_dict("records"):
        series = pd.Series(row)
        priority, reason = _review_priority(
            series,
            extreme_threshold_mm15=extreme_threshold_mm15,
            spatial_screening_threshold_mm15=spatial_screening_threshold_mm15,
        )
        start_local, start_utc = _format_timestamp(row["start_timestamp"], timezone)
        end_local, end_utc = _format_timestamp(row["end_timestamp"], timezone)
        statuses = [
            SPATIAL_STATUS_LABELS.get(status, status)
            for status in str(row["spatial_statuses"]).split("|")
        ]
        rows.append({
            "Prioridade para revisao": priority,
            "Estacao": row["station_name"],
            "ID da estacao": int(row["station_id"]),
            "Episodio": int(row["episode_number"]),
            "Inicio - horario local": start_local,
            "Fim - horario local": end_local,
            "Inicio - UTC": start_utc,
            "Fim - UTC": end_utc,
            "Timestamps sinalizados": int(row["candidate_timestamps"]),
            "Maximo m05 (mm/5 min)": row["max_m05"],
            "Maximo m15 (mm/15 min)": row["max_m15"],
            "Maior divergencia dos acumulados (mm)": row[
                "max_consistency_abs_error_mm"
            ],
            "Timestamps com inconsistencia temporal": int(
                row["temporal_inconsistent_count"]
            ),
            "Picos sem apoio multissensor": int(
                row["multi_source_unsupported_count"]
            ),
            "Timestamps com eco RGB no radar": int(row["radar_supported_count"]),
            "Timestamps com vizinho AlertaRio molhado": int(
                row["alertario_supported_count"]
            ),
            "Timestamps com consenso WebSirene": int(row["websirene_consensus_count"]),
            "Diagnostico automatico": "; ".join(statuses),
            "Motivo da revisao": reason,
            "Parecer da Juliana": "",
            "Comentario da Juliana": "",
        })
    result = pd.DataFrame(rows)
    result["_priority"] = result["Prioridade para revisao"].map(PRIORITY_ORDER)
    return result.sort_values(
        ["_priority", "Maximo m15 (mm/15 min)", "Maior divergencia dos acumulados (mm)"],
        ascending=[True, False, False],
    ).drop(columns="_priority").reset_index(drop=True)


def _episode_lookup(episodes: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    lookup: dict[int, list[dict[str, object]]] = {}
    for row in episodes.to_dict("records"):
        lookup.setdefault(int(row["station_id"]), []).append({
            **row,
            "start": pd.to_datetime(row["start_timestamp"], utc=True),
            "end": pd.to_datetime(row["end_timestamp"], utc=True),
        })
    return lookup


def prepare_observation_sheet(
    observations: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    timezone: str,
) -> pd.DataFrame:
    required = {
        "station_id",
        "station_name",
        "timestamp",
        "m05",
        "m10",
        "m15",
        "h01",
        "previous_m05",
        "next_m05",
        "previous_m15",
        "next_m15",
        "expected_m10_from_m05",
        "expected_m15_from_m05",
        "expected_h01_from_m05",
        "m10_consistency_error_mm",
        "m15_consistency_error_mm",
        "h01_consistency_error_mm",
        "temporal_inconsistent",
        "spatially_screened",
        "alertario_extended_window_wet_count",
        "alertario_extended_window_max_m15",
        "alertario_extended_window_nearest_wet_distance_km",
        "radar_context_available",
        "radar_echo_present",
        "websirene_context_available",
        "websirene_consensus_support",
        "spatial_status",
    }
    if missing := required - set(observations.columns):
        raise ValueError(f"Observacoes espaciais sem colunas: {sorted(missing)}")

    lookup = _episode_lookup(episodes)
    rows = []
    for row in observations.to_dict("records"):
        timestamp = pd.to_datetime(row["timestamp"], utc=True)
        episode = next(
            (
                item
                for item in lookup.get(int(row["station_id"]), [])
                if item["start"] <= timestamp <= item["end"]
            ),
            None,
        )
        local_text, utc_text = _format_timestamp(timestamp, timezone)
        rows.append({
            "Prioridade do episodio": (
                episode["Prioridade para revisao"] if episode is not None else ""
            ),
            "Estacao": row["station_name"],
            "ID da estacao": int(row["station_id"]),
            "Episodio": int(episode["episode_number"]) if episode is not None else "",
            "Data e hora local": local_text,
            "Data e hora UTC": utc_text,
            "m05 (mm/5 min)": row["m05"],
            "m10 (mm/10 min)": row["m10"],
            "m15 (mm/15 min)": row["m15"],
            "h01 (mm/1 h)": row["h01"],
            "m05 anterior": row["previous_m05"],
            "m05 seguinte": row["next_m05"],
            "m15 anterior": row["previous_m15"],
            "m15 seguinte": row["next_m15"],
            "m10 esperado pela soma de m05": row["expected_m10_from_m05"],
            "m15 esperado pela soma de m05": row["expected_m15_from_m05"],
            "h01 esperado pela soma de m05": row["expected_h01_from_m05"],
            "Erro de consistencia m10 (mm)": row["m10_consistency_error_mm"],
            "Erro de consistencia m15 (mm)": row["m15_consistency_error_mm"],
            "Erro de consistencia h01 (mm)": row["h01_consistency_error_mm"],
            "Inconsistencia temporal": _yes_no(row["temporal_inconsistent"]),
            "Submetido a triagem espacial": _yes_no(row["spatially_screened"]),
            "Vizinhos AlertaRio molhados (ate 10 km)": int(
                row["alertario_extended_window_wet_count"]
            ),
            "Maior m15 vizinho (mm/15 min)": row[
                "alertario_extended_window_max_m15"
            ],
            "Distancia do vizinho molhado mais proximo (km)": row[
                "alertario_extended_window_nearest_wet_distance_km"
            ],
            "Frame de radar disponivel": _yes_no(
                row["radar_context_available"], unavailable=True
            ),
            "Eco RGB proximo no radar": (
                _yes_no(row["radar_echo_present"])
                if bool(row["radar_context_available"])
                else "Não disponível"
            ),
            "WebSirene disponivel": _yes_no(row["websirene_context_available"]),
            "Consenso positivo WebSirene": (
                _yes_no(row["websirene_consensus_support"])
                if bool(row["websirene_context_available"])
                else "Não disponível"
            ),
            "Diagnostico espacial": SPATIAL_STATUS_LABELS.get(
                str(row["spatial_status"]), str(row["spatial_status"])
            ),
        })
    result = pd.DataFrame(rows)
    result["_priority"] = result["Prioridade do episodio"].map(PRIORITY_ORDER)
    return result.sort_values(
        ["_priority", "Estacao", "Data e hora UTC"], na_position="last"
    ).drop(columns="_priority").reset_index(drop=True)


def _style_tabular_sheet(worksheet, *, priority_column: int | None = 1) -> None:
    dark_fill = PatternFill("solid", fgColor="1F4E78")
    white_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="D9E2F3")
    for cell in worksheet[1]:
        cell.fill = dark_fill
        cell.font = white_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin_gray)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.sheet_view.showGridLines = False
    worksheet.row_dimensions[1].height = 42

    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        if priority_column is not None:
            priority = row[priority_column - 1].value
            if priority in PRIORITY_FILLS:
                row[priority_column - 1].fill = PatternFill(
                    "solid", fgColor=PRIORITY_FILLS[priority]
                )

    for index, column_cells in enumerate(worksheet.columns, start=1):
        values = [str(cell.value or "") for cell in column_cells[:100]]
        maximum = max((len(value) for value in values), default=10)
        width = min(max(maximum + 2, 12), 42)
        worksheet.column_dimensions[get_column_letter(index)].width = width


def _write_readme_sheet(
    workbook,
    *,
    episode_count: int,
    priority_count: int,
    observation_count: int,
    timezone: str,
) -> None:
    sheet = workbook["LEIA-ME"]
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["B"].width = 110
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Revisão meteorológica das observações AlertaRio de 2026"
    sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    sheet["A1"].alignment = Alignment(horizontal="center", vertical="center")
    sheet.row_dimensions[1].height = 28

    content = [
        ("Objetivo", "Solicitar avaliação meteorológica de casos sinalizados por testes temporais e espaciais. Nenhuma linha foi automaticamente declarada incorreta por esta planilha."),
        ("Escopo", f"{episode_count} episódios de revisão e {observation_count} horários de 2026; {priority_count} episódios receberam prioridade alta para leitura inicial."),
        ("Horário", f"A planilha apresenta horário local ({timezone}) e UTC. Use o horário local ao consultar registros operacionais."),
        ("Radar", "Eco RGB indica apenas sinal visual próximo. Não representa refletividade quantitativa nem valida a magnitude da precipitação."),
        ("Estações vizinhas", "Uma vizinha molhada apoia a ocorrência de chuva na região, mas não confirma que a intensidade extrema da estação candidata esteja correta."),
        ("WebSirene", "Não havia dados WebSirene de 2026. A ausência aparece como 'Não disponível' e não foi tratada como evidência negativa."),
        ("Como responder", "Na aba TODOS OS EPISÓDIOS, preencher 'Parecer da Juliana' com uma das opções da LEGENDA e acrescentar observações, fontes consultadas ou informações de manutenção."),
        ("Gerado em", date.today().strftime("%d/%m/%Y")),
    ]
    for row_index, (label, value) in enumerate(content, start=3):
        sheet.cell(row=row_index, column=1, value=label).font = Font(bold=True)
        sheet.cell(row=row_index, column=1).fill = PatternFill("solid", fgColor="D9EAF7")
        sheet.cell(row=row_index, column=2, value=value)
        for column in (1, 2):
            sheet.cell(row=row_index, column=column).alignment = Alignment(
                vertical="top", wrap_text=True
            )
        sheet.row_dimensions[row_index].height = 45

    question_start = len(content) + 5
    sheet.merge_cells(start_row=question_start, start_column=1, end_row=question_start, end_column=2)
    sheet.cell(row=question_start, column=1, value="Perguntas principais").font = Font(
        size=13, bold=True, color="FFFFFF"
    )
    sheet.cell(row=question_start, column=1).fill = PatternFill("solid", fgColor="4472C4")
    for offset, question in enumerate(review_questions(), start=1):
        sheet.cell(row=question_start + offset, column=1, value=offset)
        sheet.cell(row=question_start + offset, column=2, value=question)
        sheet.cell(row=question_start + offset, column=2).alignment = Alignment(wrap_text=True)
        sheet.row_dimensions[question_start + offset].height = 42


def _write_legend_sheet(workbook) -> None:
    sheet = workbook["LEGENDA"]
    sheet.delete_rows(1, sheet.max_row)
    sheet.sheet_view.showGridLines = False
    rows = [
        ("Prioridade alta", "Ler primeiro. Combina evidências ou envolve magnitude extrema; não significa rejeição automática."),
        ("Prioridade média", "Requer confirmação, mas tem menos evidências simultâneas ou menor magnitude."),
        ("Prioridade baixa", "Diferença pequena, útil principalmente para confirmar semântica e arredondamento."),
        ("Ocorrência apoiada", "Havia chuva em vizinha ou eco RGB. Isso não valida quantitativamente a magnitude."),
        ("Sem apoio", "Não foi encontrada chuva vizinha nem eco RGB entre as fontes disponíveis."),
        ("Contexto insuficiente", "Faltou radar alinhado ou quantidade mínima de vizinhos para concluir."),
        ("Parecer: válida", "A observação é meteorologicamente plausível e a semântica dos acumulados está correta."),
        ("Parecer: problema de acumulação", "O valor parece decorrer de reset, propagação ou interpretação incorreta dos acumulados."),
        ("Parecer: falha de sensor/telemetria", "Há indício de medição, transmissão, calibração ou manutenção incorreta."),
        ("Parecer: inconclusivo", "Os registros disponíveis não permitem decidir."),
    ]
    sheet.append(["Termo", "Interpretação"])
    for row in rows:
        sheet.append(row)
    _style_tabular_sheet(sheet, priority_column=None)
    sheet.column_dimensions["A"].width = 35
    sheet.column_dimensions["B"].width = 105


def review_questions() -> list[str]:
    return [
        "Os campos m05, m10, m15 e h01 são acumulados móveis encerrados no horário informado? O horário representa o fim do intervalo?",
        "Os valores de 417,2 mm/15 min em Barra/Barrinha e 260,4 mm/15 min em Piedade podem representar códigos, contadores acumulados, overflow ou outra convenção operacional?",
        "Há registro de chuva real, manutenção, calibração, troca de equipamento ou falha de telemetria nos períodos destacados?",
        "É operacionalmente plausível um acumulado muito alto ser seguido, poucos minutos depois, por acumulados iguais ou próximos de zero?",
        "Quando há eco no radar ou chuva fraca em uma estação vizinha, mas a magnitude da candidata é muito maior, qual critério o AlertaRio usa para validar ou invalidar a leitura?",
        "O limite atual de 100 mm/15 min para mascarar automaticamente uma observação é meteorologicamente e operacionalmente defensável? Qual limite ou regra seria preferível?",
        "Além de -99,99, existem outros valores sentinela ou padrões conhecidos que representem ausência, falha ou leitura ainda não consolidada?",
        "Para cada episódio, o parecer deve ser: observação válida, problema de acumulação, falha de sensor/telemetria ou inconclusivo?",
    ]


def write_questions(path: Path, priority_episodes: pd.DataFrame) -> None:
    lines = [
        "# Perguntas para revisão meteorológica do AlertaRio (2026)",
        "",
        "A planilha anexa contém casos sinalizados automaticamente. Eles não foram declarados erros nem removidos do dataset com base nesta auditoria.",
        "",
        "## Perguntas",
        "",
    ]
    lines.extend(f"{index}. {question}" for index, question in enumerate(review_questions(), 1))
    lines.extend([
        "",
        "## Casos para verificar primeiro",
        "",
        "| Prioridade | Estação | Início local | Fim local | Máximo m15 | Motivo |",
        "|---|---|---|---|---:|---|",
    ])
    for row in priority_episodes.to_dict("records"):
        lines.append(
            f"| {row['Prioridade para revisao']} | {row['Estacao']} | "
            f"{row['Inicio - horario local']} | {row['Fim - horario local']} | "
            f"{row['Maximo m15 (mm/15 min)']:.1f} | {row['Motivo da revisao']} |"
        )
    lines.extend([
        "",
        "## Forma sugerida de resposta",
        "",
        "Preencher na aba `TODOS OS EPISÓDIOS` uma destas categorias para cada linha:",
        "",
        "- observação válida;",
        "- problema de acumulação;",
        "- falha de sensor ou telemetria;",
        "- inconclusivo.",
        "",
        "Sempre que possível, registrar a fonte consultada e qualquer informação sobre manutenção ou ocorrência meteorológica.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_workbook(
    episodes_path: Path,
    observations_path: Path,
    output_workbook: Path,
    output_questions: Path,
    *,
    timezone: str = "America/Sao_Paulo",
    extreme_threshold_mm15: float = 100.0,
    spatial_screening_threshold_mm15: float = 6.25,
    overwrite: bool = False,
) -> None:
    ZoneInfo(timezone)
    for path in (output_workbook, output_questions):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} ja existe; use --overwrite.")
    episodes_raw = pd.read_csv(episodes_path)
    observations_raw = pd.read_parquet(observations_path)
    episodes = prepare_episode_sheet(
        episodes_raw,
        timezone=timezone,
        extreme_threshold_mm15=extreme_threshold_mm15,
        spatial_screening_threshold_mm15=spatial_screening_threshold_mm15,
    )
    episode_priorities = episodes[
        ["ID da estacao", "Episodio", "Prioridade para revisao"]
    ].rename(columns={
        "ID da estacao": "station_id",
        "Episodio": "episode_number",
    })
    episodes_for_lookup = episodes_raw.merge(
        episode_priorities,
        on=["station_id", "episode_number"],
        how="left",
        validate="one_to_one",
    )
    observations = prepare_observation_sheet(
        observations_raw,
        episodes_for_lookup,
        timezone=timezone,
    )
    priority_episodes = episodes.loc[episodes["Prioridade para revisao"].eq("Alta")].copy()

    output_workbook.parent.mkdir(parents=True, exist_ok=True)
    output_questions.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_workbook, engine="openpyxl") as writer:
        pd.DataFrame([[""]]).to_excel(writer, sheet_name="LEIA-ME", header=False, index=False)
        priority_episodes.rename(columns=DISPLAY_COLUMN_NAMES).to_excel(
            writer, sheet_name="PRIORITÁRIOS", index=False
        )
        episodes.rename(columns=DISPLAY_COLUMN_NAMES).to_excel(
            writer, sheet_name="TODOS OS EPISÓDIOS", index=False
        )
        observations.rename(columns=DISPLAY_COLUMN_NAMES).to_excel(
            writer, sheet_name="OBSERVAÇÕES", index=False
        )
        pd.DataFrame([[""]]).to_excel(writer, sheet_name="LEGENDA", header=False, index=False)

    workbook = load_workbook(output_workbook)
    _write_readme_sheet(
        workbook,
        episode_count=len(episodes),
        priority_count=len(priority_episodes),
        observation_count=len(observations),
        timezone=timezone,
    )
    for sheet_name in ("PRIORITÁRIOS", "TODOS OS EPISÓDIOS", "OBSERVAÇÕES"):
        _style_tabular_sheet(workbook[sheet_name])
    _write_legend_sheet(workbook)
    workbook.properties.title = "Revisão meteorológica AlertaRio 2026"
    workbook.properties.subject = "Candidatos de controle de qualidade temporal e espacial"
    workbook.properties.creator = "Projeto RioNowcast"
    workbook.save(output_workbook)
    write_questions(output_questions, priority_episodes)


def main() -> None:
    args = parse_args()
    export_workbook(
        args.episodes,
        args.observations,
        args.output_workbook,
        args.output_questions,
        timezone=args.timezone,
        extreme_threshold_mm15=args.extreme_threshold_mm15,
        spatial_screening_threshold_mm15=args.spatial_screening_threshold_mm15,
        overwrite=args.overwrite,
    )
    print(f"Planilha salva em {args.output_workbook}")
    print(f"Perguntas salvas em {args.output_questions}")


if __name__ == "__main__":
    main()
