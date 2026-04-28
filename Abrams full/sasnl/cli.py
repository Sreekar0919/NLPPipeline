from __future__ import annotations

import typer
from rich import print

from sasnl.config import PipelineConfig
from sasnl.pipeline import PipelineOrchestrator


def process(
    input_source: str = typer.Option(..., "--input-source", help="Path to audio file OR transcript (JSON/text/file path)"),
    student_id: str = typer.Option(..., help="Student identifier"),
    session_type: str = typer.Option("SCS", help="Assessment type (SCS or BOSC)"),
    input_type: str = typer.Option("auto", help="Input type: 'audio', 'transcript', or 'auto' (default)"),
    strict_mode: bool = typer.Option(True, "--strict/--no-strict", help="Enable hard failures for missing deps/Bedrock"),
    enable_t3: bool = typer.Option(False, "--enable-t3/--disable-t3", help="Enable research tier agents"),
) -> None:
    cfg = PipelineConfig(strict_mode=strict_mode, enable_t3_research_agents=enable_t3)
    pipeline = PipelineOrchestrator(config=cfg)
    out = pipeline.run(
        input_source=input_source,
        student_id=student_id,
        session_type=session_type,
        input_type=input_type,
    )
    print(f"[green]Session output written:[/green] {out}")


app = typer.Typer(help="SASNL Autism Speaker Study Tool")
app.command()(process)


if __name__ == "__main__":
    app()
