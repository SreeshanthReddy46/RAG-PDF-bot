import shutil
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

from src.config import settings
from src.ingestion import ingest_pdf
from src.rag_engine import build_conversational_rag_chain

cli = typer.Typer(help="CLI Assistant for PDF ingestion and interactive chat.")
console = Console()

@cli.command()
def upload(pdf_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Path to PDF")):
    """Copies the PDF to the uploads folder and indexes it in ChromaDB."""
    if pdf_path.suffix.lower() != ".pdf":
        console.print("[red]Error: Only .pdf files are supported![/red]")
        raise typer.Exit(code=1)

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = settings.UPLOAD_DIR / pdf_path.name
    if pdf_path.resolve() != dest_path.resolve():
        shutil.copy2(pdf_path, dest_path)

    with console.status(f"[bold cyan]Processing and indexing {pdf_path.name}...[/bold cyan]"):
        stats = ingest_pdf(dest_path)

    console.print(Panel(
        f"[green] Successfully ingested:[/green] [bold]{stats['filename']}[/bold]\n"
        f"Pages: [yellow]{stats['total_pages']}[/yellow] | chunks: [yellow]{stats['total_chunks']}[/yellow]",
        title="Ingestion complete!"
    ))
@cli.command()
def chat(session_id: str = typer.Option("cli_session", help="Session ID for conversation history")):
    """Starts an interactive terminal chat with your indexed documents."""
    console.print(Panel(
        "[bold cyan]RAG Document Chat[/bold cyan]\n"
        "Type your question and press enter. Type [bold red]'exit[/bold red] to quit.",
        title="Session active"
    ))

    chain = build_conversational_rag_chain()

    while True:
        try:
            query = console.input("[bold blue]You >[/bold blue]")
            if query.strip().lower() in ("exit", "quit", "q"):
                console.print("[yellow]Closing session. Good Bye![/yellow]")
                break
            with console.status("[italic green]Thinking...[/italic green]"):
                result = chain.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": session_id}}
                )
            console.print(f"\n[bold green]Assistant:[/bold green]\n{result['answer']}\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Session aborted.[/yellow]")
            break

if __name__ == "__main__":
    cli()
