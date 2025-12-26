#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress

console = Console()

def run_vivado_tcl(tcl_file: Path, **params):
    """Run a Vivado TCL script with parameters."""
    cmd = ['vivado', '-mode', 'batch', '-nojournal', '-nolog', '-source', str(tcl_file)]
    for k, v in params.items():
        cmd.extend(['-tclargs', f'-{k}', str(v)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Error running Vivado:[/red]\n{result.stderr}")
        sys.exit(1)
    return result.stdout

@click.command()
@click.option('--part', default='xc7a35tcpg236-1', help='FPGA part number')
@click.option('--top', default='top', help='Top module name')
def build(part: str, top: str):
    """Build FPGA bitstream using Vivado in batch mode."""
    project_root = Path(__file__).parent.parent
    build_dir = project_root / 'build'
    scripts_dir = project_root / 'scripts' / 'vivado'
    
    # Ensure directories exist
    build_dir.mkdir(exist_ok=True)
    
    with Progress() as progress:
        task = progress.add_task("[green]Building project...", total=3)
        
        # Run synthesis
        console.print("\n[bold blue]Running synthesis...[/bold blue]")
        run_vivado_tcl(
            scripts_dir / 'synth.tcl',
            part=part,
            top=top,
            project_dir=build_dir,
            rtl_dir=project_root / 'rtl',
            constraints_dir=project_root / 'constraints'
        )
        progress.update(task, advance=1)
        
        # Run implementation
        console.print("\n[bold blue]Running implementation...[/bold blue]")
        run_vivado_tcl(
            scripts_dir / 'impl.tcl',
            project_dir=build_dir
        )
        progress.update(task, advance=1)
        
        # Generate bitstream
        console.print("\n[bold blue]Generating bitstream...[/bold blue]")
        run_vivado_tcl(
            scripts_dir / 'bitstream.tcl',
            project_dir=build_dir
        )
        progress.update(task, advance=1)
    
    console.print("\n[bold green]Build completed successfully![/bold green]")
    console.print(f"Bitstream available at: {build_dir}/project.bit")

if __name__ == '__main__':
    build() 