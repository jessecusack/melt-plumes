"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Melt Plumes."""


if __name__ == "__main__":
    main(prog_name="melt-plumes")  # pragma: no cover
