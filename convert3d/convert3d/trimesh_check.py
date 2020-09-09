#!/usr/bin/env python3
import click
import trimesh


@click.command()
@click.option("--input", "-i", required=True)
def main(input):
    trimesh.load(input, force="mesh")
    print("Ok")


if __name__ == "__main__":
    main()
