#!/usr/bin/env python
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

from joblib import Parallel, delayed
import click

# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script


def create_filter_script(filter_file_path, target_num_face):
    filter_script_mlx = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
    <filter name="Simplification: Quadric Edge Collapse Decimation">
    <Param type="RichInt" value="{target_num_face}" name="TargetFaceNum"/>
    <Param type="RichFloat" value="0" name="TargetPerc"/>
    <Param type="RichFloat" value="0.3" name="QualityThr"/>
    </filter>
    </FilterScript>
    """
    with open(filter_file_path, "w") as f:
        f.write(filter_script_mlx)


def reduce_faces(in_file, out_file, filter_file_path):
    command = [
        "meshlabserver",
        "-i",
        str(in_file),
        "-s",
        str(filter_file_path),
        "-o",
        str(out_file),
    ]
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        )
    except subprocess.CalledProcessError as exc:
        print(f"Return code: {exc.returncode}\n", exc.output)
        raise


def simplify(in_mesh, out_name, num_iterations, filter_file_path):
    with tempfile.TemporaryDirectory() as tmp_folder_name:

        tmp_dir = Path(tmp_folder_name)

        for it in range(num_iterations):
            out_mesh = tmp_dir / f"it_{it}.obj"
            reduce_faces(in_mesh, out_mesh, filter_file_path)
            in_mesh = out_mesh

        shutil.copy(out_mesh, out_name)


def process_dir(input_dir, output_dir, num_iterations, filter_file_path):
    in_paths = list(Path(input_dir).rglob("*.obj"))
    out_paths = [
        Path(output_dir) / in_path.relative_to(input_dir) for in_path in in_paths
    ]
    for parent in set([out.parent for out in out_paths]):
        parent.mkdir(exist_ok=True, parents=True)

    for in_path, out_path in zip(in_paths, out_paths):
        print(in_path, out_path)

    Parallel(n_jobs=-1, verbose=20, backend='threading')(
        delayed(simplify)(in_path, out_path, num_iterations, filter_file_path)
        for in_path, out_path in zip(in_paths, out_paths)
    )


@click.command()
@click.option("--in-mesh", "-i")
@click.option("--out-name", "-o")
@click.option("--in-dir")
@click.option("--out-dir")
@click.option("--num-iterations", "-n", default=1, type=int)
@click.option("--target-num-face", "-f", default=1000, type=int)
def main(in_dir, out_dir, in_mesh, out_name, num_iterations, target_num_face):


    with tempfile.TemporaryDirectory() as filter_file_dir:
        filter_file_path = Path(filter_file_dir) / "filter.mlx"
        create_filter_script(filter_file_path, target_num_face)

        if in_dir and out_dir:
            process_dir(in_dir, out_dir, num_iterations, filter_file_path)
        else:
            simplify(in_mesh, out_name, num_iterations, filter_file_path)


if __name__ == "__main__":
    main()
