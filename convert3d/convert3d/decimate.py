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


def create_filter_script(filter_file_path, target_face_num):
    filter_script_mlx = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
    <filter name="Simplification: Quadric Edge Collapse Decimation">
    <Param type="RichInt" value="{target_face_num}" name="TargetFaceNum"/>
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
            try:
                reduce_faces(in_mesh, out_mesh, filter_file_path)
            except Exception:
                print(f"Mesh {in_mesh} failed.")
                return
            in_mesh = out_mesh

        shutil.copy(out_mesh, out_name)


def process_dir(input_dir, output_dir, num_iterations, filter_file_path, n_jobs):

    in_paths = [
        p for p in Path(input_dir).rglob("*.*") if p.name.lower().endswith(".obj")
    ]

    out_paths = [
        Path(output_dir) / in_path.relative_to(input_dir) for in_path in in_paths
    ]
    for parent in set([out.parent for out in out_paths]):
        parent.mkdir(exist_ok=True, parents=True)

    for in_path, out_path in zip(in_paths, out_paths):
        print(in_path, out_path)

    Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(simplify)(in_path, out_path, num_iterations, filter_file_path)
        for in_path, out_path in zip(in_paths, out_paths)
    )


@click.command()
@click.option("--in-mesh", "-i")
@click.option("--out-name", "-o")
@click.option("--in-dir")
@click.option("--out-dir")
@click.option("--num-iterations", "-n", default=1, type=int)
@click.option("--target-face-num", "-f", default=1000, type=int)
@click.option("--n-jobs", "-j", default=-1, type=int)
def main(in_dir, out_dir, in_mesh, out_name, num_iterations, target_face_num, n_jobs):

    with tempfile.TemporaryDirectory() as filter_file_dir:
        filter_file_path = Path(filter_file_dir) / "filter.mlx"
        create_filter_script(filter_file_path, target_face_num)

        if in_dir and out_dir:
            process_dir(in_dir, out_dir, num_iterations, filter_file_path, n_jobs)
        else:
            simplify(in_mesh, out_name, num_iterations, filter_file_path)


if __name__ == "__main__":
    main()
