from pathlib import Path

import click
import open3d as o3d
from joblib import Parallel, delayed


def convert(input_path, output_path, voxel_size):
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    o3d.io.write_voxel_grid(str(output_path), vox)


@click.command()
@click.option("--input-dir", "-i", required=True)
@click.option("--output-dir", "-o", required=True)
@click.option("--voxel-size", "-s", type=float, default=1.0)
@click.option("--n-jobs", "-n", type=int, default=-1)
@click.option("--overwrite", "-w", is_flag=True)
@click.option("--dryrun", "-d", is_flag=True)
def main(input_dir, output_dir, voxel_size, n_jobs, overwrite, dryrun):

    input_paths = list(Path(input_dir).rglob("*.*"))

    input_output_pairs = [
        (
            input_path,
            Path(output_dir)
            .joinpath(input_path.relative_to(input_dir))
            .with_suffix(".ply"),
        )
        for input_path in input_paths
    ]

    if not overwrite:
        input_output_pairs = [
            (input_path, output_path)
            for input_path, output_path in input_output_pairs
            if not output_path.exists()
        ]

    dirs = set([path.parent for _, path in input_output_pairs])
    for directory in dirs:
        if not dryrun:
            directory.mkdir(parents=True, exist_ok=True)

    kwargs_list = [
        {"input_path": input_path, "output_path": output_path, "voxel_size": voxel_size}
        for input_path, output_path in input_output_pairs
    ]

    if dryrun:
        for input_path, output_path in input_output_pairs:
            print(f"{input_path} -> {output_path}")

    else:
        Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(convert)(**kwargs) for kwargs in kwargs_list
        )


if __name__ == "__main__":
    main()
