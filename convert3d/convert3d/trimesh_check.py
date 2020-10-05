#!/usr/bin/env python3
import click
import trimesh
import numpy as np


def mesh2arrayCentered(mesh, voxel_size=1, array_length=32):
    # given array length 64, voxel size 2, then output array size is [128,128,128]
    array_size = np.ceil(
        np.array([array_length, array_length, array_length]) / voxel_size
    ).astype(int)
    vox_array = np.zeros(
        array_size, dtype=bool
    )  # tanh: voxel representation [-1,1], sigmoid: [0,1]
    # scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(
        trimesh.transformations.scale_matrix((array_length - 1) / max_length)
    )  # now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  # max voxel array length = array_length / voxel_size

    # find indices in the v.matrix to center it in vox_array
    indices = ((array_size - v.matrix.shape) / 2).astype(int)
    vox_array[
        indices[0] : indices[0] + v.matrix.shape[0],
        indices[1] : indices[1] + v.matrix.shape[1],
        indices[2] : indices[2] + v.matrix.shape[2],
    ] = v.matrix

    return vox_array


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--task", "-t", type=click.Choice(["check", "voxelize"]))
def cli(input, task):
    print(f"Loading {input}")
    mesh = trimesh.load(input, force="mesh")
    if task == "voxelize":
        array = mesh2arrayCentered(mesh)
        voxelmesh = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(array)
        ).marching_cubes
        # voxelmesh.show()
        print("ok")


main = cli

if __name__ == "__main__":
    cli()
