# convert3d

- Currently only wavefront `obj` to `ply` voxel is supported.
- It should be simple to add support for more formats if necessary.

## Todo

- [x] Check if pip install from github works.
- [ ] Verify if `obj` to `ply` conversion matches current workflow.

Install with

    pip install -e "git+https://github.com/buganart/BUGAN.git#egg=convert3d&subdirectory=convert3d"

Or run

    pip install -e .

From this directory for an editable installation that reflects changes made to the local files.

### Decimating meshes

Make sure `meshlab` is installed, then

     decimate -j 8 --in-dir my-data-tir --target-face-num 1000 --out-dir my-out-dir

Or run `decimate --help` to see all the options.


