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

Run

    convert3d --help

To show usage.

Convert a directory of wavefron mesh files to ply pointcloud files:

    convert3d -i /my/obj/files -o /my/ply/files
