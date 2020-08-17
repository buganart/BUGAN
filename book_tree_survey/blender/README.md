# Scripted rendering with Blender

## Dependencies

- Install `blender`.

## Usage

    blender --background --python render.py -- --input /path/to/mesh.obj --output renders

Render with CYCLES engine (for GPU rendering)

    blender --background -E CYCLES --python render.py -- --input /path/to/mesh.obj --output renders
