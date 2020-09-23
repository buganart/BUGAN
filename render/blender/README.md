# Scripted rendering with Blender

## Dependencies

- Install `blender`.

## Usage

    blender --background -noaudio --python render.py -- --input /path/to/mesh.obj --output renders

Render with CYCLES engine (for GPU rendering)

    blender --background -noaudio -E CYCLES --python render.py -- --input /path/to/mesh.obj --output renders


## Make collage of renders from obj files

``` sh
# Pull a wandb run (if desired)
wandb pull -p tree-gan -e bugan mu19vac7

# Render images with blender (Note: CYCLES looks too dark for our generated meshes)
blender --background -noaudio --python render.py -- --input /path/to/obj --output my-renders

# Zeropad the step numbers to make it sortable
./resort_wandb_names.py my-renders

# Tile with 5 columns (requires imagemagick to be installed)
montage -geometry 256x256 -tile 5x my-renders/* collage.jpg
```
